import numpy as np
import matplotlib.pyplot as plt
import cvxpy
from cvxpy import *
from cvxpy.settings import SOLUTION_PRESENT

from adarad.medicine.prognosis import health_prog_act
from adarad.optimization.dose_init import dyn_init_dose
from adarad.optimization.seq_cvx.quad_funcs import dyn_quad_treat
from adarad.utilities.beam_utils import line_integral_mat
from adarad.visualization.plot_funcs import plot_health, plot_treatment

from adarad.tests.base_test import BaseTest
from examples.utilities.simple_utils import *

class TestCCP(BaseTest):
    """Unit tests for AdaRad CCP solution algorithm."""

    def setUp(self):
        np.random.seed(1)
        self.setUpSimpleProblem()

    def test_simple(self):
        # Problem constants.
        is_target = self.patient_rx["is_target"]
        beam_upper = self.patient_rx["beam_constrs"]["upper"]
        dose_lower = self.patient_rx["dose_constrs"]["lower"]
        dose_upper = self.patient_rx["dose_constrs"]["upper"]
        health_lower = self.patient_rx["health_constrs"]["lower"]
        health_upper = self.patient_rx["health_constrs"]["upper"]

        T, K = self.patient_rx["dose_goal"].shape
        n = self.patient_rx["beam_constrs"]["upper"].shape[1]

        # Initialize dose.
        res_init = dyn_init_dose(self.A_list, self.alpha, self.beta, self.gamma, self.h_init, self.patient_rx,
                                 use_dyn_slack=True, dyn_slack_weight=1e4, solver="MOSEK", init_verbose=True)
        d_init = res_init["doses"]

        # Define variables.
        b = Variable((T, n), nonneg=True)
        d = vstack([self.A_list[t] @ b[t] for t in range(T)])
        h = Variable((T + 1, K))

        # Used in Taylor expansion of PTV health dynamics.
        h_slack_weight = 1e4
        h_slack = Variable((T,), nonneg=True)  # Slack in approximation.
        # d_parm = Parameter(d.shape, nonneg=True)   # Dose point around which to linearize.
        d_parm = Parameter((T,), nonneg=True)

        # Form objective.
        d_penalty = sum_squares(d[:, :-1]) + 0.25 * sum_squares(d[:, -1])
        h_penalty = sum(pos(h[1:, 0])) + 0.25 * sum(neg(h[1:, 1:]))
        s_penalty = h_slack_weight * sum(h_slack)
        obj = d_penalty + h_penalty + s_penalty

        # Health dynamics.
        constrs = [h[0] == self.h_init]
        for t in range(T):
            # For PTV, use first-order Taylor expansion of dose around d_parm.
            constrs += [h[t+1,0] == h[t,0] - self.alpha[t,0] * d[t,0] -
                            (2*d[t,0] - d_parm[t])*self.beta[t,0]*d_parm[t] + self.gamma[t,0] - h_slack[t]]

            # For OAR, use linear-quadratic model with lossless relaxation.
            constrs += [h[t+1,1:] <= h[t,1:] - multiply(self.alpha[t,1:], d[t,1:]) -
                            multiply(self.beta[t,1:], square(d[t,1:])) + self.gamma[t,1:]]

        # Additional constraints.
        constrs += [b <= beam_upper, d <= dose_upper, d >= dose_lower, h[1:,0] <= health_upper[:,0],
                        h[1:,1:] >= health_lower[:,1:]]
        prob_cvx = Problem(Minimize(obj), constrs)

        # Solve using CCP.
        print("Main Stage: Solving dynamic problem with CCP...")
        ccp_max_iter = 20
        ccp_eps = 1e-3
        ccp_cvx_solve_time = 0

        obj_old = np.inf
        d_parm.value = d_init[:,0]
        for k in range(ccp_max_iter):
            # Solve linearized problem.
            prob_cvx.solve(solver="MOSEK", warm_start=True)
            if prob_cvx.status not in SOLUTION_PRESENT:
                raise RuntimeError("Main Stage CCP: Solver failed on iteration {0} with status {1}".format(k, prob_cvx.status))
            ccp_cvx_solve_time += prob_cvx.solver_stats.solve_time

            # Terminate if change in objective is small.
            obj_diff = obj_old - prob_cvx.value
            print("CCP Iteration {0}, Objective Difference: {1}".format(k, obj_diff))
            if obj_diff <= ccp_eps:
                break

            obj_old = prob_cvx.value
            d_parm.value = d.value[:,0]

        # Save results.
        b_cvx = b.value
        d_cvx = d.value
        # h_cvx = h.value
        h_cvx = health_prog_act(self.h_init, T, self.alpha, self.beta, self.gamma, d_cvx, is_target)
        s_cvx = h_slack.value

        print("Main Stage Results")
        print("Objective:", prob_cvx.value)
        # print("Optimal Dose:", d_cvx)
        # print("Optimal Health:", h_cvx)
        # print("Optimal Health Slack:", s_cvx)
        print("Solve Time:", ccp_cvx_solve_time)

        # Compare with AdaRad package.
        res_ada = dyn_quad_treat(self.A_list, self.alpha, self.beta, self.gamma, self.h_init, self.patient_rx,
                                 use_slack = True, slack_weight = h_slack_weight, max_iter = ccp_max_iter, solver = "MOSEK",
                                 ccp_verbose = True, auto_init = True)
        if res_ada["status"] not in SOLUTION_PRESENT:
            raise RuntimeError("AdaRad: CCP solve failed with status {0}".format(res_ada))

        print("Compare with AdaRad")
        print("Difference in Objective:", np.abs(prob_cvx.value - res_ada["obj"]))
        print("Normed Difference in Beam:", np.linalg.norm(b_cvx - res_ada["beams"]))
        print("Normed Difference in Dose:", np.linalg.norm(d_cvx - res_ada["doses"]))
        print("Normed Difference in Health:", np.linalg.norm(h_cvx - res_ada["health"]))
        print("AdaRad Solve Time:", res_ada["solve_time"])

        self.assertAlmostEqual(prob_cvx.value, res_ada["obj"], places=2)
        self.assertItemsAlmostEqual(b_cvx, res_ada["beams"], places=2)
        self.assertItemsAlmostEqual(d_cvx, res_ada["doses"], places=3)
        self.assertItemsAlmostEqual(h_cvx, res_ada["health"], places=4)

        # Plot optimal dose and health over time.
        plot_health(res_ada["health"], curves=self.h_curves, stepsize=10, bounds=(health_lower, health_upper),
                    title="Health Status vs. Time", label="Treated", color=self.colors[0], one_idx=True)
        plot_treatment(res_ada["doses"], stepsize=10, bounds=(dose_lower, dose_upper), title="Treatment Dose vs. Time",
                       one_idx=True)
