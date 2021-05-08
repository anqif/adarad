import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import cvxpy
from cvxpy import *
from cvxpy.settings import SOLUTION_PRESENT

from adarad.medicine.prognosis import health_prog_act
from adarad.optimization.dose_init import dyn_init_dose
from adarad.optimization.admm.quad_admm_funcs import dyn_quad_treat_admm
from adarad.visualization.plot_funcs import plot_health, plot_treatment

from adarad.tests.base_test import BaseTest
from examples.utilities.simple_utils import *

class TestCCP(BaseTest):
    """Unit tests for AdaRad ADMM solution algorithm."""

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

        # ADMM: Dynamic optimal control problem.
        rho = Parameter(pos=True)
        u = Parameter((T, K))

        # Beam subproblems.
        prob_b_list = []
        for t in range(T):
            b_t = Variable((n,), nonneg=True)
            d_t = self.A_list[t] @ b_t
            d_tld_cons_t_parm = Parameter((K,), nonneg=True)

            d_penalty = sum_squares(d_t[:-1]) + 0.25 * square(d_t[-1])
            c_penalty = (rho / 2.0) * sum_squares(d_t - d_tld_cons_t_parm - u[t])
            obj = d_penalty + c_penalty
            constrs = [b_t <= beam_upper[t], d_t <= dose_upper[t], d_t >= dose_lower[t]]

            prob_b = Problem(Minimize(obj), constrs)
            prob_b_dict = {"prob": prob_b, "b": b_t, "d": d_t, "d_tld_cons_parm": d_tld_cons_t_parm}
            prob_b_list.append(prob_b_dict)

        # Health subproblem.
        h = Variable((T + 1, K))
        d_tld = Variable((T, K), nonneg=True)
        d_cons_parm = Parameter((T, K), nonneg=True)
        d_tayl_parm = Parameter((T,), nonneg=True)

        # Used in Taylor expansion of PTV health dynamics.
        h_slack_weight = 1e4
        h_slack = Variable((T,), nonneg=True)  # Slack in approximation.

        h_penalty = sum(pos(h[1:,0])) + 0.25*sum(neg(h[1:,1:]))
        s_penalty = h_slack_weight*sum(h_slack)
        c_penalty = (rho / 2.0)*sum_squares(d_tld - d_cons_parm + u)
        obj = h_penalty + s_penalty + c_penalty

        # Health dynamics.
        constrs = [h[0] == self.h_init]
        for t in range(T):
            # For PTV, use first-order Taylor expansion of dose around d_parm.
            constrs += [h[t+1,0] == h[t,0] - self.alpha[t,0] * d_tld[t,0] - (2*d_tld[t,0] - d_tayl_parm[t])*self.beta[t,0]*d_tayl_parm[t] +
                        self.gamma[t,0] - h_slack[t]]

            # For OAR, use linear-quadratic model with lossless relaxation.
            constrs += [h[t+1,1:] <= h[t,1:] - multiply(self.alpha[t,1:], d_tld[t,1:]) - multiply(self.beta[t,1:], square(d_tld[t, 1:])) + self.gamma[t, 1:]]
        constrs += [d_tld <= dose_upper, d_tld >= dose_lower, h[1:,0] <= health_upper[:,0], h[1:,1:] >= health_lower[:,1:]]

        prob_h = Problem(Minimize(obj), constrs)
        prob_h_dict = {"prob": prob_h, "h": h, "h_slack": h_slack, "d_tld": d_tld, "d_cons_parm": d_cons_parm,
                       "d_tayl_parm": d_tayl_parm}

        # Initialize parameter values.
        rho.value = 5.0
        u.value = np.zeros(u.shape)

        # Solve using ADMM
        admm_max_iter = 500
        eps_abs = 1e-6  # Absolute stopping tolerance.
        eps_rel = 1e-3  # Relative stopping tolerance.
        ccp_max_iter = 15
        ccp_eps = 1e-3

        print("ADMM: Solving dynamic problem...")
        solve_time_cvx = 0
        iters_cvx = 0
        d_tld_var_val = d_init
        d_tld_var_val_old = d_init

        for k in range(admm_max_iter):
            if k % 10 == 0:
                print("ADMM Iteration {0}".format(k))

            d_var_t_val_list = []
            solve_time_list = []

            for t in range(T):
                prob_b_list[t]["d_tld_cons_parm"].value = d_tld_var_val[t]
                prob_b_list[t]["prob"].solve(solver="MOSEK")
                if prob_b_list[t]["prob"].status not in SOLUTION_PRESENT:
                    raise RuntimeError(
                        "ADMM: Solver failed on iteration {0} of session {1} with status {2}".format(k, t, prob_b_list[t]["prob"].status))
                d_var_t_val_list.append(prob_b_list[t]["d"].value)
                solve_time_list.append(prob_b_list[t]["prob"].solver_stats.solve_time)

            d_var_val = np.row_stack(d_var_t_val_list)
            solve_time_cvx += np.max(solve_time_list)  # Take max of all solve times, since subproblems solved in parallel.

            # Solve health subproblem using CCP.
            obj_old = np.inf
            prob_h_dict["d_cons_parm"].value = d_var_val
            prob_h_dict["d_tayl_parm"].value = d_tld_var_val_old[:,0]  # TODO: What dose point should we linearize PTV health dynamics around?

            for l in range(ccp_max_iter):
                # Solve linearized problem.
                prob_h_dict["prob"].solve(solver="MOSEK")
                if prob_h_dict["prob"].status not in SOLUTION_PRESENT:
                    raise RuntimeError(
                        "CCP: Solver failed on ADMM iteration {0}, CCP iteration {1} with status {2}".format(k, l, prob_h_dict["prob"].status))
                solve_time_cvx += prob_h_dict["prob"].solver_stats.solve_time

                # Terminate if change in objective is small.
                obj_diff = obj_old - prob_h_dict["prob"].value
                # print("ADMM CCP Iteration {0}, Objective Difference: {1}".format(l, obj_diff))
                if obj_diff <= ccp_eps:
                    break
                obj_old = prob_h_dict["prob"].value
                prob_h_dict["d_tayl_parm"].value = prob_h_dict["d_tld"].value[:, 0]

            d_tld_var_val_old = d_tld_var_val
            d_tld_var_val = prob_h_dict["d_tld"].value

            # Update dual values.
            u.value = u.value + d_tld_var_val - d_var_val
            iters_cvx = iters_cvx + 1

            # Calculate residuals.
            r_prim = d_var_val - d_tld_var_val
            r_dual = rho.value * (d_tld_var_val - d_tld_var_val_old)

            # Check stopping criteria.
            r_prim_norm = LA.norm(r_prim)
            r_dual_norm = LA.norm(r_dual)
            eps_prim = eps_abs * np.sqrt(T * K) + eps_rel * np.max([LA.norm(d_var_val), LA.norm(d_tld_var_val)])
            eps_dual = eps_abs * np.sqrt(T * K) + eps_rel * LA.norm(u.value)
            if r_prim_norm <= eps_prim and r_dual_norm <= eps_dual:
                break

        # Save results.
        b_t_val_list = [prob_b_list[t]["b"].value for t in range(T)]
        b_cvx = np.row_stack(b_t_val_list)
        d_cvx = (d_var_val + d_tld_var_val) / 2.0
        # h_cvx = prob_h_dict["h"].value
        h_cvx = health_prog_act(self.h_init, T, self.alpha, self.beta, self.gamma, d_cvx, is_target)
        h_slack_cvx = prob_h_dict["h_slack"].value

        # Calculate true objective.
        d_penalty_cvx = np.sum(d_cvx[:,:-1] ** 2) + 0.25*np.sum(d_cvx[:,-1]**2)
        h_penalty_cvx = np.sum(np.maximum(h_cvx[1:,0], 0)) + 0.25*np.sum(np.maximum(-h_cvx[1:,1:], 0))
        s_penalty_cvx = h_slack_weight*np.sum(h_slack_cvx)
        obj_cvx = d_penalty_cvx + h_penalty_cvx + s_penalty_cvx
        solve_time_cvx += res_init["solve_time"]

        print("ADMM Results")
        print("Objective:", obj_cvx)
        # print("Optimal Dose:", d_cvx)
        # print("Optimal Health:", h_cvx)
        # print("Optimal Health Slack:", h_slack_cvx)
        print("Solve Time:", solve_time_cvx)
        print("Iterations:", iters_cvx)

        # Compare with AdaRad package.
        res_ada = dyn_quad_treat_admm(self.A_list, self.alpha, self.beta, self.gamma, self.h_init, self.patient_rx, rho=rho.value,
                                      use_slack=True, slack_weight=h_slack_weight, ccp_max_iter=ccp_max_iter, ccp_eps=ccp_eps,
                                      admm_max_iter=admm_max_iter, solver="MOSEK", eps_abs=eps_abs, eps_rel=eps_rel, admm_verbose=True,
                                      auto_init=True)
        if res_ada["status"] not in SOLUTION_PRESENT:
            raise RuntimeError("AdaRad: ADMM solve failed with status {0}".format(res_ada))

        print("Compare with AdaRad")
        print("Difference in Objective:", np.abs(obj_cvx - res_ada["obj"]))
        print("Normed Difference in Beam:", np.linalg.norm(b_cvx - res_ada["beams"]))
        print("Normed Difference in Dose:", np.linalg.norm(d_cvx - res_ada["doses"]))
        print("Normed Difference in Health:", np.linalg.norm(h_cvx - res_ada["health"]))
        print("AdaRad Solve Time:", res_ada["solve_time"])

        # self.assertAlmostEqual(obj_cvx, res_ada["obj"], places=2)
        # self.assertItemsAlmostEqual(b_cvx, res_ada["beams"], places=2)
        # self.assertItemsAlmostEqual(d_cvx, res_ada["doses"], places=3)
        # self.assertItemsAlmostEqual(h_cvx, res_ada["health"], places=4)

        # Plot optimal dose and health over time.
        plot_health(res_ada["health"], curves=self.h_curves, stepsize=10, bounds=(health_lower, health_upper),
                    title="Health Status vs. Time", label="Treated", color=self.colors[0], one_idx=True)
        plot_treatment(res_ada["doses"], stepsize=10, bounds=(dose_lower, dose_upper), title="Treatment Dose vs. Time",
                       one_idx=True)
