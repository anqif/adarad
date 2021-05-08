import numpy as np
import matplotlib.pyplot as plt
import cvxpy
from cvxpy import *
from cvxpy.settings import SOLUTION_PRESENT

from adarad.optimization import ccp_solve
from adarad.optimization.dose_init import *
from adarad.medicine.prognosis import health_prog_act
from adarad.utilities.beam_utils import line_integral_mat
from adarad.visualization.plot_funcs import plot_treatment, plot_health

from adarad.tests.base_test import BaseTest
from examples.utilities.simple_utils import *

def form_step_xy(x, y, buf = 0, shift = 0):
    x_shift = x - shift
    x_buf = np.zeros(x_shift.shape[0] + 2)
    x_buf[0] = x_shift[0] - buf
    x_buf[-1] = x_shift[-1] + buf
    x_buf[1:-1] = x_shift

    y_buf = np.zeros(y.shape[0] + 2)
    y_buf[0] = y[0]
    y_buf[-1] = y[-1]
    y_buf[1:-1] = y

    return x_buf, y_buf

class TestDoseInit(BaseTest):
    """Unit tests for AdaRad dose initialization functions."""

    def setUp(self):
        np.random.seed(1)

        # Problem data.
        T = 20  # Length of treatment.
        n_grid = 1000
        offset = 5  # Displacement between beams (pixels).
        n_angle = 20  # Number of angles.
        n_bundle = 50  # Number of beams per angle.
        n = n_angle * n_bundle  # Total number of beams.
        self.t_s = 0  # Static session.

        # Anatomical structures.
        x_grid, y_grid, regions = simple_structures(n_grid, n_grid)
        struct_kw = simple_colormap(one_idx=True)
        K = np.unique(regions).size  # Number of structures.

        A, angles, offs_vec = line_integral_mat(regions, angles=n_angle, n_bundle=n_bundle, offset=offset)
        A = A / n_grid
        self.A_list = T * [A]

        self.alpha = np.array(T * [[0.01, 0.50, 0.25, 0.15, 0.005]])
        self.beta = np.array(T * [[0.001, 0.05, 0.025, 0.015, 0.0005]])
        self.gamma = np.array(T * [[0.05, 0, 0, 0, 0]])
        self.h_init = np.array([1] + (K - 1) * [0])

        is_target = np.array([True] + (K - 1) * [False])
        num_ptv = np.sum(is_target)
        num_oar = K - num_ptv

        # Health prognosis.
        prop_cycle = plt.rcParams['axes.prop_cycle']
        self.colors = prop_cycle.by_key()['color']
        h_prog = health_prog_act(self.h_init, T, gamma=self.gamma)
        self.h_curves = [{"h": h_prog, "label": "Untreated", "kwargs": {"color": self.colors[1]}}]

        # Beam constraints.
        beam_upper = np.full((T, n), 1.0)

        # Dose constraints.
        dose_lower = np.zeros((T, K))
        dose_upper = np.full((T, K), 20)

        # Health constraints.
        health_lower = np.full((T, K), -np.inf)
        health_upper = np.full((T, K), np.inf)
        health_lower[:, 1] = -1.0  # Lower bound on OARs.
        health_lower[:, 2] = -2.0
        health_lower[:, 3] = -2.0
        health_lower[:, 4] = -3.0
        health_upper[:15, 0] = 2.0  # Upper bound on PTV for t = 1,...,15.
        health_upper[15:, 0] = 0.05  # Upper bound on PTV for t = 16,...,20.

        self.patient_rx = {"is_target": is_target,
                          "dose_goal": np.zeros((T, K)),
                          "dose_weights": np.array((K - 1) * [1] + [0.25]),
                          "health_goal": np.zeros((T, K)),
                          "health_weights": [np.array([0] + (K - 1) * [0.25]), np.array([1] + (K - 1) * [0])],
                          "beam_constrs": {"upper": beam_upper},
                          "dose_constrs": {"lower": dose_lower, "upper": dose_upper},
                          "health_constrs": {"lower": health_lower, "upper": health_upper}}

    def test_static(self):
        # Problem constants.
        is_target = self.patient_rx["is_target"]
        beam_upper = self.patient_rx["beam_constrs"]["upper"]
        dose_lower = self.patient_rx["dose_constrs"]["lower"]
        dose_upper = self.patient_rx["dose_constrs"]["upper"]
        health_lower = self.patient_rx["health_constrs"]["lower"]
        health_upper = self.patient_rx["health_constrs"]["upper"]

        T, K = self.patient_rx["dose_goal"].shape
        n = self.patient_rx["beam_constrs"]["upper"].shape[1]

        # Stage 1: Static beam problem.
        # Define variables.
        b = Variable((n,), nonneg=True)
        d = self.A_list[self.t_s] @ b

        h_lin = self.h_init - multiply(self.alpha[self.t_s], d) + self.gamma[self.t_s]
        h_quad = self.h_init - multiply(self.alpha[self.t_s], d) - multiply(self.beta[self.t_s], square(d)) + self.gamma[self.t_s]
        h_ptv = h_lin[is_target]
        h_oar = h_quad[~is_target]
        h = multiply(h_lin, is_target) + multiply(h_quad, ~is_target)

        # Form objective.
        d_penalty = sum_squares(d[:-1]) + 0.25 * square(d[-1])
        h_penalty_ptv = sum(pos(h_ptv))
        h_penalty_oar = 0.25 * sum(neg(h_oar))
        h_penalty = h_penalty_ptv + h_penalty_oar

        h_lo_slack_weight = 0.25
        h_lo_slack = Variable(h_oar.shape, nonneg=True)
        s_lo_penalty = h_lo_slack_weight * sum(h_lo_slack)

        s_penalty = s_lo_penalty
        obj = d_penalty + h_penalty + s_penalty

        # Additional constraints.
        constrs = [b <= np.sum(beam_upper, axis=0), h_ptv <= health_upper[-1, 0], h_oar >= health_lower[-1, 1:] - h_lo_slack]

        # Solve problem.
        print("Stage 1: Solving problem...")
        prob_1 = Problem(Minimize(obj), constrs)
        prob_1.solve(solver="MOSEK")
        if prob_1.status not in SOLUTION_PRESENT:
            raise RuntimeError("Stage 1: Solver failed with status {0}".format(prob_1.status))

        # Save results.
        b_static = b.value  # Save optimal static beams for stage 2.
        d_static = np.vstack([self.A_list[t] @ b_static for t in range(T)])
        d_stage_1 = d.value
        # h_stage_1 = h.value
        h_stage_1 = self.h_init - self.alpha[self.t_s] * d_stage_1 - self.beta[self.t_s] * d_stage_1 ** 2 + self.gamma[self.t_s]

        print("Stage 1 Results")
        print("Objective:", prob_1.value)
        print("Optimal Beam (Max):", np.max(b_static))
        print("Optimal Dose:", d_stage_1)
        print("Optimal Health:", h_stage_1)
        print("Solve Time:", prob_1.solver_stats.solve_time)

        # Compare with AdaRad package.
        prob_1_ada, b_1_ada, h_1_ada, d_1_ada, h_actual_1_ada, h_slack_1_ada = \
            build_stat_init_prob(self.A_list, self.alpha, self.beta, self.gamma, self.h_init, self.patient_rx,
                                 t_static = self.t_s, slack_oar_weight = h_lo_slack_weight)
        prob_1_ada.solve(solver = "MOSEK")
        if prob_1_ada.status not in SOLUTION_PRESENT:
            raise RuntimeError("AdaRad Stage 1: Solver failed with status {0}".format(prob_1_ada.status))

        print("Compare with AdaRad")
        print("Difference in Objective:", np.abs(prob_1.value - prob_1_ada.value))
        print("Normed Difference in Beam:", np.linalg.norm(b_static - b_1_ada.value))
        print("Normed Difference in Dose:", np.linalg.norm(d_stage_1 - d_1_ada.value))
        print("Normed Difference in Health:", np.linalg.norm(h.value - h_1_ada.value))
        print("Normed Difference in Health Slack:", np.linalg.norm(h_lo_slack.value - h_slack_1_ada[1:].value))
        print("AdaRad Solve Time:", prob_1_ada.solver_stats.solve_time)

        self.assertAlmostEqual(prob_1.value, prob_1_ada.value, places=4)
        self.assertItemsAlmostEqual(b_static, b_1_ada.value, places=2)
        self.assertItemsAlmostEqual(d_stage_1, d_1_ada.value, places=3)
        self.assertItemsAlmostEqual(h.value, h_1_ada.value, places=3)
        self.assertItemsAlmostEqual(h_lo_slack.value, h_slack_1_ada[1:].value, places=3)

        # Plot optimal dose and health per structure.
        xlim_eps = 0.5
        plt.bar(range(K), d_stage_1, width=0.8)
        plt.step(*form_step_xy(np.arange(K), dose_lower[-1, :], buf=0.5), where="mid", lw=1, ls="--", color=self.colors[1])
        plt.step(*form_step_xy(np.arange(K), dose_upper[-1, :], buf=0.5), where="mid", lw=1, ls="--", color=self.colors[1])
        plt.title("Treatment Dose vs. Structure")
        plt.xlim(-xlim_eps, K - 1 + xlim_eps)
        plt.show()

        health_bounds_fin = np.concatenate(([health_upper[-1, 0]], health_lower[-1, 1:]))
        plt.bar(range(K), h_stage_1, width=0.8)
        plt.step(*form_step_xy(np.arange(K), health_bounds_fin, buf=0.5), where="mid", lw=1, ls="--", color=self.colors[1])
        plt.title("Health Status vs. Structure")
        plt.xlim(-xlim_eps, K - 1 + xlim_eps)
        plt.show()

    def test_scale_const_lin(self):
        # Problem constants.
        is_target = self.patient_rx["is_target"]
        beam_upper = self.patient_rx["beam_constrs"]["upper"]
        dose_lower = self.patient_rx["dose_constrs"]["lower"]
        dose_upper = self.patient_rx["dose_constrs"]["upper"]
        health_lower = self.patient_rx["health_constrs"]["lower"]
        health_upper = self.patient_rx["health_constrs"]["upper"]

        T, K = self.patient_rx["dose_goal"].shape
        n = self.patient_rx["beam_constrs"]["upper"].shape[1]
        num_oar = K - 1

        # Stage 1: Static beam problem.
        print("Stage 1: Solving problem...")
        prob_1_ada, b_1_ada, h_1_ada, d_1_ada, h_actual_1_ada, h_slack_1_ada = \
            build_stat_init_prob(self.A_list, self.alpha, self.beta, self.gamma, self.h_init, self.patient_rx,
                                 t_static=self.t_s, slack_oar_weight=0.25)
        prob_1_ada.solve(solver="MOSEK")
        if prob_1_ada.status not in SOLUTION_PRESENT:
            raise RuntimeError("AdaRad Stage 1: Solver failed with status {0}".format(prob_1_ada.status))

        b_static = b_1_ada.value
        # b_static = np.abs(np.random.randn(n))   # Set a random static beam as starting point.
        d_static = np.vstack([self.A_list[t] @ b_static for t in range(T)])

        # Stage 2a: Dynamic scaling problem with constant factor.
        # Define variables.
        u = Variable(nonneg=True)
        b = u * b_static
        d = u * d_static
        h = Variable((T + 1, K))

        # Used in Taylor expansion of PTV health dynamics.
        h_tayl_slack_weight = 1e4
        h_tayl_slack = Variable((T,), nonneg=True)  # Slack in approximation.

        # Form objective.
        d_penalty = square(u) * (sum_squares(d_static[:,:-1]) + 0.25 * sum_squares(d_static[:,-1])).value
        h_penalty = sum(pos(h[1:, 0])) + 0.25 * sum(neg(h[1:, 1:]))
        s_tayl_penalty = h_tayl_slack_weight * sum(h_tayl_slack)

        # Add slack to lower health bounds.
        h_lo_slack_weight = 0.25
        h_lo_slack = Variable((T,num_oar), nonneg=True)
        s_lo_penalty = h_lo_slack_weight * sum(h_lo_slack)

        s_penalty = s_tayl_penalty + s_lo_penalty
        obj = d_penalty + h_penalty + s_penalty

        # Health dynamics.
        constrs = [h[0] == self.h_init]
        for t in range(T):
            # For PTV, use simple linear model (beta_t = 0).
            constrs += [h[t+1,0] == h[t,0] - self.alpha[t,0]*u*d_static[t,0] + self.gamma[t,0] - h_tayl_slack[t]]

            # For OAR, use linear-quadratic model with lossless relaxation.
            constrs += [h[t+1,1:] <= h[t,1:] - u*multiply(self.alpha[t,1:], d_static[t,1:]).value -
                        square(u)*multiply(self.beta[t,1:], square(d_static[t,1:])).value + self.gamma[t,1:]]

        # Additional constraints.
        constrs += [b <= np.min(beam_upper, axis=0), u*d_static <= dose_upper, u*d_static >= dose_lower,
                    h[1:,0] <= health_upper[:,0], h[1:,1:] >= health_lower[:,1:] - h_lo_slack]

        # Warm start.
        u.value = 1

        # Solve problem.
        print("Stage 2: Solving initial problem...")
        prob_2a = Problem(Minimize(obj), constrs)
        prob_2a.solve(solver="MOSEK", warm_start=True)
        if prob_2a.status not in SOLUTION_PRESENT:
            raise RuntimeError("Stage 2 Initialization: Solver failed with status {0}".format(prob_2a.status))

        # Save results.
        u_stage_2_init = u.value
        d_stage_2_init = d.value  # Save optimal doses derived from constant factor for stage 2b.
        # h_stage_2_init = h.value
        h_stage_2_init = health_prog_act(self.h_init, T, self.alpha, self.beta, self.gamma, d_stage_2_init, is_target)
        s_stage_2_init = h_tayl_slack.value

        print("Stage 2 Initialization")
        print("Objective:", prob_2a.value)
        print("Optimal Beam Weight:", u_stage_2_init)
        print("Optimal Beam (Max):", np.max(b.value))
        # print("Optimal Dose:", d_stage_2_init)
        # print("Optimal Health:", h_stage_2_init)
        # print("Optimal Health Slack:", s_stage_2_init)
        print("Solve Time:", prob_2a.solver_stats.solve_time)

        # Compare with AdaRad package.
        prob_2a_ada, u_2a_ada, b_2a_ada, h_2a_ada, d_2a_ada, h_lin_dyn_slack_2a_ada, h_lin_bnd_slack_2a_ada = \
            build_scale_lin_init_prob(self.A_list, self.alpha, self.beta, self.gamma, self.h_init, self.patient_rx, b_static,
                                      use_dyn_slack = True, dyn_slack_weight= h_tayl_slack_weight, use_bnd_slack = True,
                                      bnd_slack_weight= h_lo_slack_weight)
        prob_2a_ada.solve(solver="MOSEK")
        if prob_2a_ada.status not in SOLUTION_PRESENT:
            raise RuntimeError("AdaRad Stage 2a: Solver failed with status {0}".format(prob_2a_ada.status))

        print("Compare with AdaRad")
        print("Difference in Objective:", np.abs(prob_2a.value - prob_2a_ada.value))
        print("Normed Difference in Beam:", np.linalg.norm(u_stage_2_init*b_static - b_2a_ada.value))
        print("Normed Difference in Dose:", np.linalg.norm(d_stage_2_init - d_2a_ada.value))
        print("Normed Difference in Health:", np.linalg.norm(h.value - h_2a_ada.value))
        print("Normed Difference in Health Slack (Dynamics):", np.linalg.norm(s_stage_2_init - h_lin_dyn_slack_2a_ada[:,0].value))
        print("Normed Difference in Health Slack (Bound):", np.linalg.norm(h_lo_slack.value - h_lin_bnd_slack_2a_ada[:,1:].value))
        print("AdaRad Solve Time:", prob_2a_ada.solver_stats.solve_time)

        self.assertAlmostEqual(prob_2a.value, prob_2a_ada.value, places=4)
        self.assertItemsAlmostEqual(u_stage_2_init*b_static, b_2a_ada.value, places=2)
        self.assertItemsAlmostEqual(d_stage_2_init, d_2a_ada.value, places=3)
        self.assertItemsAlmostEqual(h.value, h_2a_ada.value, places=3)
        self.assertItemsAlmostEqual(s_stage_2_init, h_lin_dyn_slack_2a_ada[:,0].value, places=3)
        self.assertItemsAlmostEqual(h_lo_slack.value, h_lin_bnd_slack_2a_ada[:,1:].value, places=3)

        # Plot optimal dose and health over time.
        plot_treatment(d_stage_2_init, stepsize=10, bounds=(dose_lower, dose_upper), title="Treatment Dose vs. Time",
                       color=self.colors[0], one_idx=True)
        plot_health(h_stage_2_init, curves=self.h_curves, stepsize=10, bounds=(health_lower, health_upper),
                    title="Health Status vs. Time", label="Treated", color=self.colors[0], one_idx=True)

    def test_scale_const(self):
        # Problem constants.
        is_target = self.patient_rx["is_target"]
        beam_upper = self.patient_rx["beam_constrs"]["upper"]
        dose_lower = self.patient_rx["dose_constrs"]["lower"]
        dose_upper = self.patient_rx["dose_constrs"]["upper"]
        health_lower = self.patient_rx["health_constrs"]["lower"]
        health_upper = self.patient_rx["health_constrs"]["upper"]

        T, K = self.patient_rx["dose_goal"].shape
        n = self.patient_rx["beam_constrs"]["upper"].shape[1]
        num_oar = K - 1

        # Stage 1: Static beam problem.
        print("Stage 1: Solving problem...")
        prob_1_ada, b_1_ada, h_1_ada, d_1_ada, h_actual_1_ada, h_slack_1_ada = \
            build_stat_init_prob(self.A_list, self.alpha, self.beta, self.gamma, self.h_init, self.patient_rx,
                                 t_static=self.t_s, slack_oar_weight=0.25)
        prob_1_ada.solve(solver="MOSEK")
        if prob_1_ada.status not in SOLUTION_PRESENT:
            raise RuntimeError("AdaRad Stage 1: Solver failed with status {0}".format(prob_1_ada.status))

        b_static = b_1_ada.value
        # b_static = np.abs(np.random.randn(n))  # Set a random static beam as starting point.
        d_static = np.vstack([self.A_list[t] @ b_static for t in range(T)])

        # Stage 2a: Dynamic beam problem with constant scaling factor and linear PTV dynamics (beta_t = 0).
        prob_2a_ada, u_2a_ada, b_2a_ada, h_2a_ada, d_2a_ada, h_lin_dyn_slack_2a_ada, h_lin_bnd_slack_2a_ada = \
            build_scale_lin_init_prob(self.A_list, self.alpha, self.beta, self.gamma, self.h_init, self.patient_rx,
                                      b_static, use_dyn_slack=True, dyn_slack_weight=1e4, use_bnd_slack=True,
                                      bnd_slack_weight=0.25)
        prob_2a_ada.solve(solver="MOSEK")
        if prob_2a_ada.status not in SOLUTION_PRESENT:
            raise RuntimeError("AdaRad Stage 2a: Solver failed with status {0}".format(prob_2a_ada.status))
        d_stage_2_init = d_2a_ada.value

        # Stage 2b: Dynamic beam problem with time-varying scaling factors.
        # Define variables.
        u = Variable((T,), nonneg=True)
        b = vstack([u[t] * b_static for t in range(T)])
        d = vstack([u[t] * d_static[t,:] for t in range(T)])
        h = Variable((T + 1, K))

        # Used in Taylor expansion of PTV health dynamics.
        h_tayl_slack_weight = 1e4
        h_tayl_slack = Variable((T,), nonneg=True)  # Slack in approximation.
        d_parm = Parameter((T,), nonneg=True)

        # Form objective.
        d_penalty = sum_squares(d[:,:-1]) + 0.25 * sum_squares(d[:,-1])
        h_penalty = sum(pos(h[1:,0])) + 0.25 * sum(neg(h[1:,1:]))
        s_tayl_penalty = h_tayl_slack_weight*sum(h_tayl_slack)

        # Add slack to lower health bounds.
        h_lo_slack_weight = 0.25
        h_lo_slack = Variable((T, num_oar), nonneg=True)
        s_lo_penalty = h_lo_slack_weight * sum(h_lo_slack)

        s_penalty = s_tayl_penalty + s_lo_penalty
        obj = d_penalty + h_penalty + s_penalty

        # Health dynamics.
        constrs = [h[0] == self.h_init]
        for t in range(T):
            # For PTV, use first-order Taylor expansion of dose around d_parm.
            constrs += [h[t+1,0] == h[t,0] - self.alpha[t,0]*u[t]*d_static[t,0] -
                        (2*u[t]*d_static[t,0] - d_parm[t])*self.beta[t,0]*d_parm[t] + self.gamma[t,0] - h_tayl_slack[t]]

            # For OAR, use linear-quadratic model with lossless relaxation.
            constrs += [h[t+1,1:] <= h[t,1:] - u[t]*multiply(self.alpha[t,1:], d_static[t,1:]).value -
                        square(u[t])*multiply(self.beta[t,1:], square(d_static[t,1:])).value + self.gamma[t,1:]]

        # Additional constraints.
        constrs += [b <= beam_upper, d <= dose_upper, d >= dose_lower, h[1:,0] <= health_upper[:,0],
                    h[1:,1:] >= health_lower[:,1:] - h_lo_slack]
        prob_2b = Problem(Minimize(obj), constrs)

        # Solve using CCP.
        print("Stage 2: Solving dynamic problem with CCP...")
        ccp_max_iter = 20
        ccp_eps = 1e-3
        ccp_2b_solve_time = 0

        obj_old = np.inf
        d_parm.value = d_stage_2_init[:,0]
        for k in range(ccp_max_iter):
            # Solve linearized problem.
            prob_2b.solve(solver="MOSEK", warm_start=True)
            if prob_2b.status not in SOLUTION_PRESENT:
                raise RuntimeError(
                    "Stage 2 CCP: Solver failed on iteration {0} with status {1}".format(k, prob_2b.status))
            ccp_2b_solve_time += prob_2b.solver_stats.solve_time

            # Terminate if change in objective is small.
            obj_diff = obj_old - prob_2b.value
            print("CCP Iteration {0}, Objective Difference: {1}".format(k, obj_diff))
            if obj_diff <= ccp_eps:
                break

            obj_old = prob_2b.value
            d_parm.value = d.value[:,0]

        # Save results.
        u_stage_2 = u.value
        b_stage_2 = b.value
        d_stage_2 = d.value
        # h_stage_2 = h.value
        h_stage_2 = health_prog_act(self.h_init, T, self.alpha, self.beta, self.gamma, d_stage_2, is_target)
        s_stage_2 = h_tayl_slack.value

        print("Stage 2 Results")
        print("Objective:", prob_2b.value)
        # print("Optimal Beam Weight:", u_stage_2)
        print("Optimal Beam Weight (Median):", np.median(u_stage_2))
        # print("Optimal Dose:", d_stage_2)
        # print("Optimal Health:", h_stage_2)
        # print("Optimal Health Slack:", s_stage_2)
        print("Solve Time:", ccp_2b_solve_time)

        # Compare with AdaRad package.
        prob_2b_ada, u_2b_ada, b_2b_ada, h_2b_ada, d_2b_ada, d_parm_2b_ada, h_dyn_slack_2b_ada, h_bnd_slack_2b_ada = \
            build_scale_init_prob(self.A_list, self.alpha, self.beta, self.gamma, self.h_init, self.patient_rx, b_static,
                                  use_dyn_slack = True, dyn_slack_weight= h_tayl_slack_weight, use_bnd_slack = True,
                                  bnd_slack_weight= h_lo_slack_weight)

        result_2b_ada = ccp_solve(prob_2b_ada, d_2b_ada, d_parm_2b_ada, d_stage_2_init, h_dyn_slack_2b_ada,
                                  max_iter = ccp_max_iter, ccp_eps = ccp_eps, solver = "MOSEK", warm_start = True)
        if result_2b_ada["status"] not in SOLUTION_PRESENT:
            raise RuntimeError("Stage 2b: CCP solve failed with status {0}".format(result_2b_ada["status"]))

        print("Compare with AdaRad")
        print("Difference in Objective:", np.abs(prob_2b.value - prob_2b_ada.value))
        print("Normed Difference in Beam:", np.linalg.norm(b_stage_2 - b_2b_ada.value))
        print("Normed Difference in Dose:", np.linalg.norm(d_stage_2 - d_2b_ada.value))
        print("Normed Difference in Health:", np.linalg.norm(h.value - h_2b_ada.value))
        print("Normed Difference in Health Slack (Dynamics):", np.linalg.norm(s_stage_2 - h_dyn_slack_2b_ada[:,0].value))
        print("Normed Difference in Health Slack (Bound):", np.linalg.norm(h_lo_slack.value - h_bnd_slack_2b_ada[:,1:].value))
        print("AdaRad Solve Time:", result_2b_ada["solve_time"])

        self.assertAlmostEqual(prob_2b.value, prob_2b_ada.value, places=4)
        self.assertItemsAlmostEqual(b_stage_2, b_2b_ada.value, places=2)
        self.assertItemsAlmostEqual(d_stage_2, d_2b_ada.value, places=3)
        self.assertItemsAlmostEqual(h.value, h_2b_ada.value, places=3)
        self.assertItemsAlmostEqual(s_stage_2, h_dyn_slack_2b_ada[:,0].value, places=3)
        self.assertItemsAlmostEqual(h_lo_slack.value, h_bnd_slack_2b_ada[:,1:].value, places=3)

        # Plot optimal dose and health over time.
        plot_treatment(d_stage_2, stepsize=10, bounds=(dose_lower, dose_upper), title="Treatment Dose vs. Time",
                       color=self.colors[0], one_idx=True)
        plot_health(h_stage_2, curves=self.h_curves, stepsize=10, bounds=(health_lower, health_upper),
                    title="Health Status vs. Time", label="Treated", color=self.colors[0], one_idx=True)

    def test_dose_init(self):
        # Problem constants.
        is_target = self.patient_rx["is_target"]
        dose_lower = self.patient_rx["dose_constrs"]["lower"]
        dose_upper = self.patient_rx["dose_constrs"]["upper"]
        health_lower = self.patient_rx["health_constrs"]["lower"]
        health_upper = self.patient_rx["health_constrs"]["upper"]
        T, K = self.patient_rx["dose_goal"].shape

        res_init = dyn_init_dose(self.A_list, self.alpha, self.beta, self.gamma, self.h_init, self.patient_rx,
                                 use_dyn_slack=True, dyn_slack_weight=1e4, solver="MOSEK", init_verbose=True)

        # Plot initial health and dose curves.
        h_equal = health_prog_act(self.h_init, T, self.alpha, self.beta, self.gamma, res_init["doses"], is_target)
        plot_health(h_equal, curves=self.h_curves, stepsize=10, bounds=(health_lower, health_upper),
                    title="Initial Stage: Health Status vs. Time", label="Treated", color=self.colors[0], one_idx=True)
        plot_treatment(res_init["doses"], stepsize=10, bounds=(dose_lower, dose_upper),
                       title="Initial Stage: Treatment Dose vs. Time", one_idx=True)
