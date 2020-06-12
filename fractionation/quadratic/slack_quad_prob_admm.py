import numpy as np
import cvxpy
from cvxpy import *
from collections import defaultdict

from fractionation.problem.dyn_prob import dose_penalty, health_penalty, rx_to_constrs
from fractionation.problem.slack_quad_prob import slack_quad_penalty, slack_quad_constrs, rx_to_slack_quad_constrs

def build_dyn_slack_quad_prob_dose(A_list, patient_rx, s_weights = None, s_final = True):
    T_treat = len(A_list)
    K, n = A_list[0].shape
    if patient_rx["dose_goal"].shape != (T_treat, K):
        raise ValueError("dose_goal must have dimensions ({0},{1})".format(T_treat, K))

    # Main variables.
    b = Variable((T_treat, n), nonneg=True, name="beams")  # Beams.
    d = vstack([A_list[t] * b[t] for t in range(T_treat)])  # Doses.

    # Dose penalty function.
    obj = sum([dose_penalty(d[t], patient_rx["dose_goal"][t], patient_rx["dose_weights"]) for t in range(T_treat)])
    constrs = []

    # Additional beam constraints.
    s_vars = dict()
    if "beam_constrs" in patient_rx:
    	# b_slack_lo = Variable((T_treat, n), pos=True, name="beam lower slack")  # Slack for beam constraints.
        # b_slack_hi = Variable((T_treat, n), pos=True, name="beam upper slack")
        # s_vars["beam"] = [b_slack_lo, b_slack_hi]
        # constrs += rx_to_slack_constrs(b, patient_rx["beam_constrs"], s_vars["beam"])
        constrs += rx_to_constrs(b, patient_rx["beam_constrs"])

    # Additional dose constraints.
    if "dose_constrs" in patient_rx:
    	# d_slack_lo = Variable((T_treat, K), nonneg=True, name="dose lower slack")  # Slack for dose constraints.
        # d_slack_hi = Variable((T_treat, K), nonneg=True, name="dose upper slack")
        # s_vars["dose"] = [d_slack_lo, d_slack_hi]
        # constrs += rx_to_slack_constrs(d, patient_rx["dose_constrs"], s_vars["dose"])
        constrs += rx_to_constrs(d, patient_rx["dose_constrs"])

    # Final problem.
    obj += slack_quad_penalty(s_vars, patient_rx["is_target"], s_weights)
    constrs += slack_quad_constrs(s_vars, patient_rx["is_target"], s_final)
    prob = Problem(Minimize(obj), constrs)
    return prob, b, d, s_vars

def build_dyn_slack_quad_prob_dose_period(A, patient_rx, s_weights = None, s_final = True):
	K, n = A.shape

	# Define variables for period.
    b_t = Variable(n, nonneg=True, name="beams")  # Beams.
    d_t = A * b_t

    # Dose penalty current period.
    obj = dose_penalty(d_t, patient_rx["dose_goal"], patient_rx["dose_weights"])
    constrs = []

    # Additional beam constraints in period.
    s_t_vars = dict()
    if "beam_constrs" in patient_rx:
    	# b_t_slack_lo = Variable(n, pos=True, name="beam lower slack")  # Slack for beam constraints.
        # b_t_slack_hi = Variable(n, pos=True, name="beam upper slack")
        # s_t_vars["beam"] = [b_t_slack_lo, b_t_slack_hi]
        # constrs += rx_to_slack_constrs(b_t, patient_rx["beam_constrs"], s_t_vars["beam"])
        constrs += rx_to_constrs(b_t, patient_rx["beam_constrs"])

    # Additional dose constraints in period.
    if "dose_constrs" in patient_rx:
    	# d_t_slack_lo = Variable(K, nonneg=True, name="dose lower slack")  # Slack for dose constraints.
        # d_t_slack_hi = Variable(K, nonneg=True, name="dose upper slack")
        # s_t_vars["dose"] = [d_t_slack_lo, d_t_slack_hi]
        # constrs += rx_to_slack_constrs(d_t, patient_rx["dose_constrs"], s_t_vars["dose"])
        constrs += rx_to_constrs(d_t, patient_rx["dose_constrs"])

    # Final problem.
    obj += slack_quad_penalty(s_t_vars, patient_rx["is_target"], s_weights)
    constrs += slack_quad_constrs(s_t_vars, patient_rx["is_target"], s_final)
    prob = Problem(Minimize(obj), constrs)
    return prob, b_t, d_t, s_t_vars

def build_dyn_slack_quad_prob_health(alpha, beta, gamma, patient_rx, h_init, patient_rx, T_treat, T_recov = 0):
	K = h_init.shape[0]
    if patient_rx["health_goal"].shape != (T_treat, K):
        raise ValueError("health_goal must have dimensions ({0},{1})".format(T_treat, K))

    # Define variables.
    h = Variable((T_treat + 1, K), name="health")  # Health statuses.
    d = Variable((T_treat, K), nonneg=True, name="doses")  # Doses.
    d_parm = Parameter(d.shape, nonneg=True, name="dose parameter")  # Dose point around which to linearize dynamics.

    # Health penalty function.
    obj = sum([health_penalty(h[t+1], patient_rx["health_goal"][t], patient_rx["health_weights"]) for t in range(T_treat)])

    # Health dynamics for treatment stage.
    h_lin = h[:-1] - multiply(alpha, d) + gamma[:T_treat]
    h_quad = h_lin - multiply(beta, square(d))
    h_taylor = h_lin - multiply(multiply(beta, d_parm), 2*d - d_parm)

    constrs = [h[0] == h_init]
    for t in range(T_treat):
        # For PTV, approximate dynamics via a first-order Taylor expansion.
        constrs.append(h[t+1, patient_rx["is_target"]] == h_taylor[t, patient_rx["is_target"]])

        # For OAR, relax dynamics constraint to an upper bound that is always tight at optimum.
        constrs.append(h[t+1, ~patient_rx["is_target"]] <= h_quad[t, ~patient_rx["is_target"]])

    # Additional health constraints.
    if "health_constrs" in patient_rx:
    	# TODO: Only create lower/upper slacks for organ-at-risk/target structures.
        h_slack_lo = Variable((T_treat, K), nonneg=True, name="health lower slack")  # Slack for health status constraints.
        h_slack_hi = Variable((T_treat, K), nonneg=True, name="health upper slack")
        s_vars["health"] = {"lower": h_slack_lo, "upper": h_slack_hi}
        constrs += rx_to_slack_quad_constrs(h[1:], patient_rx["health_constrs"], patient_rx["is_target"], s_vars["health"])

    # Health dynamics for recovery stage.
    # TODO: Should we return h_r or calculate it later?
    if T_recov > 0:
        gamma_r = gamma[T_treat:]

        h_r = Variable((T_recov, K), name="recovery")
        constrs_r = [h_r[0] == h[-1] + gamma_r[0]]
        for t in range(T_recov - 1):
            constrs_r.append(h_r[t + 1] == h_r[t] + gamma_r[t + 1])

        # Additional health constraints during recovery.
        if "recov_constrs" in patient_rx:
            h_r_slack_lo = Variable((T_recov, K), nonneg=True, name="health recovery lower slack")  # Slack for health status constraints in recovery phase.
            h_r_slack_hi = Variable((T_recov, K), nonneg=True, name="health recovery upper slack")
            s_vars["health_recov"] = {"lower": h_r_slack_lo, "upper": h_r_slack_hi}
            # constrs_r += rx_to_slack_constrs(h_r, patient_rx["recov_constrs"], s_vars["health_recov"])
            constrs_r += rx_to_slack_quad_constrs(h_r, patient_rx["recov_constrs"], patient_rx["is_target"], s_vars["health_recov"])
        constrs += constrs_r

    # Final problem.
    obj += slack_quad_penalty(s_vars, patient_rx["is_target"], s_weights)
    constrs += slack_quad_constrs(s_vars, patient_rx["is_target"], s_final)
    prob = Problem(Minimize(obj), constrs)
    return prob, h, d, d_parm, s_vars