import numpy as np
import cvxpy
from cvxpy import *

from fractionation.problem.dyn_prob import dose_penalty, health_penalty, rx_to_constrs
from fractionation.quadratic.dyn_quad_prob import rx_to_quad_constrs

def build_dyn_quad_prob_dose(A_list, patient_rx):
    T_treat = len(A_list)
    K, n = A_list[0].shape
    if patient_rx["dose_goal"].shape != (T_treat, K):
        raise ValueError("dose_goal must have dimensions ({0},{1})".format(T_treat, K))

    # Define variables.
    b = Variable((T_treat, n), nonneg=True, name="beams")   # Beams.
    d = vstack([A_list[t] * b[t] for t in range(T_treat)])  # Doses.

    # Dose penalty function.
    obj = sum([dose_penalty(d[t], patient_rx["dose_goal"][t], patient_rx["dose_weights"]) for t in range(T_treat)])

    # Additional beam constraints.
    constrs = []
    if "beam_constrs" in patient_rx:
        constrs += rx_to_constrs(b, patient_rx["beam_constrs"])

    # Additional dose constraints.
    if "dose_constrs" in patient_rx:
        constrs += rx_to_constrs(d, patient_rx["dose_constrs"])

    prob = Problem(Minimize(obj), constrs)
    return prob, b, d

def build_dyn_quad_prob_dose_period(A, patient_rx):
    K, n = A.shape

    # Define variables for period.
    b_t = Variable(n, nonneg=True, name="beams")  # Beams.
    d_t = A * b_t

    # Dose penalty current period.
    obj = dose_penalty(d_t, patient_rx["dose_goal"], patient_rx["dose_weights"])

    # Additional beam constraints in period.
    constrs = []
    if "beam_constrs" in patient_rx:
        constrs += rx_to_constrs(b_t, patient_rx["beam_constrs"])

    # Additional dose constraints in period.
    if "dose_constrs" in patient_rx:
        constrs += rx_to_constrs(d_t, patient_rx["dose_constrs"])

    prob_t = Problem(Minimize(obj), constrs)
    return prob_t, b_t, d_t

def build_dyn_quad_prob_health(alpha, beta, gamma, h_init, patient_rx, T_treat, T_recov = 0, use_slack = False, slack_weight = 0):
    K = h_init.shape[0]
    if patient_rx["health_goal"].shape != (T_treat, K):
        raise ValueError("health_goal must have dimensions ({0},{1})".format(T_treat, K))

    # Define variables.
    h = Variable((T_treat + 1, K), name="health")  # Health statuses.
    d = Variable((T_treat, K), nonneg=True, name="doses")  # Doses.
    d_parm = Parameter(d.shape, nonneg=True, name="dose parameter")  # Dose point around which to linearize dynamics.

    # Health penalty function.
    obj = sum([health_penalty(h[t + 1], patient_rx["health_goal"][t], patient_rx["health_weights"]) for t in range(T_treat)])

    # Health dynamics for treatment stage.
    h_lin = h[:-1] - multiply(alpha, d) + gamma[:T_treat]
    h_quad = h_lin - multiply(beta, square(d))
    h_taylor = h_lin - multiply(multiply(beta, d_parm), 2*d - d_parm)

    # Allow slack in health dynamics constraints.
    h_dyn_slack = Constant(0)
    if use_slack:
        h_dyn_slack = Variable((T_treat, K), nonneg=True, name="health dynamics slack")
        obj += slack_weight * sum(h_dyn_slack)  # TODO: Set slack weight relative to overall health penalty.
        h_taylor -= h_dyn_slack

    constrs = [h[0] == h_init]
    for t in range(T_treat):
        # For PTV, approximate dynamics via a first-order Taylor expansion.
        constrs.append(h[t+1, patient_rx["is_target"]] == h_taylor[t, patient_rx["is_target"]])

        # For OAR, relax dynamics constraint to an upper bound that is always tight at optimum.
        constrs.append(h[t+1, ~patient_rx["is_target"]] <= h_quad[t, ~patient_rx["is_target"]])

    # Additional dose constraints.
    if "dose_constrs" in patient_rx:
        constrs += rx_to_constrs(d, patient_rx["dose_constrs"])

    # Additional health constraints.
    if "health_constrs" in patient_rx:
        constrs += rx_to_quad_constrs(h[1:], patient_rx["health_constrs"], patient_rx["is_target"])

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
            # constrs_r += rx_to_constrs(h_r, patient_rx["recov_constrs"])
            constrs_r += rx_to_quad_constrs(h[1:], patient_rx["health_constrs"], patient_rx["is_target"])
        constrs += constrs_r

    prob = Problem(Minimize(obj), constrs)
    return prob, h, d, d_parm, h_dyn_slack
