import cvxpy
import numpy as np
from cvxpy import *

from fractionation.problem.dyn_prob import dose_penalty, health_penalty, rx_to_constrs

# Full objective function.
def dyn_quad_obj(d_var, h_var, patient_rx):
    T, K = d_var.shape
    if h_var.shape[0] != T + 1:
        raise ValueError("h_var must have exactly {0} rows".format(T + 1))
    if patient_rx["dose_goal"].shape != (T, K):
        raise ValueError("dose_goal must have dimensions ({0},{1})".format(T, K))
    if patient_rx["health_goal"].shape != (T, K):
        raise ValueError("health_goal must have dimensions ({0},{1})".format(T, K))

    penalties = []
    for t in range(T):
        d_penalty = dose_penalty(d_var[t], patient_rx["dose_goal"][t], patient_rx["dose_weights"])
        h_penalty = health_penalty(h_var[t + 1], patient_rx["health_goal"][t], patient_rx["health_weights"])
        penalties.append(d_penalty + h_penalty)
    return sum(penalties)

# Extract constraints from patient prescription.
def rx_to_quad_constrs(expr, rx_dict, is_target):
    constrs = []

    # Lower bound.
    if "lower" in rx_dict:
        rx_lower = rx_dict["lower"]
        expr_oar = expr[:,~is_target]
        if np.any(rx_lower == np.inf):
            raise ValueError("Lower bound cannot be infinity")

        if np.isscalar(rx_lower):
            if np.isfinite(rx_lower):
                constrs.append(expr_oar >= rx_lower)
        else:
            if rx_lower.shape != expr_oar.shape:
                raise ValueError("rx_lower must have dimensions {0}".format(expr_oar.shape))
            is_finite = np.isfinite(rx_lower)
            if np.any(is_finite):
                constrs.append(expr_oar[is_finite] >= rx_lower[is_finite])

    # Upper bound.
    if "upper" in rx_dict:
        rx_upper = rx_dict["upper"]
        expr_ptv = expr[:,is_target]
        if np.any(rx_upper == -np.inf):
            raise ValueError("Upper bound cannot be negative infinity")

        if np.isscalar(rx_upper):
            if np.isfinite(rx_upper):
                constrs.append(expr_ptv <= rx_upper)
        else:
            if rx_upper.shape != expr_ptv.shape:
                raise ValueError("rx_upper must have dimensions {0}".format(expr_ptv.shape))
            is_finite = np.isfinite(rx_upper)
            if np.any(is_finite):
                constrs.append(expr_ptv[is_finite] <= rx_upper[is_finite])
    return constrs

# Construct optimal control problem.
def build_dyn_quad_prob(A_list, alpha, beta, gamma, h_init, patient_rx, T_recov=0, use_slack=False):
    T_treat = len(A_list)
    K, n = A_list[0].shape
    if h_init.shape[0] != K:
        raise ValueError("h_init must be a vector of {0} elements".format(K))

    # Define variables.
    b = Variable((T_treat, n), nonneg=True, name="beams")   # Beams.
    h = Variable((T_treat + 1, K), name="health")           # Health statuses.
    d = vstack([A_list[t] * b[t] for t in range(T_treat)])  # Doses.
    d_parm = Parameter(d.shape, nonneg=True, name="dose parameter")  # Dose point around which to linearize target dynamics.

    # Objective function.
    obj = dyn_quad_obj(d, h, patient_rx)

    # Health dynamics for treatment stage.
    h_lin = h[:-1] - multiply(alpha, d) + gamma[:T_treat]
    h_quad = h_lin - multiply(beta, square(d))
    h_taylor = h_lin - multiply(multiply(beta, d_parm), 2*d - d_parm)

    # Allow slack in health dynamics constraints.
    h_dyn_slack = Constant(0)
    if use_slack:
        h_dyn_slack = Variable((T_treat, K), nonneg=True, name="health dynamics slack")
        h_dyn_slack_weight = Parameter(nonneg=True, name="health dynamics slack weight")
        h_dyn_slack_weight.value = 1e4   # TODO: Set slack weight relative to overall health penalty.
        obj += h_dyn_slack_weight*sum(h_dyn_slack)
        h_taylor -= h_dyn_slack

    constrs = [h[0] == h_init]
    for t in range(T_treat):
        # For PTV, approximate dynamics via a first-order Taylor expansion.
        constrs.append(h[t+1, patient_rx["is_target"]] == h_taylor[t, patient_rx["is_target"]])

        # For OAR, relax dynamics constraint to an upper bound that is always tight at optimum.
        constrs.append(h[t+1, ~patient_rx["is_target"]] <= h_quad[t, ~patient_rx["is_target"]])

    # Additional beam constraints.
    if "beam_constrs" in patient_rx:
        constrs += rx_to_constrs(b, patient_rx["beam_constrs"])

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
    return prob, b, h, d, d_parm, h_dyn_slack
