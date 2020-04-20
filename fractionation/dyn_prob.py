import cvxpy
import numpy as np
from cvxpy import *

def rx_slice(patient_rx, t_start, t_end, t_step=1, squeeze=True):
    t_slice = slice(t_start, t_end, t_step)

    rx_cur = patient_rx.copy()
    for goal_key in {"dose_goal", "health_goal"}:
        if goal_key in patient_rx:
            rx_old_slice = patient_rx[goal_key][t_slice]
            if squeeze:
                rx_old_slice = np.squeeze(rx_old_slice)
            rx_cur[goal_key] = rx_old_slice

    for constr_key in {"beam_constrs", "dose_constrs", "health_constrs"}:
        if constr_key in patient_rx:
            rx_cur[constr_key] = {}
            for lu_key in {"lower", "upper"}:
                if lu_key in patient_rx[constr_key]:
                    rx_old_slice = patient_rx[constr_key][lu_key][t_slice]
                    if squeeze:
                        rx_old_slice = np.squeeze(rx_old_slice)
                    rx_cur[constr_key][lu_key] = rx_old_slice
    return rx_cur


# Dose penalty per period.
def dose_penalty(dose, goal=None, weights=None):
    if goal is None:
        goal = np.zeros(dose.shape)
    if weights is None:
        weights = np.ones(dose.shape)
    return weights * square(dose - goal)


# Health status penalty per period.
def health_penalty(health, goal, weights):
    w_under, w_over = weights
    return w_under * neg(health - goal) + w_over * pos(health - goal)


# Full objective function.
def dyn_objective(d_var, h_var, patient_rx):
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
def rx_to_constrs(expr, rx_dict):
    constrs = []

    # Lower bound.
    if "lower" in rx_dict:
        rx_lower = rx_dict["lower"]
        if np.any(rx_lower == np.inf):
            raise ValueError("Lower bound cannot be infinity")

        if np.isscalar(rx_lower):
            if np.isfinite(rx_lower):
                constrs.append(expr >= rx_lower)
        else:
            if rx_lower.shape != expr.shape:
                raise ValueError("rx_lower must have dimensions {0}".format(expr.shape))
            is_finite = np.isfinite(rx_lower)
            if np.any(is_finite):
                constrs.append(expr[is_finite] >= rx_lower[is_finite])

    # Upper bound.
    if "upper" in rx_dict:
        rx_upper = rx_dict["upper"]
        if np.any(rx_upper == -np.inf):
            raise ValueError("Upper bound cannot be negative infinity")

        if np.isscalar(rx_upper):
            if np.isfinite(rx_upper):
                constrs.append(expr <= rx_upper)
        else:
            if rx_upper.shape != expr.shape:
                raise ValueError("rx_upper must have dimensions {0}".format(expr.shape))
            is_finite = np.isfinite(rx_upper)
            if np.any(is_finite):
                constrs.append(expr[is_finite] <= rx_upper[is_finite])
    return constrs


# Construct optimal control problem.
def build_dyn_prob(A_list, F_list, G_list, r_list, h_init, patient_rx, T_recov=0):
    T_treat = len(A_list)
    K, n = A_list[0].shape
    if h_init.shape[0] != K:
        raise ValueError("h_init must be a vector of {0} elements".format(K))

    # Define variables.
    b = Variable((T_treat, n), pos=True, name="beams")  # Beams.
    h = Variable((T_treat + 1, K), name="health")  # Health statuses.
    d = vstack([A_list[t] * b[t] for t in range(T_treat)])  # Doses.

    # Objective function.
    obj = dyn_objective(d, h, patient_rx)

    # Health dynamics for treatment stage.
    # constrs = [h[0] == h_init, b >= 0]
    constrs = [h[0] == h_init]
    for t in range(T_treat):
        constrs.append(h[t + 1] == F_list[t] * h[t] + G_list[t] * d[t] + r_list[t])

    # Additional beam constraints.
    if "beam_constrs" in patient_rx:
        constrs += rx_to_constrs(b, patient_rx["beam_constrs"])

    # Additional dose constraints.
    if "dose_constrs" in patient_rx:
        constrs += rx_to_constrs(d, patient_rx["dose_constrs"])

    # Additional health constraints.
    if "health_constrs" in patient_rx:
        constrs += rx_to_constrs(h[1:], patient_rx["health_constrs"])

    # Health dynamics for recovery stage.
    # TODO: Should we return h_r or calculate it later?
    if T_recov > 0:
        F_recov = F_list[T_treat:]
        r_recov = r_list[T_treat:]

        h_r = Variable((T_recov, K), name="recovery")
        constrs_r = [h_r[0] == F_recov[0] * h[-1] + r_recov[0]]
        for t in range(T_recov - 1):
            constrs_r.append(h_r[t + 1] == F_recov[t + 1] * h_r[t] + r_recov[t + 1])

        # Additional health constraints during recovery.
        if "recov_constrs" in patient_rx:
            constrs_r += rx_to_constrs(h_r, patient_rx["recov_constrs"])
        constrs += constrs_r

    prob = Problem(Minimize(obj), constrs)
    return prob, b, h, d