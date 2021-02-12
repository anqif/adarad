import cvxpy
import numpy as np
import cvxpy.settings as cvxpy_s
from cvxpy import *

from fractionation.problem.dyn_prob import dose_penalty, health_penalty
from fractionation.quadratic.dyn_quad_prob import rx_to_quad_constrs
from fractionation.utilities.data_utils import check_quad_vectors

def dyn_init_dose(A_list, alpha, beta, gamma, h_init, patient_rx, T_recov=0, use_slack = False, slack_weight = 0, *args, **kwargs):
    T_treat = len(A_list)
    K, n = A_list[0].shape
    alpha, beta, gamma = check_quad_vectors(alpha, beta, gamma, K, T_treat, T_recov)

    # Build problem for dose initialization stage.
    prob, b, h, d, h_lin_slack = build_dyn_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx, T_recov, use_slack, slack_weight)
    prob.solve(*args, **kwargs)
    if prob.status not in cvxpy_s.SOLUTION_PRESENT:
        raise RuntimeError("Solver failed with status {0}".format(prob.status))

    # Constant dose per fraction
    d_init = np.array(T_treat*[d.value])
    return {"dose": d_init, "solve_time": prob.solver_stats.solve_time}

# Objective function for initialization problem.
def dyn_obj_init(d_var, h_var, patient_rx):
    T_plus1, K = h_var.shape
    T = T_plus1 - 1
    if d_var.shape[0] not in [(K,), (K,1)]:
        raise ValueError("d_var must have dimensions ({0},)".format(K))
    if patient_rx["dose_goal"].shape != (T, K):
        raise ValueError("dose_goal must have dimensions ({0},{1})".format(T, K))
    if patient_rx["health_goal"].shape != (T, K):
        raise ValueError("health_goal must have dimensions ({0},{1})".format(T, K))

    dose_init_goal = np.mean(patient_rx["dose_goal"], axis = 0)   # Average dose goal over time.
    h_penalties = [health_penalty(h_var[t + 1], patient_rx["health_goal"][t], patient_rx["health_weights"]) for t in range(T)]
    d_penalty = T * dose_penalty(d_var, dose_init_goal)   # Same dose in each fraction.
    return sum(h_penalties) + d_penalty

# Change time-varying bounds to static bound per fraction.
def get_init_bnd(rx_bound, is_lower, type = "first"):
    if type == "first":
        return rx_bound[0]
    elif type == "tight":   # [max(lower_bound), min(upper_bound)]
        return np.max(rx_bound, axis = 0) if is_lower else np.min(rx_bound, axis = 0)
    elif type == "loose":   # [min(lower_bound), max(upper_bound)]
        return np.min(rx_bound, axis = 0) if is_lower else np.max(rx_bound, axis = 0)
    elif type == "mean":
        return np.mean(rx_bound, axis = 0)
    else:
        raise ValueError("type must be 'first', 'tight', 'loose', or 'mean'")

def get_init_lower_bnd(rx_bound, type = "first"):
    return get_init_bnd(rx_bound, is_lower = True, type = type)

def get_init_upper_bnd(rx_bound, type = "first"):
    return get_init_bnd(rx_bound, is_lower = False, type = type)

# Extract initialization constraints from patient prescription.
def rx_to_init_constrs(expr, rx_dict, type = "first", is_lower = True):
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
            init_lower = get_init_lower_bnd(rx_lower, type)
            if init_lower.shape != expr.shape:
                raise RuntimeError("init_lower must have dimensions {0}".format(expr.shape))
            is_finite = np.isfinite(init_lower)
            if np.any(is_finite):
                constrs.append(expr[is_finite] >= init_lower[is_finite])

    # Upper bound.
    if "upper" in rx_dict:
        rx_upper = rx_dict["upper"]
        if np.any(rx_upper == -np.inf):
            raise ValueError("Upper bound cannot be negative infinity")

        if np.isscalar(rx_upper):
            if np.isfinite(rx_upper):
                constrs.append(expr <= rx_upper)
        else:
            init_upper = get_init_upper_bnd(rx_upper, type)
            if init_upper.shape != expr.shape:
                raise RuntimeError("init_upper must have dimensions {0}".format(expr.shape))
            is_finite = np.isfinite(init_upper)
            if np.any(is_finite):
                constrs.append(expr[is_finite] <= init_upper[is_finite])
    return constrs

# Construct optimal control problem.
def build_dyn_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx, T_recov = 0, use_slack = False, slack_weight = 0):
    T_treat = len(A_list)
    K, n = A_list[0].shape
    if h_init.shape[0] != K:
        raise ValueError("h_init must be a vector of {0} elements".format(K))

    b = Variable((n,), nonneg=True, name="beams")  # Beams per fraction.
    h = Variable((T_treat + 1, K), name="health")  # Health statuses.
    d = A_list[0] @ b   # Constant dose per fraction.
    d_rep = vstack(T_treat*[d])

    # Objective function.
    obj = dyn_obj_init(d, h, patient_rx)

    # Health dynamics for treatment stage.
    h_lin = h[:-1] - multiply(alpha, d_rep) + gamma[:T_treat]
    h_quad = h_lin - multiply(beta, square(d_rep))

    # Allow slack in PTV health dynamics constraint.
    h_slack = Constant(0)
    if use_slack:
        h_slack = Variable((T_treat, K), nonneg=True, name="health dynamics slack")
        obj += slack_weight * sum(h_slack)  # TODO: Set slack weight relative to overall health penalty.
    h_lin_slack = h_lin - h_slack

    constrs = [h[0] == h_init]
    for t in range(T_treat):
        # For PTV, linearize dynamics constraint by dropping quadratic dose term.
        constrs.append(h[t + 1, patient_rx["is_target"]] == h_lin_slack[t, patient_rx["is_target"]])

        # For OAR, relax dynamics constraint to an upper bound that is always tight at optimum.
        constrs.append(h[t + 1, ~patient_rx["is_target"]] <= h_quad[t, ~patient_rx["is_target"]])

    # Additional beam constraints.
    if "beam_constrs" in patient_rx:
        constrs += rx_to_init_constrs(b, patient_rx["beam_constrs"])

    # Additional dose constraints.
    if "dose_constrs" in patient_rx:
        constrs += rx_to_init_constrs(d, patient_rx["dose_constrs"])

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

    prob = Problem(Minimize(obj), constrs)
    return prob, b, h, d, h_lin_slack
