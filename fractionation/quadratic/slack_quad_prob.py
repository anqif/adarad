import cvxpy
import numpy as np
import itertools

from cvxpy import *
from collections import defaultdict

from fractionation.problem.slack_prob import rx_to_constrs
from fractionation.quadratic.dyn_quad_prob import dyn_quad_obj

# Total slack penalty across periods.
def slack_quad_penalty(slack_vars, is_target, slack_weights = None):
    if slack_weights is None:
        slack_weights = defaultdict(lambda: 1.0)

    # Objective = \sum_{i=1}^N w_i*||s_i||_2^2, where N = number of slack constraint categories.
    slack_wss = []
    for key, slack in slack_vars.items():
        base_penalty = 0
        if "lower" in slack:
            base_penalty += sum_squares(slack["lower"][:,~is_target])
        if "upper" in slack:
            base_penalty += sum_squares(slack["upper"][:,is_target])
        slack_wss += [slack_weights[key]*base_penalty]
    return sum(slack_wss)

# Constraints on slack variables.
def slack_quad_constrs(slack_vars, is_target, slack_final = True):
    constrs = []
    for slack_dict in slack_vars.values():
        for slack in slack_dict.values():
            if not slack.is_nonneg():
                constrs += [slack >= 0]

    # No slack for constraints in final period.
    if not slack_final:
        if "dose" in slack_vars:
            constrs += [slack[-1] == 0 for slack in slack_vars["dose"].values()]
        if "health_recov" in slack_vars:
            constrs += [slack[-1] == 0 for slack in slack_vars["health_recov"].values()]
        elif "health" in slack_vars:
            constrs += [slack[-1] == 0 for slack in slack_vars["health"].values()]
    return constrs

# Convert constraints to slack penalty.
def rx_to_slack_quad_penalty(expr, rx_dict, is_target, weights = None):
    T, K = expr.shape
    if weights is None:
        weights = np.ones((T,K))
    elif np.isscalar(weights):
        weights = np.full((T,K), weights)
    elif weights.shape != (T,K):
        raise ValueError("weights must have dimensions ({0},{1})".format(T,K))
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")

    penalty = 0
    # Lower bound.
    if "lower" in rx_dict:
        rx_lower = rx_dict["lower"]
        expr_oar = expr[:,~is_target]
        w_lower = weights[:,~is_target]

        if np.any(rx_lower == np.inf):
            raise ValueError("Lower bound cannot be infinity")
        if np.isscalar(rx_lower):
            if np.isfinite(rx_lower):
                penalty += sum(multiply(w_lower, pos(rx_lower - expr_oar)))
        else:
            if rx_lower.shape != expr_oar.shape:
                raise ValueError("rx_lower must have dimensions {0}".format(expr_oar.shape))
            is_finite = np.isfinite(rx_lower)
            if np.any(is_finite):
                penalty += sum(multiply(w_lower[is_finite], pos(rx_lower[is_finite] - expr_oar[is_finite])))

    # Upper bound.
    if "upper" in rx_dict:
        rx_upper = rx_dict["upper"]
        expr_ptv = expr[:,is_target]
        w_upper = weights[is_target]

        if np.any(rx_upper == -np.inf):
            raise ValueError("Upper bound cannot be negative infinity")

        if np.isscalar(rx_upper):
            if np.isfinite(rx_upper):
                penalty += sum(multiply(w_upper, pos(expr_ptv - rx_upper)))
        else:
            if rx_upper.shape != expr_ptv.shape:
                raise ValueError("rx_upper must have dimensions {0}".format(expr_ptv.shape))
            is_finite = np.isfinite(rx_upper)
            if np.any(is_finite):
                penalty += sum(multiply(w_upper[is_finite], pos(expr_ptv[is_finite] - rx_upper[is_finite])))
    return penalty

# Extract constraints from patient prescription.
def rx_to_slack_quad_constrs(expr, rx_dict, is_target, slack):
    slack_lo, slack_hi = slack
    if slack_lo.shape != expr.shape:
        raise ValueError("slack_lo must have dimensions ({0},{1})".format(*expr.shape))
    if slack_hi.shape != expr.shape:
        raise ValueError("slack_hi must have dimensions ({0},{1})".format(*expr.shape))

    constrs = []
    # Lower bound.
    if "lower" in rx_dict:
        rx_lower = rx_dict["lower"]
        expr_oar = expr[:,~is_target]
        slack_lo_oar = slack_lo[:,~is_target]

        if np.any(rx_lower == np.inf):
            raise ValueError("Lower bound cannot be infinity")

        if np.isscalar(rx_lower):
            if np.isfinite(rx_lower):
                constrs.append(expr_oar + slack_lo_oar >= rx_lower)
        else:
            if rx_lower.shape != expr_oar.shape:
                raise ValueError("rx_lower must have dimensions {0}".format(expr_oar.shape))
            is_finite = np.isfinite(rx_lower)
            if np.any(is_finite):
                constrs.append(expr_oar[is_finite] + slack_lo_oar[is_finite] >= rx_lower[is_finite])

    # Upper bound.
    if "upper" in rx_dict:
        rx_upper = rx_dict["upper"]
        expr_ptv = expr[:,is_target]
        slack_hi_ptv = slack_hi[:,is_target]

        if np.any(rx_upper == -np.inf):
            raise ValueError("Upper bound cannot be negative infinity")

        if np.isscalar(rx_upper):
            if np.isfinite(rx_upper):
                constrs.append(expr_ptv - slack_hi_ptv <= rx_upper)
        else:
            if rx_upper.shape != expr_ptv.shape:
                raise ValueError("rx_upper must have dimensions {0}".format(expr_ptv.shape))
            is_finite = np.isfinite(rx_upper)
            if np.any(is_finite):
                constrs.append(expr_ptv[is_finite] - slack_hi_ptv[is_finite] <= rx_upper[is_finite])
    return constrs

# Construct optimal control problem with slack health/dose constraints.
def build_dyn_slack_quad_prob(A_list, alpha, beta, gamma, h_init, patient_rx, T_recov = 0, use_h_dyn_slack = False,
                              h_dyn_slack_weight = 0, h_bnd_slack_weights = 1):
    T_treat = len(A_list)
    K, n = A_list[0].shape
    if h_init.shape[0] != K:
        raise ValueError("h_init must be a vector of {0} elements".format(K))

    # Main variables.
    b = Variable((T_treat, n), nonneg=True, name="beams")  # Beams.
    h = Variable((T_treat + 1, K), name="health")  # Health statuses.
    d = vstack([A_list[t] * b[t] for t in range(T_treat)])  # Doses.
    d_parm = Parameter(d.shape, nonneg=True, name="dose parameter")  # Dose point around which to linearize dynamics.

    # Objective function.
    obj = dyn_quad_obj(d, h, patient_rx)

    # Health dynamics for treatment stage.
    h_lin = h[:-1] - multiply(alpha, d) + gamma[:T_treat]
    h_quad = h_lin - multiply(beta, square(d))
    h_taylor = h_lin - multiply(multiply(beta, d_parm), 2*d - d_parm)

    # Allow slack in health dynamics constraints.
    h_dyn_slack = Constant(0)
    if use_h_dyn_slack:
        h_dyn_slack = Variable((T_treat, K), nonneg=True, name="health dynamics slack")
        obj += h_dyn_slack_weight * sum(h_dyn_slack)  # TODO: Set slack weight relative to overall health penalty.
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

    # Additional health constraints go into objective as a penalty.
    if "health_constrs" in patient_rx:
        obj += rx_to_slack_quad_penalty(h[1:], patient_rx["health_constrs"], patient_rx["is_target"], h_bnd_slack_weights)

    # Health dynamics for recovery stage.
    # TODO: Should we return h_r or calculate it later?
    if T_recov > 0:
        gamma_r = gamma[T_treat:]

        h_r = Variable((T_recov, K), name="recovery")
        constrs_r = [h_r[0] == h[-1] + gamma_r[0]]
        for t in range(T_recov - 1):
            constrs_r.append(h_r[t + 1] == h_r[t] + gamma_r[t + 1])

        # Additional health constraints during recovery go into objective as a penalty.
        if "recov_constrs" in patient_rx:
            obj += rx_to_slack_quad_penalty(h_r, patient_rx["recov_constrs"], patient_rx["is_target"], h_bnd_slack_weights)
        constrs += constrs_r

    # Final problem.
    prob = Problem(Minimize(obj), constrs)
    return prob, b, h, d, d_parm, h_dyn_slack
