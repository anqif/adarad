import cvxpy
import numpy as np
import itertools
from cvxpy import *

from fractionation.mpc_funcs import dyn_objective, rx_to_constrs

# Full objective function with slack penalty.
def dyn_slack_objective(d_var, h_var, patient_rx, s_vars=[], s_weights=None):
    n_slack = len(s_vars)
    if s_weights is None:
        s_weights = np.ones(n_slack)
    if len(s_weights) != n_slack:
        raise ValueError("s_weights must be a vector of length {0}".format(n_slack))

    main_loss = dyn_objective(d_var, h_var, patient_rx)
    s_sum_sqs = hstack([sum_squares(slack) for slack in s_vars])
    slack_loss = s_weights * s_sum_sqs
    return main_loss + slack_loss

# Extract constraints from patient prescription and add slack.
def rx_to_slack_constrs(expr, rx_dict, slack):
    slack_lo, slack_hi = slack
    if slack_lo.shape != expr.shape:
        raise ValueError("slack_lo must have dimensions ({0},{1})".format(*expr.shape))
    if slack_hi.shape != expr.shape:
        raise ValueError("slack_hi must have dimensions ({0},{1})".format(*expr.shape))

    constrs = []
    # Lower bound.
    if "lower" in rx_dict:
        rx_lower = rx_dict["lower"]
        if np.any(rx_lower == np.inf):
            raise ValueError("Lower bound cannot be infinity")

        if np.isscalar(rx_lower):
            if np.isfinite(rx_lower):
                constrs.append(expr + slack_lo >= rx_lower)
        else:
            if rx_lower.shape != expr.shape:
                raise ValueError("rx_lower must have dimensions {0}".format(expr.shape))
            is_finite = np.isfinite(rx_lower)
            if np.any(is_finite):
                constrs.append(expr[is_finite] + slack_lo[is_finite] >= rx_lower[is_finite])

    # Upper bound.
    if "upper" in rx_dict:
        rx_upper = rx_dict["upper"]
        if np.any(rx_upper == -np.inf):
            raise ValueError("Upper bound cannot be negative infinity")

        if np.isscalar(rx_upper):
            if np.isfinite(rx_upper):
                constrs.append(expr - slack_hi <= rx_upper)
        else:
            if rx_upper.shape != expr.shape:
                raise ValueError("rx_upper must have dimensions {0}".format(expr.shape))
            is_finite = np.isfinite(rx_upper)
            if np.any(is_finite):
                constrs.append(expr[is_finite] - slack_hi[is_finite] <= rx_upper[is_finite])
    return constrs

# Construct optimal control problem with slack health/dose constraints.
def build_dyn_slack_prob(A_list, F_list, G_list, r_list, h_init, patient_rx, T_recov=0, slack_final=False):
    T_treat = len(A_list)
    K, n = A_list[0].shape
    if h_init.shape[0] != K:
        raise ValueError("h_init must be a vector of {0} elements".format(K))

    # Main variables.
    b = Variable((T_treat, n), pos=True, name="beams")  # Beams.
    h = Variable((T_treat + 1, K), name="health")  # Health statuses.
    d = vstack([A_list[t] * b[t] for t in range(T_treat)])  # Doses.

    # Slack variables.
    # b_slack_lo = Variable((T_treat, n), pos=True, name="beam lower slack")  # Slack for beam constraints.
    # b_slack_hi = Variable((T_treat, n), pos=True, name="beam upper slack")

    h_slack_lo = Variable((T_treat, K), pos=True, name="health lower slack")  # Slack for health status constraints.
    h_slack_hi = Variable((T_treat, K), pos=True, name="health upper slack")

    d_slack_lo = Variable((T_treat, K), pos=True, name="dose lower slack")  # Slack for dose constraints.
    d_slack_hi = Variable((T_treat, K), pos=True, name="dose upper slack")
    s_vars = {"health": [h_slack_lo, h_slack_hi], "dose": [d_slack_lo, d_slack_hi]}
    # s_vars = {"beam": [b_slack_lo, b_slack_hi], "health": [h_slack_lo, h_slack_hi], "dose": [d_slack_lo, d_slack_hi]}

    # Health dynamics for treatment stage.
    # constrs = [h[0] == h_init, b >= 0]
    constrs = [h[0] == h_init]
    for t in range(T_treat):
        constrs.append(h[t + 1] == F_list[t] * h[t] + G_list[t] * d[t] + r_list[t])

    # Additional beam constraints.
    if "beam_constrs" in patient_rx:
        constrs += rx_to_constrs(b, patient_rx["beam_constrs"])
        # constrs += rx_to_slack_constrs(b, patient_rx["beam_constrs"], s_vars["beam"])

    # Additional dose constraints.
    if "dose_constrs" in patient_rx:
        constrs += rx_to_slack_constrs(d, patient_rx["dose_constrs"], s_vars["dose"])

    # Additional health constraints.
    if "health_constrs" in patient_rx:
        constrs += rx_to_slack_constrs(h[1:], patient_rx["health_constrs"], s_vars["health"])

    # Health dynamics for recovery stage.
    # TODO: Should we return h_r or calculate it later?
    if T_recov > 0:
        F_recov = F_list[T_treat:]
        r_recov = r_list[T_treat:]

        h_r = Variable((T_recov, K), name="recovery")
        h_r_slack_lo = Variable((T_recov, K), pos=True, name="health recovery lower slack")
        h_r_slack_hi = Variable((T_recov, K), pos=True, name="health recovery upper slack")
        s_vars["health_recov"] = [h_r_slack_lo, h_r_slack_hi]

        constrs_r = [h_r[0] == F_recov[0] * h[-1] + r_recov[0]]
        for t in range(T_recov - 1):
            constrs_r.append(h_r[t + 1] == F_recov[t + 1] * h_r[t] + r_recov[t + 1])

        # Additional health constraints during recovery.
        if "recov_constrs" in patient_rx:
            constrs_r += rx_to_slack_constrs(h_r, patient_rx["recov_constrs"], s_vars["health_recov"])
        constrs += constrs_r

    # Disable slack for final session.
    if not slack_final:
        constrs += [slack[-1] == 0 for slack in s_vars["dose"]]
        s_health_fin = s_vars["health_recov"] if T_recov > 0 else s_vars["health"]
        constrs += [slack[-1] == 0 for slack in s_health_fin]

    # Objective function.
    s_vars_list = list(itertools.chain(*s_vars.values()))
    obj = dyn_slack_objective(d, h, patient_rx, s_vars_list)
    prob = Problem(Minimize(obj), constrs)
    return prob, b, h, d, s_vars
