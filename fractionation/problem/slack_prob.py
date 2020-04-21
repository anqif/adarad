import cvxpy
import numpy as np
import itertools

from cvxpy import *
from collections import defaultdict

from fractionation.mpc_funcs import dyn_objective, rx_to_constrs

def dyn_slack_problem(slack_vars, slack_weights=defaultdict(lambda: 1.0), slack_final=False):
    # Objective = \sum_{i=1}^N w_i*||s_i||_2^2, where N = number of slack constraint categories.
    slack_wss = []
    for key, slack in slack_vars.items():
        if isinstance(slack, list):
            slack = hstack(slack)
        slack_wss += [slack_weights[key]*sum_squares(slack)]
    obj = sum(slack_wss)

    # Disable slack for final session.
    constrs = []
    if not slack_final:
        if "dose" in slack_vars:
            constrs += [slack[-1] == 0 for slack in slack_vars["dose"]]
        if "health_recov" in slack_vars:
            constrs += [slack[-1] == 0 for slack in slack_vars["health_recov"]]
        elif "health" in slack_vars:
            constrs += [slack[-1] == 0 for slack in slack_vars["health"]]
    return Problem(Minimize(obj), constrs)

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
def build_dyn_slack_prob(A_list, F_list, G_list, r_list, h_init, patient_rx, T_recov=0, s_weights=defaultdict(lambda: 1.0), s_final=False):
    T_treat = len(A_list)
    K, n = A_list[0].shape
    if h_init.shape[0] != K:
        raise ValueError("h_init must be a vector of {0} elements".format(K))

    # Main variables.
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
    s_vars = dict()
    if "beam_constrs" in patient_rx:
        # b_slack_lo = Variable((T_treat, n), pos=True, name="beam lower slack")  # Slack for beam constraints.
        # b_slack_hi = Variable((T_treat, n), pos=True, name="beam upper slack")
        # s_vars["beams"] = [b_slack_lo, b_slack_hi]
        # constrs += rx_to_slack_constrs(b, patient_rx["beam_constrs"], s_vars["beam"])
        constrs += rx_to_constrs(b, patient_rx["beam_constrs"])

    # Additional dose constraints.
    if "dose_constrs" in patient_rx:
        d_slack_lo = Variable((T_treat, K), pos=True, name="dose lower slack")  # Slack for dose constraints.
        d_slack_hi = Variable((T_treat, K), pos=True, name="dose upper slack")
        s_vars["dose"] = [d_slack_lo, d_slack_hi]
        constrs += rx_to_slack_constrs(d, patient_rx["dose_constrs"], s_vars["dose"])

    # Additional health constraints.
    if "health_constrs" in patient_rx:
        h_slack_lo = Variable((T_treat, K), pos=True, name="health lower slack")  # Slack for health status constraints.
        h_slack_hi = Variable((T_treat, K), pos=True, name="health upper slack")
        s_vars["health"] = [h_slack_lo, h_slack_hi]
        constrs += rx_to_slack_constrs(h[1:], patient_rx["health_constrs"], s_vars["health"])

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
            h_r_slack_lo = Variable((T_recov, K), pos=True, name="health recovery lower slack")  # Slack for health status constraints in recovery phase.
            h_r_slack_hi = Variable((T_recov, K), pos=True, name="health recovery upper slack")
            s_vars["health_recov"] = [h_r_slack_lo, h_r_slack_hi]
            constrs_r += rx_to_slack_constrs(h_r, patient_rx["recov_constrs"], s_vars["health_recov"])
        constrs += constrs_r

    # Final problem.
    prob_main = Problem(Minimize(obj), constrs)
    prob_slack = dyn_slack_problem(s_vars, s_weights, s_final)
    prob = prob_main + prob_slack
    return prob, b, h, d, s_vars
