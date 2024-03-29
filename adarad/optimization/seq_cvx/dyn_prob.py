import numpy as np
from cvxpy import *

from adarad.optimization.penalty import dose_penalty, health_penalty
from adarad.optimization.constraint import rx_to_quad_constrs, rx_to_constrs

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

def form_dyn_constrs(b, h, d, alpha, beta, gamma, h_init, patient_rx, T_recov = 0, use_taylor = True, use_slack = False, slack_weight = 0):
    T_treat, K = d.shape

    # Health dynamics for optimization stage.
    h_lin = h[:-1] - multiply(alpha, d) + gamma[:T_treat]
    h_quad = h_lin - multiply(beta, square(d))

    d_parm = Constant(np.zeros(d.shape))
    h_approx = h_lin
    if use_taylor:
        d_parm = Parameter(d.shape, nonneg=True, name="dose parameter")  # Dose point around which to linearize dynamics.
        h_approx -= multiply(multiply(beta, d_parm), 2*d - d_parm)       # First-order Taylor expansion of quadratic.

    # Allow slack in PTV health dynamics constraints.
    if use_slack:
        h_dyn_slack = Variable((T_treat, K), nonneg=True, name="health dynamics slack")
        obj_slack = slack_weight*sum(h_dyn_slack)   # TODO: Set slack weight relative to overall health penalty.
        h_approx -= h_dyn_slack
    else:
        h_dyn_slack = Constant(0)
        obj_slack = 0

    constrs = [h[0] == h_init]
    for t in range(T_treat):
        # For PTV, approximate dynamics via a first-order Taylor expansion.
        constrs.append(h[t+1, patient_rx["is_target"]] == h_approx[t, patient_rx["is_target"]])

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
            constrs_r += rx_to_quad_constrs(h_r, patient_rx["recov_constrs"], patient_rx["is_target"])
        constrs += constrs_r
    return constrs, d_parm, obj_slack, h_dyn_slack

# PTV health dynamics is just linear portion: h_{t+1} = h_t - alpha_t*d_t + gamma_t
def form_lin_constrs(b, h, d, alpha, beta, gamma, h_init, patient_rx, T_recov = 0, use_slack = False, slack_weight = 0):
    return form_dyn_constrs(b, h, d, alpha, beta, gamma, h_init, patient_rx, T_recov = T_recov, use_taylor = False, 
                            use_slack = use_slack, slack_weight = slack_weight)

# PTV health dynamics is first-order Taylor expansion: h_{t+1} = h_t - alpha_t*d_t - beta*d_t^{parm}*(2*d_t - d_t^{parm}) + gamma_t
def form_taylor_constrs(b, h, d, alpha, beta, gamma, h_init, patient_rx, T_recov = 0, use_slack = False, slack_weight = 0):
    return form_dyn_constrs(b, h, d, alpha, beta, gamma, h_init, patient_rx, T_recov = T_recov, use_taylor = True, 
                            use_slack = use_slack, slack_weight = slack_weight)

# Construct optimal control problem.
def build_dyn_quad_prob(A_list, alpha, beta, gamma, h_init, patient_rx, T_recov = 0, use_slack = False, slack_weight = 0):
    T_treat = len(A_list)
    K, n = A_list[0].shape
    if h_init.shape[0] != K:
        raise ValueError("h_init must be a vector of {0} elements".format(K))

    # Define variables.
    b = Variable((T_treat, n), nonneg=True, name="beams")   # Beams.
    h = Variable((T_treat + 1, K), name="health")           # Health statuses.
    d = vstack([A_list[t] @ b[t] for t in range(T_treat)])  # Doses.
    
    # Objective function.
    obj_base = dyn_quad_obj(d, h, patient_rx)

    # Form constraints with slack.
    constrs, d_parm, obj_slack, h_dyn_slack = form_taylor_constrs(b, h, d, alpha, beta, gamma, h_init, patient_rx, 
                                                        T_recov = T_recov, use_slack = use_slack, slack_weight = slack_weight)

    obj = obj_base + obj_slack
    prob = Problem(Minimize(obj), constrs)
    return prob, b, h, d, d_parm, h_dyn_slack
