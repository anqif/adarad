import numpy as np
from cvxpy import *

from adarad.optimization.seq_cvx.dyn_prob import dyn_quad_obj
from adarad.optimization.constraint import rx_to_constrs, rx_to_quad_constrs

# Scaled beam problem with b_t = u_t*b^{static}, where u_t >= 0 are scaling factors and b^{static} is a beam constant.
def build_scale_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx, b_static, use_dyn_slack=False,
                          dyn_slack_weight=0, use_bnd_slack=True, bnd_slack_weight=1):
    T_treat = len(A_list)
    K, n = A_list[0].shape
    if h_init.shape[0] != K:
        raise ValueError("h_init must be a vector of {0} elements".format(K))
    if b_static.shape[0] != n:
        raise ValueError("b_static must be a vector of {0} elements".format(n))
    if not np.all(b_static >= 0):
        raise ValueError("b_static must be a nonnegative vector")

    # Define variables.
    u = Variable((T_treat,), nonneg=True, name="beam weights")  # Beam scaling factors.
    h = Variable((T_treat + 1, K), name="health")  # Health statuses.

    b_list = []
    d_list = []
    d_static_list = []
    for t in range(T_treat):
        d_static_t = A_list[t] @ b_static
        b_list.append(u[t] * b_static)  # b_t = u_t * b_static
        d_list.append(u[t] * d_static_t)  # d_t = A_tb_t = u_t * (A_tb_static)
        d_static_list.append(d_static_t)
    b = vstack(b_list)  # Beams.
    d = vstack(d_list)  # Doses.
    d_static = np.vstack(d_static_list)

    # Objective function.
    obj_base = dyn_quad_obj(d, h, patient_rx)

    # Form constraints with slack.
    constrs, d_parm, obj_slack, h_dyn_slack, h_bnd_slack = \
        form_scale_constrs(b, h, u, d_static, alpha, beta, gamma, h_init, patient_rx, T_recov=0, use_taylor=True,
                           use_dyn_slack=use_dyn_slack, dyn_slack_weight=dyn_slack_weight, use_bnd_slack=use_bnd_slack,
                           bnd_slack_weight=bnd_slack_weight)

    obj = obj_base + obj_slack
    prob = Problem(Minimize(obj), constrs)
    return prob, u, b, h, d, d_parm, h_dyn_slack, h_bnd_slack

def form_scale_constrs(b, h, u, d_static, alpha, beta, gamma, h_init, patient_rx, T_recov=0, use_taylor=True,
                       use_dyn_slack=False, dyn_slack_weight=0, use_bnd_slack=True, bnd_slack_weight=1):
    T_treat, K = d_static.shape
    d = vstack([u[t] * d_static[t, :] for t in range(T_treat)])

    # Define variables.
    if use_taylor:
        d_parm = Parameter((T_treat, K), nonneg=True,
                           name="dose parameter")  # Dose point around which to linearize dynamics.
    else:
        d_parm = Constant(np.zeros(T_treat, K))

    # Allow slack in PTV health dynamics constraints.
    if use_dyn_slack:
        h_dyn_slack = Variable((T_treat, K), nonneg=True, name="health dynamics slack")
        obj_dyn_slack = dyn_slack_weight * sum(
            h_dyn_slack)  # TODO: Set slack weight relative to overall health penalty.
    else:
        h_dyn_slack = Constant(np.zeros((T_treat, K)))
        obj_dyn_slack = 0

    # Pre-compute constant expressions.
    is_target = patient_rx["is_target"]
    alpha_mul_d_stat = multiply(alpha, d_static).value
    beta_mul_d_stat_sq = multiply(beta, square(d_static)).value

    # Health dynamics for optimization stage.
    constrs = [h[0] == h_init]
    for t in range(T_treat):
        # For PTV, approximate dynamics via a first-order Taylor expansion.
        h_ptv_t = h[t, is_target] - u[t] * alpha_mul_d_stat[t, is_target] + gamma[t, is_target]
        if use_taylor:
            # h_ptv_t -= multiply(multiply(beta[t,is_target], d_parm[t,is_target]), 2*u[t]*d_static[t,is_target] - d_parm[t,is_target])
            h_ptv_t -= multiply(multiply(beta[t, is_target], d_parm[t, is_target]),
                                2 * d[t, is_target] - d_parm[t, is_target])
        if use_dyn_slack:
            h_ptv_t -= h_dyn_slack[t, is_target]
        constrs.append(h[t + 1, is_target] == h_ptv_t)

        # For OAR, relax dynamics constraint to an upper bound that is always tight at optimum.
        constrs.append(h[t + 1, ~is_target] <= h[t, ~is_target] - u[t] * alpha_mul_d_stat[t, ~is_target]
                       - square(u[t]) * beta_mul_d_stat_sq[t, ~is_target] + gamma[t, ~is_target])

    # Additional beam constraints.
    if "beam_constrs" in patient_rx:
        constrs += rx_to_constrs(b, patient_rx["beam_constrs"])

    # Additional dose constraints.
    if "dose_constrs" in patient_rx:
        constrs += rx_to_constrs(d, patient_rx["dose_constrs"])

    # Allow slack in OAR health lower bound constraints.
    if use_bnd_slack:
        h_bnd_slack = Variable((T_treat + T_recov, K), nonneg=True, name="health bound slack")
        obj_bnd_slack = bnd_slack_weight * sum(h_bnd_slack)
    else:
        h_bnd_slack = Constant((T_treat + T_recov, K))
        obj_bnd_slack = 0

    # Additional health constraints.
    if "health_constrs" in patient_rx:
        # constrs += rx_to_quad_constrs(h[1:], patient_rx["health_constrs"], patient_rx["is_target"])
        constrs += rx_to_quad_constrs(h[1:], patient_rx["health_constrs"], patient_rx["is_target"],
                                      slack_lower=h_bnd_slack[:T_treat])

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
            # constrs_r += rx_to_quad_constrs(h_r, patient_rx["recov_constrs"], patient_rx["is_target"])
            constrs_r += rx_to_quad_constrs(h_r, patient_rx["recov_constrs"], patient_rx["is_target"],
                                            slack_lower=h_bnd_slack[T_treat:])
        constrs += constrs_r

    obj_slack = obj_dyn_slack + obj_bnd_slack
    return constrs, d_parm, obj_slack, h_dyn_slack, h_bnd_slack
