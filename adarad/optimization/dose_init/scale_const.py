import numpy as np
from cvxpy import *
from warnings import warn

from adarad.optimization.penalty import dose_penalty, health_penalty
from adarad.optimization.constraint import rx_to_constrs, rx_to_quad_constrs

# Objective function for constant scaled problem.
def dyn_scale_const_obj(u_var, d_static, h_var, patient_rx, fast_ssq = False):
    T_plus_1, K = h_var.shape
    T = T_plus_1 - 1
    if d_static.shape != (T, K):
        raise ValueError("d_static must have dimensions ({0},{1})".format(T, K))
    if patient_rx["dose_goal"].shape != (T, K):
        raise ValueError("dose_goal must have dimensions ({0},{1})".format(T, K))
    if patient_rx["health_goal"].shape != (T, K):
        raise ValueError("health_goal must have dimensions ({0},{1})".format(T, K))

    # Fast handling of total dose penalty = \sum_{t,k} w_k * d_{tk}^2,
    # i.e., case when dose_penalty = square_penalty and dose_goal = 0.
    if fast_ssq:
        if np.count_nonzero(patient_rx["dose_goal"]) != 0:
            warn("Ignoring dose_goal even though not all elements are zero.")

        d_weights = patient_rx["dose_weights"]
        if d_weights is None:
            d_weights = np.ones(K)
        if d_weights.shape not in [(K,), (K,1)]:
            raise ValueError("dose_weights must have exactly {0} rows".format(K))
        if np.any(d_weights < 0):
            raise ValueError("dose_weights must all be nonnegative")

        const_vec_ssq = np.array([sum_squares(d_static[:,k]).value for k in range(K)])
        const_ssq = d_weights @ const_vec_ssq
        d_penalty = square(u_var) * const_ssq
        h_penalties = [health_penalty(h_var[t + 1], patient_rx["health_goal"][t], patient_rx["health_weights"]) for t in range(T)]
        return d_penalty + sum(h_penalties)
    else:
        penalties = []
        for t in range(T):
            d_penalty = dose_penalty(u_var * d_static[t], patient_rx["dose_goal"][t], patient_rx["dose_weights"])
            h_penalty = health_penalty(h_var[t + 1], patient_rx["health_goal"][t], patient_rx["health_weights"])
            penalties.append(d_penalty + h_penalty)
        return sum(penalties)

# Scaled beam problem with constant scaling factor (u >= 0), linear model (beta_t = 0) of PTV health status,
# and slack in lower bound on OAR health status.
def build_scale_lin_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx, b_static, use_dyn_slack = False,
                              dyn_slack_weight = 0, use_bnd_slack = True, bnd_slack_weight = 1):
    prob, u, b, h, d, d_parm, h_dyn_slack, h_bnd_slack = \
        build_scale_const_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx, b_static, use_taylor = False,
                                    use_dyn_slack = use_dyn_slack, dyn_slack_weight= dyn_slack_weight,
                                    use_bnd_slack = use_bnd_slack, bnd_slack_weight= bnd_slack_weight)
    return prob, u, b, h, d, h_dyn_slack, h_bnd_slack

# Scaled beam problem with b_t = u*b^{static}, where u >= 0 is a scaling factor and b^{static} is a beam constant.
def build_scale_const_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx, b_static, use_taylor = True,
                                use_dyn_slack = False, dyn_slack_weight = 0, use_bnd_slack = True, bnd_slack_weight = 1):
    T_treat = len(A_list)
    K, n = A_list[0].shape
    if h_init.shape[0] != K:
        raise ValueError("h_init must be a vector of {0} elements".format(K))
    if b_static.shape[0] != n:
        raise ValueError("b_static must be a vector of {0} elements".format(n))
    if not np.all(b_static >= 0):
        raise ValueError("b_static must be a non-negative vector")

    # Define variables.
    u = Variable(nonneg=True, name="beam weight")         # Beam scaling factor.
    h = Variable((T_treat + 1, K), name="health")         # Health statuses.
    b = u * b_static                                      # Beams per session.
    # d = vstack([A_list[t] @ b for t in range(T_treat)])   # Doses.
    d_static = np.vstack([A_list[t] @ b_static for t in range(T_treat)])
    d = u * d_static

    # Objective function.
    # obj_base = dyn_quad_obj(d, h, patient_rx)
    obj_base = dyn_scale_const_obj(u, d_static, h, patient_rx)

    # Since beams are same across sessions, set constraints to max(B_t^{lower}) <= b <= min(B_t^{upper}),
    # where the max/min are taken over sessions t = 1,...,T.
    patient_rx_bcond = patient_rx
    if "beam_constrs" in patient_rx and patient_rx["beam_constrs"]:
        rx_b = patient_rx["beam_constrs"]
        patient_rx_bcond = patient_rx.copy()
        patient_rx_bcond["beam_constrs"] = {}
        if "lower" in rx_b:
            patient_rx_bcond["beam_constrs"]["lower"] = np.max(rx_b["lower"], axis = 0)
        if "upper" in rx_b:
            patient_rx_bcond["beam_constrs"]["upper"] = np.min(rx_b["upper"], axis = 0)

    # Form constraints with slack.
    constrs, d_parm, obj_slack, h_dyn_slack, h_bnd_slack = \
        form_scale_const_constrs(b, h, u, d_static, alpha, beta, gamma, h_init, patient_rx_bcond, T_recov = 0,
                                 use_taylor = use_taylor, use_dyn_slack = use_dyn_slack, dyn_slack_weight= dyn_slack_weight,
                                 use_bnd_slack = use_bnd_slack, bnd_slack_weight= bnd_slack_weight)

    obj = obj_base + obj_slack
    prob = Problem(Minimize(obj), constrs)
    return prob, u, b, h, d, d_parm, h_dyn_slack, h_bnd_slack

def form_scale_const_constrs(b, h, u, d_static, alpha, beta, gamma, h_init, patient_rx, T_recov = 0, use_taylor = True,
                             use_dyn_slack = False, dyn_slack_weight = 0, use_bnd_slack = True, bnd_slack_weight = 1):
    T_treat, K = d_static.shape

    # Health dynamics for optimization stage.
    h_lin = h[:-1] - u * multiply(alpha, d_static).value + gamma[:T_treat]
    h_quad = h_lin - square(u) * multiply(beta, square(d_static)).value

    d_parm = Constant(np.zeros((T_treat, K)))
    h_approx = h_lin
    if use_taylor:
        d_parm = Parameter((T_treat, K), nonneg=True, name="dose parameter")  # Dose point around which to linearize dynamics.
        h_approx -= multiply(multiply(beta, d_parm), 2*u*d_static - d_parm)     # First-order Taylor expansion of quadratic.

    # Allow slack in PTV health dynamics constraints.
    if use_dyn_slack:
        h_dyn_slack = Variable((T_treat, K), nonneg=True, name="health dynamics slack")
        obj_dyn_slack = dyn_slack_weight * sum(h_dyn_slack)   # TODO: Set slack weight relative to overall health penalty.
        h_approx -= h_dyn_slack
    else:
        h_dyn_slack = Constant(np.zeros((T_treat, K)))
        obj_dyn_slack = 0

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
        constrs += rx_to_constrs(u * d_static, patient_rx["dose_constrs"])

    # Allow slack in OAR health lower bound constraints.
    if use_bnd_slack:
        h_bnd_slack = Variable((T_treat + T_recov, K), nonneg=True, name="health bound slack")
        obj_bnd_slack = bnd_slack_weight * sum(h_bnd_slack)
    else:
        h_bnd_slack = Constant(np.zeros((T_treat + T_recov, K)))
        obj_bnd_slack = 0

    # Additional health constraints.
    if "health_constrs" in patient_rx:
        # constrs += rx_to_quad_constrs(h[1:], patient_rx["health_constrs"], patient_rx["is_target"])
        constrs += rx_to_quad_constrs(h[1:], patient_rx["health_constrs"], patient_rx["is_target"], slack_lower = h_bnd_slack[:T_treat])

    # Health dynamics for recovery stage.
    # TODO: Should we return h_r or calculate it later?
    if T_recov > 0:
        gamma_r = gamma[T_treat:]

        h_r = Variable((T_recov, K), name="recovery")
        constrs_r = [h_r[0] == h[-1] + gamma_r[0]]
        for t in range(T_recov - 1):
            constrs_r.append(h_r[t+1] == h_r[t] + gamma_r[t + 1])

        # Additional health constraints during recovery.
        if "recov_constrs" in patient_rx:
            # constrs_r += rx_to_quad_constrs(h_r, patient_rx["recov_constrs"], patient_rx["is_target"])
            constrs_r += rx_to_quad_constrs(h_r, patient_rx["recov_constrs"], patient_rx["is_target"], slack_lower = h_bnd_slack[T_treat:])
        constrs += constrs_r

    obj_slack = obj_dyn_slack + obj_bnd_slack
    return constrs, d_parm, obj_slack, h_dyn_slack, h_bnd_slack
