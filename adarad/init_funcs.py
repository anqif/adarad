import numpy as np
from warnings import warn

import cvxpy
from cvxpy import *
import cvxpy.settings as cvxpy_s

from adarad.ccp_funcs import ccp_solve
from adarad.problem.dyn_quad_prob import dyn_quad_obj
from adarad.problem.penalty import dose_penalty, health_penalty
from adarad.problem.constraint import *
from adarad.utilities.data_utils import check_quad_vectors

# Pos penalty function.
def pos_penalty(var, goal=None, weight=None):
    if goal is None:
        goal = np.zeros(var.shape)
    if weight is None:
        weight = np.ones(var.shape)
    # if weight.shape[1] != var.shape[0]:
    #    raise ValueError("weight must have {0} columns".format(var.shape[0]))
    if np.any(weight < 0):
        raise ValueError("weight must be nonnegative")

    return weight @ pos(var - goal)

# Neg penalty function.
def neg_penalty(var, goal=None, weight=None):
    if goal is None:
        goal = np.zeros(var.shape)
    if weight is None:
        weight = np.ones(var.shape)
    # if weight.shape[1] != var.shape[0]:
    #    raise ValueError("weight must have {0} columns".format(var.shape[0]))
    if np.any(weight < 0):
        raise ValueError("weight must be nonnegative")

    return weight @ neg(var - goal)

def dyn_init_dose(A_list, alpha, beta, gamma, h_init, patient_rx, T_recov = 0, use_dyn_slack = False, slack_dyn_weight = 0,
                  slack_bnd_weight = 1, *args, **kwargs):
    T_treat = len(A_list)
    K, n = A_list[0].shape
    alpha, beta, gamma = check_quad_vectors(alpha, beta, gamma, K, T_treat, T_recov)

    # Problem parameters.
    max_iter = kwargs.pop("max_iter", 50)   # Maximum iterations.
    ccp_eps = kwargs.pop("ccp_eps", 1e-3)   # Stopping tolerance.
    ccp_verbose = kwargs.pop("ccp_verbose", False)
    init_verbose = kwargs.pop("init_verbose", False)
    setup_time = 0
    solve_time = 0
    t_static = 0

    # Stage 1: Solve static (convex) problem in initial session.
    if init_verbose:
        print("Stage 1: Solving static problem in session {0}".format(t_static))
    prob_1, b, h, d, h_actual, h_slack = build_stat_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx, t_static = t_static)
    # prob_1, b, h, d, h_actual, h_slack = build_stat_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx, t_static = t_static,
    #     use_slack = use_slack, slack_weight = slack_weight/T_treat)
    prob_1.solve(*args, **kwargs)
    if prob_1.status not in cvxpy_s.SOLUTION_PRESENT:
        raise RuntimeError("Stage 1: Solver failed with status {0}".format(prob_1.status))
    setup_time += prob_1.solver_stats.setup_time if prob_1.solver_stats.setup_time else 0
    solve_time += prob_1.solver_stats.solve_time if prob_1.solver_stats.solve_time else 0
    # b_static = b.value/T_treat   # Save optimal static beams.
    b_static = b.value

    # Stage 2a: Solve for best constant scaling factor u^{const} >= 0.
    if init_verbose:
        print("Stage 2a: Solving dynamic problem for best constant beam scaling factor")
    prob_2a, u, b, h, d, h_lin_dyn_slack, h_lin_bnd_slack = build_scale_lin_init_prob(A_list, alpha, beta, gamma,
        h_init, patient_rx, b_static, use_dyn_slack = use_dyn_slack, slack_dyn_weight = slack_dyn_weight,
        use_bnd_slack = True, slack_bnd_weight = slack_bnd_weight)
    prob_2a.solve(*args, **kwargs)
    if prob_2a.status not in cvxpy_s.SOLUTION_PRESENT:
        raise RuntimeError("Stage 2a: Initial solve failed with status {0}".format(prob_2a.status))
    setup_time += prob_2a.solver_stats.setup_time if prob_2a.solver_stats.setup_time else 0
    solve_time += prob_2a.solver_stats.solve_time if prob_2a.solver_stats.solve_time else 0
    d_init = d.value
    if init_verbose:
        print("Stage 2a: Optimal scaling factor = {0}".format(u.value))
        # print("Stage 2a: Optimal doses = {0}".format(d.value))

    # Stage 2b: Solve dynamic (nonconvex) problem with scaled static beams.
    if init_verbose:
        print("Stage 2b: Solving dynamic problem for best time-varying beam scaling factors")
    prob_2b, u, b, h, d, d_parm, h_dyn_slack, h_bnd_slack = build_scale_init_prob(A_list, alpha, beta, gamma, h_init,
        patient_rx, b_static, use_dyn_slack = use_dyn_slack, slack_dyn_weight = slack_dyn_weight, use_bnd_slack = True,
        slack_bnd_weight = slack_bnd_weight)

    # Initialize CCP at d_t^0 = A_t*u^{const}*b^{static}.
    result = ccp_solve(prob_2b, d, d_parm, d_init, h_dyn_slack, ccp_verbose, full_hist = False, max_iter = max_iter,
                       ccp_eps = ccp_eps, *args, **kwargs)
    if result["status"] not in cvxpy_s.SOLUTION_PRESENT:
        raise RuntimeError("Stage 2b: CCP solve failed with status {0}".format(result["status"]))
    total_time = result["total_time"] + setup_time + solve_time
    solve_time += result["solve_time"]
    return {"beams": b.value, "doses": d.value, "total_time": total_time, "solve_time": solve_time}

# Objective function for static problem.
def dyn_stat_obj(d_var, h_var, patient_rx):
    K = d_var.shape[0]
    if h_var.shape[0] != K:
        raise ValueError("h_var must have exactly {0} rows".format(K))
    if patient_rx["dose_goal"].shape not in [(K,), (K,1)]:
        raise ValueError("dose_goal must have dimensions ({0},)".format(K))
    if patient_rx["health_goal"].shape not in [(K,), (K,1)]:
        raise ValueError("health_goal must have dimensions ({0},)".format(K))

    d_penalty = dose_penalty(d_var, patient_rx["dose_goal"], patient_rx["dose_weights"])
    h_penalty = health_penalty(h_var, patient_rx["health_goal"], patient_rx["health_weights"])
    return d_penalty + h_penalty

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

# Health bound constraints for PTV.
def rx_to_ptv_constrs(h_ptv, rx_dict_ptv, slack_upper = 0):
    constrs = []

    # Lower bound.
    if "lower" in rx_dict_ptv and not np.all(np.isneginf(rx_dict_ptv["lower"])):
        raise ValueError("Lower bound must be negative infinity for all targets")

    # Upper bound.
    if "upper" in rx_dict_ptv:
        c_upper = rx_to_upper_constrs(h_ptv, rx_dict_ptv["upper"], only_ptv = True, slack = slack_upper)
        if c_upper is not None:
            constrs.append(c_upper)
    return constrs

# Health bound constraints for OAR.
def rx_to_oar_constrs(h_oar, rx_dict_oar, slack_lower = 0):
    constrs = []

    # Lower bound.
    if "lower" in rx_dict_oar:
        c_lower = rx_to_lower_constrs(h_oar, rx_dict_oar["lower"], only_oar = True, slack = slack_lower)
        if c_lower is not None:
            constrs.append(c_lower)

    # Upper bound.
    if "upper" in rx_dict_oar and not np.all(np.isinf(rx_dict_oar["upper"])):
        raise ValueError("Upper bound must be infinity for all non-targets")
    return constrs

# def rx_to_oar_constrs_slack(h_oar, rx_dict_oar):
#     if "upper" in rx_dict_oar and not np.all(np.isinf(rx_dict_oar["upper"])):
#         raise ValueError("Upper bound must be infinity for all non-targets")
#
#     constrs = []
#     h_slack = Constant(0)
#     if "lower" in rx_dict_oar:
#         rx_lower = rx_dict_oar["lower"]
#         if np.any(rx_lower == np.inf):
#             raise ValueError("Lower bound cannot be infinity")
#
#         if np.isscalar(rx_lower):
#             if np.isfinite(rx_lower):
#                 h_slack = Variable(h_oar.shape, nonneg=True, name="OAR health lower bound slack")
#                 constrs.append(h_oar >= rx_lower - h_slack)
#         else:
#             if rx_lower.shape != h_oar.shape:
#                 raise ValueError("rx_lower must have dimensions {0}".format(h_oar.shape))
#             is_finite = np.isfinite(rx_lower)
#             if np.any(is_finite):
#                 h_slack = Variable(h_oar.shape, nonneg=True, name="OAR health lower bound slack")
#                 constrs.append(h_oar[is_finite] >= rx_lower[is_finite] - h_slack[is_finite])
#     return constrs, h_slack

def constr_sum_upper(expr, upper, T_treat):
    n = expr.shape[0]
    if np.isscalar(upper):
        if np.isfinite(upper):
            return [expr <= T_treat * upper]
        else:
            return []
    elif upper.shape == (T_treat, n):
        upper_sum = np.sum(upper, axis = 0)
        is_finite = np.isfinite(upper_sum)
        if np.any(is_finite):
            return [expr[is_finite] <= upper_sum[is_finite]]
        else:
            return []
    else:
        raise TypeError("Upper bound must be a scalar or array with dimensions ({0},{1})".format(T_treat, n))

# Static optimal control problem in session t_static.
# OAR health status is linear-quadratic, while PTV health status is linear (beta = 0) with an optional slack term.
def build_stat_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx, t_static = 0, slack_oar_weight = 1):
    T_treat = len(A_list)
    K, n = A_list[0].shape
    if h_init.shape[0] != K:
        raise ValueError("h_init must be a vector of {0} elements".format(K))
    if t_static < 0 or t_static >= T_treat or int(t_static) != t_static:
        raise ValueError("t_static must be an integer in [0, {0}]".format(T_treat-1))
    
    # Extract parameters for session t_static.
    A_t     = A_list[t_static]
    alpha_t = alpha[t_static]
    beta_t  = beta[t_static]
    gamma_t = gamma[t_static]
    patient_rx_t = rx_slice(patient_rx, t_static, t_static + 1, squeeze = True)

    # Define variables.
    b = Variable((n,), nonneg=True, name="beams")  # Beams.
    d = A_t @ b                                    # Dose.

    # Health status after treatment.
    h_lin = h_init - multiply(alpha_t, d) + gamma_t   # Linear terms only.
    h_quad = h_lin - multiply(beta_t, square(d))      # Linear-quadratic dynamics.

    # Approximate PTV health status by linearizing around d = 0 (drop quadratic dose term).
    # Keep modeling OAR health status with linear-quadratic dynamics.
    is_target = patient_rx_t["is_target"]
    h_app_ptv = h_lin[is_target]
    h_app_oar = h_quad[~is_target]
    h_app = multiply(h_lin, is_target) + multiply(h_quad, ~is_target)
    
    # Objective function.
    d_penalty = dose_penalty(d, patient_rx_t["dose_goal"], patient_rx_t["dose_weights"])
    h_penalty_ptv = pos_penalty(h_app_ptv, patient_rx_t["health_goal"][is_target], patient_rx_t["health_weights"][1][is_target])
    # h_penalty_oar = hinge_penalty(h_app_oar[~is_target], patient_rx_t["health_goal"][~is_target], [w[~is_target] for w in patient_rx_t["health_weights"]])
    h_penalty_oar = neg_penalty(h_app_oar, patient_rx_t["health_goal"][~is_target], patient_rx_t["health_weights"][0][~is_target])
    h_penalty = h_penalty_ptv + h_penalty_oar
    obj_base = d_penalty + h_penalty

    # Additional beam constraints.
    constrs = []
    if "beam_constrs" in patient_rx and "upper" in patient_rx["beam_constrs"]:
        beam_upper = patient_rx["beam_constrs"]["upper"]
        if np.any(beam_upper < 0):
            raise ValueError("Beam upper bound must be nonnegative")
        constrs += constr_sum_upper(b, beam_upper, T_treat)

    # Additional dose constraints.
    if "dose_constrs" in patient_rx and "upper" in patient_rx["dose_constrs"]:
        dose_upper = patient_rx["dose_constrs"]["upper"]
        if np.any(dose_upper < 0):
            raise ValueError("Dose upper bound must be nonnegative")
        constrs += constr_sum_upper(d, dose_upper, T_treat)

    # Additional health constraints.
    if "health_constrs" in patient_rx:
        # Health bounds from final treatment session.
        patient_rx_fin = rx_slice(patient_rx, T_treat - 1, T_treat, squeeze = True)
        rx_fin_health_constrs_ptv = get_constrs_by_struct(patient_rx_fin["health_constrs"], is_target, struct_dim = 0)
        rx_fin_health_constrs_oar = get_constrs_by_struct(patient_rx_fin["health_constrs"], ~is_target, struct_dim = 0)

        # Add slack to lower bound on health of OARs, but keep strict upper bound on health of PTVs.
        constrs += rx_to_ptv_constrs(h_app_ptv, rx_fin_health_constrs_ptv)
        # constrs += rx_to_oar_constrs(h_app_oar, rx_fin_health_constrs_oar)
        h_slack = Variable((K,), nonneg=True)
        constrs += rx_to_oar_constrs(h_app_oar, rx_fin_health_constrs_oar, slack_lower = h_slack[~is_target])
        obj_slack = slack_oar_weight*sum(h_slack)
        # constr_oar, h_oar_slack = rx_to_oar_constrs_slack(h_app_oar, rx_fin_health_constrs_oar)
        # constrs += constr_oar
        # obj_slack = slack_oar_weight*sum(h_oar_slack)
    else:
        # h_oar_slack = Constant(np.zeros(h_app_oar.shape))
        h_slack = Constant(np.zeros((K,)))
        obj_slack = 0

    obj = obj_base + obj_slack
    prob = Problem(Minimize(obj), constrs)
    return prob, b, h_app, d, h_quad, h_slack

# Scaled beam problem with b_t = u_t*b^{static}, where u_t >= 0 are scaling factors and b^{static} is a beam constant.
def build_scale_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx, b_static, use_dyn_slack = False,
                          slack_dyn_weight = 0, use_bnd_slack = True, slack_bnd_weight = 1):
    T_treat = len(A_list)
    K, n = A_list[0].shape
    if h_init.shape[0] != K:
        raise ValueError("h_init must be a vector of {0} elements".format(K))
    if b_static.shape[0] != n:
        raise ValueError("b_static must be a vector of {0} elements".format(n))
    if not np.all(b_static >= 0):
        raise ValueError("b_static must be a nonnegative vector")

    # Define variables.
    u = Variable((T_treat,), nonneg=True, name="beam weights")   # Beam scaling factors.
    h = Variable((T_treat + 1, K), name="health")          # Health statuses.

    b_list = []
    d_list = []
    d_static_list = []
    for t in range(T_treat):
        d_static_t = A_list[t] @ b_static
        b_list.append(u[t] * b_static)     # b_t = u_t * b_static
        d_list.append(u[t] * d_static_t)   # d_t = A_tb_t = u_t * (A_tb_static)
        d_static_list.append(d_static_t)
    b = vstack(b_list)   # Beams.
    d = vstack(d_list)   # Doses.
    d_static = np.vstack(d_static_list)
    
    # Objective function.
    obj_base = dyn_quad_obj(d, h, patient_rx)

    # Form constraints with slack.
    constrs, d_parm, obj_slack, h_dyn_slack, h_bnd_slack = form_scale_constrs(b, h, u, d_static, alpha, beta, gamma,
            h_init, patient_rx, T_recov = 0, use_taylor = True, use_dyn_slack = use_dyn_slack,
            slack_dyn_weight = slack_dyn_weight, use_bnd_slack = use_bnd_slack, slack_bnd_weight = slack_bnd_weight)

    obj = obj_base + obj_slack
    prob = Problem(Minimize(obj), constrs)
    return prob, u, b, h, d, d_parm, h_dyn_slack, h_bnd_slack

# Scaled beam problem with b_t = u*b^{static}, where u >= 0 is a scaling factor and b^{static} is a beam constant.
def build_scale_const_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx, b_static, use_taylor = True,
                                use_dyn_slack = False, slack_dyn_weight = 0, use_bnd_slack = True, slack_bnd_weight = 1):
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
    constrs, d_parm, obj_slack, h_dyn_slack, h_bnd_slack = form_scale_const_constrs(b, h, u, d_static, alpha, beta, gamma,
            h_init, patient_rx_bcond, T_recov = 0, use_taylor = use_taylor, use_dyn_slack = use_dyn_slack,
            slack_dyn_weight = slack_dyn_weight, use_bnd_slack = use_bnd_slack, slack_bnd_weight = slack_bnd_weight)

    obj = obj_base + obj_slack
    prob = Problem(Minimize(obj), constrs)
    return prob, u, b, h, d, d_parm, h_dyn_slack, h_bnd_slack

# Scaled beam problem with constant scaling factor (u >= 0), linear model (beta_t = 0) of PTV health status,
# and slack in lower bound on OAR health status.
def build_scale_lin_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx, b_static, use_dyn_slack = False,
                              slack_dyn_weight = 0, use_bnd_slack = True, slack_bnd_weight = 1):
    prob, u, b, h, d, d_parm, h_dyn_slack, h_bnd_slack = build_scale_const_init_prob(A_list, alpha, beta, gamma, h_init,
        patient_rx, b_static, use_taylor = False, use_dyn_slack = use_dyn_slack, slack_dyn_weight = slack_dyn_weight,
        use_bnd_slack = use_bnd_slack, slack_bnd_weight = slack_bnd_weight)
    return prob, u, b, h, d, h_dyn_slack, h_bnd_slack

def form_scale_constrs(b, h, u, d_static, alpha, beta, gamma, h_init, patient_rx, T_recov = 0, use_taylor = True,
                        use_dyn_slack = False, slack_dyn_weight = 0, use_bnd_slack = True, slack_bnd_weight = 1):
    T_treat, K = d_static.shape
    d = vstack([u[t] * d_static[t,:] for t in range(T_treat)])

    # Define variables.
    if use_taylor:
        d_parm = Parameter((T_treat, K), nonneg=True, name="dose parameter")  # Dose point around which to linearize dynamics.
    else:
        d_parm = Constant(np.zeros(T_treat,K))

    # Allow slack in PTV health dynamics constraints.
    if use_dyn_slack:
        h_dyn_slack = Variable((T_treat, K), nonneg=True, name="health dynamics slack")
        obj_dyn_slack = slack_dyn_weight*sum(h_dyn_slack)   # TODO: Set slack weight relative to overall health penalty.
    else:
        h_dyn_slack = Constant(np.zeros((T_treat, K)))
        obj_dyn_slack = 0

    # Pre-compute constant expressions.
    is_target = patient_rx["is_target"]
    alpha_mul_d_stat = multiply(alpha, d_static).value
    beta_mul_d_stat_sq = multiply(beta, square(d_static)).value

    # Health dynamics for treatment stage.
    constrs = [h[0] == h_init]
    for t in range(T_treat):
        # For PTV, approximate dynamics via a first-order Taylor expansion.
        h_ptv_t = h[t,is_target] - u[t]*alpha_mul_d_stat[t,is_target] + gamma[t,is_target]
        if use_taylor:
            # h_ptv_t -= multiply(multiply(beta[t,is_target], d_parm[t,is_target]), 2*u[t]*d_static[t,is_target] - d_parm[t,is_target])
            h_ptv_t -= multiply(multiply(beta[t,is_target], d_parm[t,is_target]), 2*d[t,is_target] - d_parm[t,is_target])
        if use_dyn_slack:
            h_ptv_t -= h_dyn_slack[t,is_target]
        constrs.append(h[t+1,is_target] == h_ptv_t)

        # For OAR, relax dynamics constraint to an upper bound that is always tight at optimum.
        constrs.append(h[t+1,~is_target] <= h[t,~is_target] - u[t]*alpha_mul_d_stat[t,~is_target]
                                                - square(u[t])*beta_mul_d_stat_sq[t,~is_target] + gamma[t,~is_target])

    # Additional beam constraints.
    if "beam_constrs" in patient_rx:
        constrs += rx_to_constrs(b, patient_rx["beam_constrs"])

    # Additional dose constraints.
    if "dose_constrs" in patient_rx:
        constrs += rx_to_constrs(d, patient_rx["dose_constrs"])

    # Allow slack in OAR health lower bound constraints.
    if use_bnd_slack:
        h_bnd_slack = Variable((T_treat + T_recov, K), nonneg=True, name="health bound slack")
        obj_bnd_slack = slack_bnd_weight * sum(h_bnd_slack)
    else:
        h_bnd_slack = Constant((T_treat + T_recov, K))
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
            constrs_r.append(h_r[t + 1] == h_r[t] + gamma_r[t + 1])

        # Additional health constraints during recovery.
        if "recov_constrs" in patient_rx:
            # constrs_r += rx_to_quad_constrs(h_r, patient_rx["recov_constrs"], patient_rx["is_target"])
            constrs_r += rx_to_quad_constrs(h_r, patient_rx["recov_constrs"], patient_rx["is_target"], slack_lower = h_bnd_slack[T_treat:])
        constrs += constrs_r

    obj_slack = obj_dyn_slack + obj_bnd_slack
    return constrs, d_parm, obj_slack, h_dyn_slack, h_bnd_slack

def form_scale_const_constrs(b, h, u, d_static, alpha, beta, gamma, h_init, patient_rx, T_recov = 0, use_taylor = True,
                             use_dyn_slack = False, slack_dyn_weight = 0, use_bnd_slack = True, slack_bnd_weight = 1):
    T_treat, K = d_static.shape

    # Health dynamics for treatment stage.
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
        obj_dyn_slack = slack_dyn_weight*sum(h_dyn_slack)   # TODO: Set slack weight relative to overall health penalty.
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
        obj_bnd_slack = slack_bnd_weight * sum(h_bnd_slack)
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
