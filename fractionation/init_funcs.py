import cvxpy
import numpy as np
import cvxpy.settings as cvxpy_s
from cvxpy import *

from fractionation.problem.dyn_prob import dose_penalty, health_penalty, rx_slice
from fractionation.quadratic.dyn_quad_prob import rx_to_quad_constrs, form_taylor_constrs
from fractionation.utilities.data_utils import check_quad_vectors

def dyn_init_dose(A_list, alpha, beta, gamma, h_init, patient_rx, T_recov = 0, use_slack = False, slack_weight = 0, *args, **kwargs):
    T_treat = len(A_list)
    K, n = A_list[0].shape
    alpha, beta, gamma = check_quad_vectors(alpha, beta, gamma, K, T_treat, T_recov)

    # Problem parameters.
    max_iter = kwargs.pop("max_iter", 50)   # Maximum iterations.
    ccp_eps = kwargs.pop("ccp_eps", 1e-3)   # Stopping tolerance.
    ccp_verbose = kwargs.pop("ccp_verbose", False)
    init_verbose = kwargs.pop("init_verbose", False)
    solve_time = 0
    t_static = 0

    # Stage 1: Solve static (convex) problem in initial session.
    if init_verbose:
        print("Stage 1: Solving static problem in session {0}".(t_static))
    prob_1, b, h, d, h_slack = build_stat_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx, t_static = t_static, use_slack = use_slack, 
                                                    slack_weight = slack_weight)
    prob_1.solve(*args, **kwargs)
    if prob_1.status not in cvxpy_s.SOLUTION_PRESENT:
        raise RuntimeError("Stage 1: Solver failed with status {0}".format(prob_1.status))
    solve_time += prob_1.solver_stats.solve_time
    b_static = b.value   # Save optimal static beams.

    # Stage 2a: Solve for best constant scaling factor u^{const} >= 0.
    if init_verbose:
        print("Stage 2a: Solving dynamic problem for best constant beam scaling factor")
    prob_2a, u, b, h, d, h_lin_slack = build_scale_lin_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx, b_static, use_slack = use_slack, 
                                                    slack_weight = slack_weight)
    prob_2a.solve(*args, **kwargs)
    if prob_2a.status not in cvxpy_s.SOLUTION_PRESENT:
        raise RuntimeError("Stage 2a: Initial solve failed with status {0}".format(prob_2a.status))
    solve_time += prob_2a.solver_stats.solve_time
    d_init = d.value
    if init_verbose:
        print("Stage 2a: Optimal scaling factor = {0}".format(u.value))
        print("Stage 2a: Optimal doses = {0}".format(d.value))

    # Stage 2b: Solve dynamic (nonconvex) problem with scaled static beams.
    if init_verbose:
        print("Stage 2b: Solving dynamic problem for best time-varying beam scaling factors")
    prob_2b, u, b, h, d, d_parm, h_dyn_slack = build_scale_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx, b_static, use_slack = use_slack, 
                                                    slack_weight = slack_weight)

    # Initialize CCP at d_t^0 = A_t*u^{const}*b^{static}.
    result = ccp_solve(prob_2b, d, d_parm, d_init, h_dyn_slack, ccp_verbose, full_hist = False, max_iter = max_iter, ccp_eps = ccp_eps, *args, **kwargs)
    if result["status"] not in cvxpy_s.SOLUTION_PRESENT:
        raise RuntimeError("Stage 2b: CCP solve failed with status {0}".format(result["status"]))
    solve_time += result["solve_time"]
    return {"beams": b.value, "doses": d.value, "solve_time": solve_time}

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

# Static optimal control problem in session t_static.
# OAR health status is linear-quadratic, while PTV health status is linear (beta = 0) with an optional slack term.
def build_stat_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx, t_static = 0, use_slack = False, slack_weight = 0):
    T_treat = len(A_list)
    K, n = A_list[0].shape
    if h_init.shape[0] != K:
        raise ValueError("h_init must be a vector of {0} elements".format(K))
    if t_static < 0 or t_static >= T or int(t_static) != t_static:
        raise ValueError("t_static must be an integer in [0, {0}]".format(T-1))
    
    # Extract parameters for session t_static.
    A_t     = A_list[t_static]
    alpha_t = alpha[t_static]
    beta_t  = beta[t_static]
    gamma_t = gamma[t_static]
    patient_rx_t = rx_slice(patient_rx, t_static, t_static + 1)

    # Define variables.
    b = Variable((n,), nonneg=True, name="beams")  # Beams per session.
    d = A_t @ b                                    # Dose per session.

    # Health status after treatment.
    h_lin = h_init - multiply(alpha_t, d) + gamma_t   # Linear terms only.
    h_quad = h_lin - multiply(beta_t, square(d))      # Linear-quadratic dynamics.

    # Allow slack in PTV health dynamics.
    if use_slack:
        h_slack = Variable((T_treat, K), nonneg=True, name="health dynamics slack")
        obj_slack = slack_weight * sum(h_slack)  # TODO: Set slack weight relative to overall health penalty.
        h_approx = h_lin - h_slack
    else:
        h_slack = Constant(0)
        obj_slack = 0

    # Approximate PTV health status by linearizing around d = 0 (drop quadratic dose term).
    # Keep modeling OAR health status with linear-quadratic dynamics.
    h = multiply(patient_rx_t["is_target"], h_approx) + multiply(~patient_rx_t["is_target"], h_quad)

    # Objective function.
    obj_base = dyn_stat_obj(d, h, patient_rx_t)

    # Additional beam constraints.
    if "beam_constrs" in patient_rx_t:
        constrs += rx_to_constrs(b, patient_rx_t["beam_constrs"])

    # Additional dose constraints.
    if "dose_constrs" in patient_rx_t:
        constrs += rx_to_constrs(d, patient_rx_t["dose_constrs"])

    # Additional health constraints.
    if "health_constrs" in patient_rx_t:
        constrs += rx_to_quad_constrs(h, patient_rx_t["health_constrs"], patient_rx_t["is_target"])

    obj = obj_base + obj_slack
    prob = Problem(Minimize(obj), constrs)
    return prob, b, h, d, h_slack

# Scaled beam problem with b_t = u_t*b^{static}, where u_t >= 0 are scaling factors and b^{static} is a beam constant.
def build_scale_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx, b_static, use_slack = False, slack_weight = 0):
    T_treat = len(A_list)
    K, n = A_list[0].shape
    if h_init.shape[0] != K:
        raise ValueError("h_init must be a vector of {0} elements".format(K))
    if b_static.shape[0] != n:
        raise ValueError("b_static must be a vector of {0} elements".format(n))
    if not np.all(b_static >= 0):
        raise ValueError("b_static must be a nonnegative vector")

    # Define variables.
    u = Variable((T,), nonneg=True, name="beam weights")   # Beam scaling factors.
    h = Variable((T_treat + 1, K), name="health")          # Health statuses.

    b_list = []
    d_list = []
    for t in range(T_treat):
        d_static = A_list[t] @ b_static
        b_list.append(u[t] * b_static)   # b_t = u_t * b_static
        d_list.append(u[t] * d_static)   # d_t = A_tb_t = u_t * (A_tb_static)
    b = vstack(b_list)   # Beams.
    d = vstack(d_list)   # Doses.
    
    # Objective function.
    obj_base = dyn_quad_obj(d, h, patient_rx)

    # Form constraints with slack.
    constrs, d_parm, obj_slack, h_dyn_slack = form_taylor_constrs(b, h, d, alpha, beta, gamma, h_init, patient_rx, T_treat, 
                                                T_recov = 0, use_slack = use_slack, slack_weight = slack_weight)

    obj = obj_base + obj_slack
    prob = Problem(Minimize(obj), constrs)
    return prob, u, b, h, d, d_parm, h_dyn_slack

# Scaled beam problem with b_t = u*b^{static}, where u >= 0 is a scaling factor and b^{static} is a beam constant.
def build_scale_const_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx, b_static, use_taylor = True, use_slack = False, slack_weight = 0):
    T_treat = len(A_list)
    K, n = A_list[0].shape
    if h_init.shape[0] != K:
        raise ValueError("h_init must be a vector of {0} elements".format(K))
    if b_static.shape[0] != n:
        raise ValueError("b_static must be a vector of {0} elements".format(n))
    if not np.all(b_static >= 0):
        raise ValueError("b_static must be a nonnegative vector")

    # Define variables.
    u = Variable(nonneg=True, name="beam weight")         # Beam scaling factor.
    h = Variable((T_treat + 1, K), name="health")         # Health statuses.
    b = u * b_static                                      # Beams per session.
    d = vstack([A_list[t] @ b for t in range(T_treat)])   # Doses.

    # Objective function.
    obj_base = dyn_quad_obj(d, h, patient_rx)

    # Form constraints with slack.
    constrs, d_parm, obj_slack, h_dyn_slack = form_dyn_constrs(b, h, d, alpha, beta, gamma, h_init, patient_rx, T_treat, 
                                                T_recov = 0, use_taylor = use_taylor, use_slack = use_slack, slack_weight = slack_weight)

    obj = obj_base + obj_slack
    prob = Problem(Minimize(obj), constrs)
    return prob, u, b, h, d, d_parm, h_dyn_slack

# Scaled beam problem with constant scaling factor and linear model (beta_t = 0) of PTV health status.
def build_scale_lin_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx, b_static, use_slack = False, slack_weight = 0):
    prob, u, b, h, d, d_parm, h_dyn_slack = build_scale_const_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx, b_static, 
            use_taylor = False, use_slack = use_slack, slack_weight = slack_weight)
    return prob, u, b, h, d, h_dyn_slack
