import cvxpy.settings as cvxpy_s

from adarad.optimization.ccp_funcs import ccp_solve
from adarad.optimization.dose_init.static import build_stat_init_prob
from adarad.optimization.dose_init.scale_const import build_scale_lin_init_prob
from adarad.optimization.dose_init.scale_var import build_scale_init_prob
from adarad.utilities.data_utils import check_quad_vectors

def dyn_init_dose(A_list, alpha, beta, gamma, h_init, patient_rx, T_recov = 0, use_dyn_slack = False, dyn_slack_weight = 0,
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
    prob_2a, u, b, h, d, h_lin_dyn_slack, h_lin_bnd_slack = \
        build_scale_lin_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx, b_static, use_dyn_slack = use_dyn_slack,
                                  dyn_slack_weight= dyn_slack_weight, use_bnd_slack = True, bnd_slack_weight= slack_bnd_weight)
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
    prob_2b, u, b, h, d, d_parm, h_dyn_slack, h_bnd_slack = \
        build_scale_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx, b_static, use_dyn_slack = use_dyn_slack,
                              dyn_slack_weight= dyn_slack_weight, use_bnd_slack = True, bnd_slack_weight= slack_bnd_weight)

    # Initialize CCP at d_t^0 = A_t*u^{const}*b^{static}.
    result = ccp_solve(prob_2b, d, d_parm, d_init, h_dyn_slack, ccp_verbose, full_hist = False, max_iter = max_iter,
                       ccp_eps = ccp_eps, *args, **kwargs)
    if result["status"] not in cvxpy_s.SOLUTION_PRESENT:
        raise RuntimeError("Stage 2b: CCP solve failed with status {0}".format(result["status"]))
    total_time = result["total_time"] + setup_time + solve_time
    solve_time += result["solve_time"]
    return {"beams": b.value, "doses": d.value, "total_time": total_time, "solve_time": solve_time}
