import numpy as np
import numpy.linalg as LA
import cvxpy.settings as cvxpy_s

from time import time
from multiprocessing import Process, Pipe
from collections import defaultdict, Counter

from adarad.ccp_funcs import ccp_solve
from adarad.init_funcs import dyn_init_dose
from adarad.mpc_funcs import print_results
from adarad.problem.dyn_prob import rx_slice

from adarad.quadratic.dyn_quad_prob import dyn_quad_obj
from adarad.quadratic.slack_quad_prob import slack_quad_penalty
from adarad.quadratic.slack_quad_prob_admm import *
from adarad.utilities.data_utils import *

def run_slack_quad_dose_worker(pipe, A, patient_rx, rho, *args, **kwargs):
    # Construct proximal dose problem.
    prob_dose, b, d = build_dyn_slack_quad_prob_dose_period(A, patient_rx)
    d_new = Parameter(d.shape, value = np.zeros(d.shape))
    u = Parameter(d.shape, value = np.zeros(d.shape))
    penalty = (rho/2)*sum_squares(d - d_new - u)
    prox = prob_dose + Problem(Minimize(penalty))

    # ADMM loop.
    finished = False
    while not finished:
        # Compute and send d_t^k.
        prox.solve(*args, **kwargs)
        if prox.status not in cvxpy_s.SOLUTION_PRESENT:
            raise RuntimeError("Solver failed on slack problem with status {0}".format(prox.status))
        pipe.send((d.value, prox.solver_stats.solve_time, prox.status))

        # Receive \tilde d_t^k.
        d_new.value = pipe.recv()

        # Update and send u_t^^k.
        u.value += d_new.value - d.value
        pipe.send(u.value)

        # Check if stopped.
        finished = pipe.recv()

    # Send final b_t^k and d_t^k along with slacks s_t^k.
    d_val = A.dot(b.value)
    pipe.send((b.value, d_val))

def dyn_quad_treat_admm_slack(A_list, alpha, beta, gamma, h_init, patient_rx, T_recov = 0, health_map = lambda h,d,t: h, d_init = None,
                              auto_init = False, use_ccp_slack = False, ccp_slack_weight = 0, mpc_slack_weights = 1, partial_results = False,
                              admm_verbose = False, *args, **kwargs):
    T_treat = len(A_list)
    K, n = A_list[0].shape
    alpha, beta, gamma = check_quad_vectors(alpha, beta, gamma, K, T_treat, T_recov)

    # Problem parameters.
    admm_max_iter = kwargs.pop("admm_max_iter", 1000)  # Maximum iterations.
    rho = kwargs.pop("rho", 1 / 10)        # Step size.
    eps_abs = kwargs.pop("eps_abs", 1e-6)  # Absolute stopping tolerance.
    eps_rel = kwargs.pop("eps_rel", 1e-3)  # Relative stopping tolerance.

    ccp_max_iter = kwargs.pop("ccp_max_iter", 50)  # Maximum iterations of CCP.
    ccp_eps = kwargs.pop("ccp_eps", 1e-3)          # Stopping tolerance of CCP.
    ccp_verbose = kwargs.pop("ccp_verbose", False)

    # Validate parameters.
    if admm_max_iter <= 0:
        raise ValueError("admm_max_iter must be a positive integer.")
    if rho <= 0:
        raise ValueError("rho must be a positive scalar.")
    if eps_abs < 0:
        raise ValueError("eps_abs must be a non-negative scalar.")
    if eps_rel < 0:
        raise ValueError("eps_rel must be a non-negative scalar.")

    # Initialize dose.
    solve_time = 0
    if d_init is None:
        if auto_init:
            if admm_verbose:
                print("Calculating initial dose...")
            result_init = dyn_init_dose(A_list, alpha, beta, gamma, h_init, patient_rx, T_recov, use_ccp_slack,
                                        ccp_slack_weight, *args, **kwargs)
            d_init = result_init["doses"]
            solve_time += result_init["solve_time"]
        else:
            d_init = np.zeros((T_treat, K))
    if admm_verbose:
        print("Initial dose per fraction: {0}".format(d_init[0]))

    # Set up dose workers.
    pipes = []
    procs = []
    for t in range(T_treat):
        rx_cur = rx_slice(patient_rx, t, t + 1)   # Get prescription at time t.
        local, remote = Pipe()
        pipes += [local]
        procs += [Process(target=run_slack_quad_dose_worker, args=(remote, A_list[t], rx_cur, rho) + args, kwargs=kwargs)]
        procs[-1].start()

    # Proximal health problem.
    prob_health, h, d_tld, d_parm, h_dyn_slack = build_dyn_slack_quad_prob_health(alpha, beta, gamma, h_init, patient_rx,
                                                    T_treat, T_recov, use_ccp_slack, ccp_slack_weight, mpc_slack_weights)
    d_new = Parameter(d_tld.shape, value=np.zeros(d_tld.shape))
    u = Parameter(d_tld.shape, value=np.zeros(d_tld.shape))
    penalty = (rho / 2) * sum_squares(d_tld - d_new + u)
    prox = prob_health + Problem(Minimize(penalty))

    # ADMM loop.
    k = 0
    finished = False
    r_prim = np.zeros(admm_max_iter)
    r_dual = np.zeros(admm_max_iter)
    status_list = []

    start = time()
    while not finished:
        if admm_verbose and k % 10 == 0:
            print("ADMM Iteration:", k)

        # Collect and stack d_t^k for t = 1,...,T.
        dt_update = [pipe.recv() for pipe in pipes]
        d_rows, d_times, d_statuses = map(list, zip(*dt_update))
        d_new.value = np.row_stack(d_rows)
        solve_time += np.max(d_times)
        d_status, d_status_count = Counter(d_statuses).most_common(1)[0]
        status_list.append(d_status)

        # Compute and send \tilde d_t^k.
        if k == 0:
            d_tld_prev = np.zeros((T_treat, K))
        else:
            d_init = d_tld_prev = d_tld.value
        result = ccp_solve(prox, d_tld, d_parm, d_init, max_iter = ccp_max_iter, ccp_eps = ccp_eps, ccp_verbose = ccp_verbose,
                           *args, **kwargs)
        if result["status"] not in cvxpy_s.SOLUTION_PRESENT:
            raise RuntimeError("CCP solve failed on slack problem with status {0}".format(result["status"]))
        solve_time += result["solve_time"]
        status_list.append(result["status"])
        for t in range(T_treat):
            pipes[t].send(d_tld[t].value)

        # Receive and update u_t^k for t = 1,...,T.
        u_rows = [pipe.recv() for pipe in pipes]
        u.value = np.row_stack(u_rows)

        # Calculate residuals.
        r_prim_mat = d_new.value - d_tld.value
        r_dual_mat = rho * (d_tld.value - d_tld_prev)
        r_prim[k] = LA.norm(r_prim_mat)
        r_dual[k] = LA.norm(r_dual_mat)

        # Check stopping criteria.
        eps_prim = eps_abs * np.sqrt(T_treat * K) + eps_rel * np.maximum(LA.norm(d_new.value), LA.norm(d_tld.value))
        eps_dual = eps_abs * np.sqrt(T_treat * K) + eps_rel * LA.norm(u.value)
        finished = (k + 1) >= admm_max_iter or (r_prim[k] <= eps_prim and r_dual[k] <= eps_dual)
        k = k + 1
        for pipe in pipes:
            pipe.send(finished)

    # Receive final values of b_t^k and d_t^k = A*b_t^k for t = 1,...,T.
    bd_final = [pipe.recv() for pipe in pipes]
    b_rows, d_rows = map(list, zip(*bd_final))
    b_val = np.row_stack(b_rows)
    d_val = np.row_stack(d_rows)

    [proc.terminate() for proc in procs]
    end = time()

    # Take majority as final status.
    status, status_count = Counter(status_list).most_common(1)[0]

    # Only used internally for calls in MPC to save time.
    if partial_results:
        return {"status": status, "num_iters": k, "solve_time": solve_time, "beams": b_val, "doses": d_val}

    # Construct full results.
    beams_all = pad_matrix(b_val, T_recov)
    doses_all = pad_matrix(d_val, T_recov)
    doses_parms = pad_matrix(d_parm.value, T_recov)
    alpha_pad = np.vstack([alpha, np.zeros((T_recov, K))])
    beta_pad = np.vstack([beta, np.zeros((T_recov, K))])

    # Extend optimal health status into recovery stage using linear-quadratic model.
    health_recov = health_prog_act_range(h.value[-1], T_treat - 1, T_treat + T_recov, gamma = gamma,
                                         is_target = patient_rx["is_target"], health_map = health_map)
    health_opt_recov = np.row_stack([h.value, health_recov[1:]])

    # Compute health status from optimal doses using linearized/linear-quadratic models.
    health_proj = health_prog_act(h_init, T_treat + T_recov, alpha_pad, beta_pad, gamma, doses_all,
                                  patient_rx["is_target"], health_map)
    health_est = health_prog_est(h_init, T_treat + T_recov, alpha_pad, beta_pad, gamma, doses_all, doses_parms,
                                 patient_rx["is_target"], health_map)
    obj = dyn_quad_obj(d_val, health_proj[:(T_treat + 1)], patient_rx).value
    if use_ccp_slack:
        obj += ccp_slack_weight*np.sum(h_dyn_slack.value)
    return {"obj": obj, "status": status, "total_time": end - start, "solve_time": solve_time, "num_iters": k,
            "primal": np.array(r_prim[:k]), "dual": np.array(r_dual[:k]), "beams": beams_all, "doses": doses_all,
            "health": health_proj, "health_opt": health_opt_recov, "health_est": health_est}
