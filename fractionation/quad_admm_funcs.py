import numpy as np
import numpy.linalg as LA
import cvxpy.settings as cvxpy_s

from time import time
from multiprocessing import Process, Pipe
from collections import Counter

from fractionation.ccp_funcs import ccp_solve
from fractionation.mpc_funcs import print_results
from fractionation.problem.dyn_prob import rx_slice

from fractionation.quad_admm_slack_funcs import dyn_quad_treat_admm_slack
from fractionation.quadratic.dyn_quad_prob import dyn_quad_obj
from fractionation.quadratic.dyn_quad_prob_admm import *
from fractionation.utilities.data_utils import pad_matrix, check_quad_vectors, health_prog_quad

def run_quad_dose_worker(pipe, A, patient_rx, rho, *args, **kwargs):
    # Construct proximal dose problem.
    prob_dose, b, d = build_dyn_quad_prob_dose_period(A, patient_rx)
    d_new = Parameter(d.shape, value = np.zeros(d.shape))
    u = Parameter(d.shape, value = np.zeros(d.shape))
    penalty = (rho/2)*sum_squares(d - d_new - u)
    prox = prob_dose + Problem(Minimize(penalty))

    # ADMM loop.
    finished = False
    while not finished:
        # Compute and send d_t^k.
        try:
            prox.solve(*args, **kwargs)
        except SolverError:
            pipe.send((d.value, prox.solver_stats.solve_time, "SolverError"))
            break
        # if prox.status not in cvxpy_s.SOLUTION_PRESENT:
        #	raise RuntimeError("Solver failed with status {0}".format(prox.status))
        pipe.send((d.value, prox.solver_stats.solve_time, prox.status))

        # Receive \tilde d_t^k.
        d_new.value = pipe.recv()

        # Update and send u_t^^k.
        u.value += d_new.value - d.value
        pipe.send(u.value)

        # Check if stopped.
        finished = pipe.recv()

    # Send final b_t^k and d_t^k.
    d_val = A.dot(b.value)
    pipe.send((b.value, d_val))

def dyn_quad_treat_admm(A_list, alpha, beta, gamma, h_init, patient_rx, T_recov = 0, health_map = lambda h,t: h, d_init = None, partial_results = False, admm_verbose = False, *args, **kwargs):
    T_treat = len(A_list)
    K, n = A_list[0].shape
    alpha, beta, gamma = check_quad_vectors(alpha, beta, gamma, K, T_treat, T_recov)

    # Problem parameters.
    max_iter = kwargs.pop("max_iter", 1000)  # Maximum iterations.
    rho = kwargs.pop("rho", 1 / 10)  # Step size.
    eps_abs = kwargs.pop("eps_abs", 1e-6)  # Absolute stopping tolerance.
    eps_rel = kwargs.pop("eps_rel", 1e-3)  # Relative stopping tolerance.

    # Validate parameters.
    if max_iter <= 0:
        raise ValueError("max_iter must be a positive integer.")
    if rho <= 0:
        raise ValueError("rho must be a positive scalar.")
    if eps_abs < 0:
        raise ValueError("eps_abs must be a non-negative scalar.")
    if eps_rel < 0:
        raise ValueError("eps_rel must be a non-negative scalar.")

    # Set up dose workers.
    pipes = []
    procs = []
    for t in range(T_treat):
        rx_cur = rx_slice(patient_rx, t, t + 1)  # Get prescription at time t.
        local, remote = Pipe()
        pipes += [local]
        procs += [Process(target=run_quad_dose_worker, args=(remote, A_list[t], rx_cur, rho) + args, kwargs=kwargs)]
        procs[-1].start()

    # Proximal health problem.
    prob_health, h, d_tld, d_parm = build_dyn_quad_prob_health(alpha, beta, gamma, h_init, patient_rx, T_treat, T_recov)
    d_new = Parameter(d_tld.shape, value=np.zeros(d_tld.shape))
    u = Parameter(d_tld.shape, value=np.zeros(d_tld.shape))
    penalty = (rho / 2) * sum_squares(d_tld - d_new + u)
    prox = prob_health + Problem(Minimize(penalty))

    # ADMM loop.
    k = 0
    finished = False
    r_prim = np.zeros(max_iter)
    r_dual = np.zeros(max_iter)
    status_list = []

    start = time()
    solve_time = 0
    has_failed = False
    while not finished:
        if admm_verbose and k % 10 == 0:
            print("Iteration:", k)

        # Collect and stack d_t^k for t = 1,...,T.
        dt_update = [pipe.recv() for pipe in pipes]
        d_rows, d_times, d_statuses = map(list, zip(*dt_update))

        # Stop if any process failed to produce a d_t^k for t = 1,...,T.
        d_failed = np.setdiff1d(d_statuses, cvxpy_s.SOLUTION_PRESENT)
        if len(d_failed) > 0:
            status, status_count = Counter(d_failed).most_common(1)[0]
            has_failed = True
            break
        d_new.value = np.row_stack(d_rows)
        solve_time += np.max(d_times)
        d_status, d_status_count = Counter(d_statuses).most_common(1)[0]
        status_list.append(d_status)

        # Compute and send \tilde d_t^k.
        if k == 0:
            d_tld_prev = np.zeros((T_treat, K))
        else:
            d_init = d_tld_prev = d_tld.value
        try:
            result = ccp_solve(prox, d_tld, d_parm, d_init, *args, **kwargs)
        except SolverError:
            status = "SolverError"
            has_failed = True
            break
        if result["status"] not in cvxpy_s.SOLUTION_PRESENT:
            # raise RuntimeError("CCP solve failed with status {0}".format(result["status"]))
            status = result["status"]
            has_failed = True
            break
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
        finished = (k + 1) >= max_iter or (r_prim[k] <= eps_prim and r_dual[k] <= eps_dual)
        k = k + 1
        for pipe in pipes:
            pipe.send(finished)

    if has_failed:
        [proc.terminate() for proc in procs]
        return {"status": status, "num_iters": k}

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
        # h_val = health_prog_quad(h_init, T_treat, alpha, beta, gamma, d_val, health_map)
        # obj = dyn_quad_obj(d_val, h_val, patient_rx).value
        return {"status": status, "num_iters": k, "solve_time": solve_time, "beams": b_val, "doses": d_val}

    # Construct full results.
    beams_all = pad_matrix(b_val, T_recov)
    doses_all = pad_matrix(d_val, T_recov)
    alpha_pad = np.vstack([alpha, np.zeros((T_recov, K))])
    beta_pad = np.vstack([alpha, np.zeros((T_recov, K))])
    health_all = health_prog_quad(h_init, T_treat + T_recov, alpha_pad, beta_pad, gamma, doses_all, health_map)
    obj = dyn_quad_obj(d_val, health_all[:(T_treat + 1)], patient_rx).value
    return {"obj": obj, "status": status, "num_iters": k, "total_time": end - start, "solve_time": solve_time,
            "beams": beams_all, "doses": doses_all, "health": health_all, "primal": np.array(r_prim[:k]), "dual": np.array(r_dual[:k])}

def mpc_quad_treat_admm(A_list, alpha, beta, gamma, h_init, patient_rx, T_recov = 0, health_map = lambda h,t: h, d_init = None, \
					    use_slack = True, slack_weights = None, slack_final = True, mpc_verbose = False, *args, **kwargs):
    T_treat = len(A_list)
    K, n = A_list[0].shape
    alpha, beta, gamma = check_quad_vectors(alpha, beta, gamma, K, T_treat, T_recov)

    # Initialize values.
    beams = np.zeros((T_treat, n))
    doses = np.zeros((T_treat, K))
    num_iters = 0
    solve_time = 0
    status_list = []

    h_cur = h_init
    for t_s in range(T_treat):
        # Drop prescription for previous periods.
        rx_cur = rx_slice(patient_rx, t_s, T_treat, squeeze=False)

        # Solve optimal control problem from current period forward.
        # TODO: Warm start next ADMM solve, or better yet, rewrite so no teardown/rebuild process between ADMM solves.
        T_left = T_treat - t_s
        result = dyn_quad_treat_admm(T_left * [A_list[t_s]], np.row_stack(T_left*[alpha[t_s]]), np.row_stack(T_left*[beta[t_s]]), \
                    np.row_stack(T_left * [gamma[t_s]]), h_cur, rx_cur, T_recov, slack_weights = slack_weights, slack_final = slack_final, \
                    partial_results = True, *args, **kwargs)

        # If not optimal, re-solve with slack constraints.
        if result["status"] not in cvxpy_s.SOLUTION_PRESENT:
            if not use_slack:
                raise RuntimeError("Solver failed with status {0}".format(result["status"]))
            # warnings.warn("\nSolver failed with status {0}. Retrying with slack enabled...".format(result["status"]), RuntimeWarning)
            print("\nSolver failed with status {0}. Retrying with slack enabled...".format(result["status"]))
            result = dyn_quad_treat_admm_slack(T_left*[A_list[t_s]], np.row_stack(T_left*[alpha[t_s]]), np.row_stack(T_left*[beta[t_s]]), \
                        np.row_stack(T_left * [gamma[t_s]]), h_cur, rx_cur, T_recov, slack_weights = slack_weights, slack_final = slack_final, \
                        partial_results = True, *args, **kwargs)

        if mpc_verbose:
            print("\nStart Time:", t_s)
            print("Status:", result["status"])
            # print("Objective:", result["obj"])
            print("Solve Time:", result["solve_time"])
            print("Iterations:", result["num_iters"])

        # Save solver statistics.
        num_iters += result["num_iters"]
        solve_time += result["solve_time"]
        status_list.append(result["status"])

        # Save beams and doses for current period.
        beams[t_s] = result["beams"][0]
        # doses[t_s] = result["doses"][0]
        doses[t_s] = A_list[t_s].dot(beams[t_s])

        # Update health for next period.
        h_cur = health_map(h_cur - alpha[t_s] * doses[t_s] - beta[t_s] * doses[t_s] ** 2 + gamma[t_s], t_s)

    # Construct full results.
    beams_all = pad_matrix(beams, T_recov)
    doses_all = pad_matrix(doses, T_recov)
    alpha_pad = np.vstack([alpha, np.zeros((T_recov, K))])
    beta_pad = np.vstack([alpha, np.zeros((T_recov, K))])
    health_all = health_prog_quad(h_init, T_treat + T_recov, alpha_pad, beta_pad, gamma, doses_all, health_map)
    obj_treat = dyn_quad_obj(doses, health_all[:(T_treat + 1)], patient_rx).value
    # TODO: How should we handle constraint violations?
    status, status_count = Counter(status_list).most_common(1)[0]  # Take majority as final status.
    return {"obj": obj_treat, "status": status, "num_iters": num_iters, "solve_time": solve_time,
            "beams": beams_all, "doses": doses_all, "health": health_all}