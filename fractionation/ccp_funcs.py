import numpy as np
from collections import Counter
from time import time

import cvxpy.settings as cvxpy_s
from cvxpy import Constant

from fractionation.utilities.data_utils import pad_matrix
from fractionation.mpc_funcs import build_dyn_prob, dyn_objective

def ccp_solve(prob, d, d_parm, d_init = None, h_slack = Constant(0), ccp_verbose = False, full_hist = False, *args, **kwargs):
	if d_init is None:
		d_init = np.zeros(d_parm.shape)

	# Problem parameters.
	max_iter = kwargs.pop("max_iter", 50)  # Maximum iterations.
	ccp_eps = kwargs.pop("ccp_eps", 1e-3)  # Stopping tolerance.
	iter_print = np.maximum(max_iter // 10, 1)

	# Validate parameters.
	if max_iter <= 0:
		raise ValueError("max_iter must be a positive integer.")
	if ccp_eps < 0:
		raise ValueError("ccp_eps must be a non-negative scalar.")
	
	k = 0
	solve_time = 0
	finished = False
	obj_cur = np.inf
	d_cur = d_init
	dose_list = []
	status_list = []
	h_slack_sum = np.zeros(max_iter)

	start = time()
	while not finished:
		if ccp_verbose and k % iter_print == 0:
			print("Iteration:", k)

		# Solve linearized problem.
		d_parm.value = d_cur
		prob.solve(*args, **kwargs)
		if prob.status not in cvxpy_s.SOLUTION_PRESENT:
			raise RuntimeError("Solver failed with status {0}".format(prob.status))
		solve_time += prob.solver_stats.solve_time
		status_list.append(prob.status)

		# Save entire history of doses.
		if full_hist:
			dose_list.append(d.value.copy())

		# Check stopping criterion.
		obj_diff = obj_cur - prob.value
		finished = (k + 1) >= max_iter or obj_diff <= ccp_eps

		# Update linearization point and objective.
		d_cur = d.value
		obj_cur = prob.value
		h_slack_sum[k] = np.sum(h_slack.value)
		k = k + 1
	end = time()

	# Take majority as final status.
	status, status_count = Counter(status_list).most_common(1)[0]
	doses = dose_list if full_hist else d.value
	return {"obj": obj_cur, "status": status, "num_iters": k, "total_time": start - end, "solve_time": solve_time,
			"doses": doses, "health_slacks": np.array(h_slack_sum[:k])}

def bed_health_prog(h_init, T, alphas, betas, doses = None, health_map = lambda h,d,t: h):
	K = h_init.shape[0]
	h_prog = np.zeros((T+1,K))
	h_prog[0] = h_init
	
	if not np.all(alphas > 0):
		raise ValueError("alphas must contain all positive values")
	if not np.all(betas > 0):
		raise ValueError("betas must contain all positive values")
	if not (alphas.shape[0] == betas.shape[0] and len(alphas.shape) <= 2 and len(betas.shape) <= 2):
		raise ValueError("alphas and betas must be vectors of the same length")
	
	# Defaults to no treatment.
	if doses is None:
		R_mat = np.zeros((K,K))
		doses = np.zeros((T,K))
	else:
		R_mat = np.diag(betas/alphas)
	
	for t in range(T):
		h_prog[t+1] = health_map(h_prog[t] - doses[t] - R_mat.dot(doses[t]**2), doses[t], t)
	return h_prog

def bed_lin(d, d_k, R_mat):
	g = d_k + R_mat.dot(d_k**2)
	g_prime = np.eye(d_k.shape[0]) + 2*R_mat.dot(d_k)
	return g + g_prime.T.dot(d - d_k)

def bed_lin_dyn_mats(d_k, R_mat, T_treat, T_recov = 0):
	K = d_k.shape[1]
	F_list = (T_treat + T_recov)*[np.eye(K)]
	q_list = (T_treat + T_recov)*[np.zeros(K)]
	
	G_list = []
	r_list = []
	for t in range(T_treat):
		G_list.append(-np.eye(K) - 2*np.diag(R_mat.dot(d_k[t])))
		r_list.append(R_mat.dot(d_k[t]**2))	
	r_list += T_recov*[np.zeros(K)]
	return F_list, G_list, q_list, r_list

def bed_ccp_dyn_treat(A_list, alphas, betas, h_init, patient_rx, T_recov = 0, health_map = lambda h,d,t: h, d_init = None, *args, **kwargs):
	T_treat = len(A_list)
	K = h_init.shape[0]
	
	if not np.all(alphas > 0):
		raise ValueError("alphas must contain all positive values")
	if not np.all(betas > 0):
		raise ValueError("betas must contain all positive values")
	if not (alphas.shape[0] == betas.shape[0] and len(alphas.shape) <= 2 and len(betas.shape) <= 2):
		raise ValueError("alphas and betas must be vectors of the same length")
	if d_init is None:
		d_init = np.zeros((T_treat,K))
	
	# Problem parameters.
	R_mat = np.diag(betas/alphas)
	max_iter = kwargs.pop("max_iter", 1000) # Maximum iterations.
	eps = kwargs.pop("eps", 1e-6)   # Stopping tolerance.
	
	k = 0
	solve_time = 0
	finished = False
	d_cur = d_init
	obj_cur = np.inf
	
	while not finished:
		# Formulate and solve problem.
		F_list, G_list, q_list, r_list = bed_lin_dyn_mats(d_cur, R_mat, T_treat, T_recov)
		prob, b, h, d, d_parm = build_dyn_prob(A_list, F_list, G_list, q_list, r_list, h_init, patient_rx, T_recov)
		prob.solve(*args, **kwargs)
		if prob.status not in ["optimal", "optimal_inaccurate"]:
			raise RuntimeError("Solver failed with status {0}".format(prob.status))
		solve_time += prob.solver_stats.solve_time
		
		# Update objective and doses.
		obj_diff = obj_cur - prob.value
		obj_cur = prob.value
		d_cur = d.value
	
		# Check stopping criterion.
		finished = (k + 1) >= max_iter or obj_diff <= eps
		k = k + 1
	
	# Construct full results.
	beams_all = pad_matrix(b.value, T_recov)
	doses_all = pad_matrix(d.value, T_recov)
	health_all = bed_health_prog(h_init, T_treat + T_recov, alphas, betas, doses_all, health_map)
	obj = dyn_objective(d.value, health_all[:(T_treat+1)], patient_rx).value
	return {"obj": obj, "status": prob.status, "num_iters": k, "solve_time": solve_time, "beams": beams_all, "doses": doses_all, "health": health_all}
