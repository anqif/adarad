import cvxpy.settings as cvxpy_s
from cvxpy import SolverError
from collections import Counter

from adarad.medicine.prognosis import health_prog_act, health_prog_act_range, health_prog_est
from adarad.optimization.ccp_funcs import ccp_solve
from adarad.optimization.dose_init.dose_init import dyn_init_dose
from adarad.optimization.constraint import rx_slice

from adarad.optimization.seq_cvx.dyn_prob import build_dyn_quad_prob, dyn_quad_obj
from adarad.optimization.seq_cvx.slack_prob import build_dyn_slack_quad_prob
from adarad.utilities.data_utils import *

def print_quad_results(result, is_target, slack_dict=None):
	if slack_dict is None:
		slack_dict = dict()
	print("Status:", result["status"])
	print("Objective:", result["obj"])
	print("Solve Time:", result["solve_time"])
	print("Iterations:", result["num_iters"])
	if len(slack_dict.items()) > 0:
		def func_ss(slack):
			slack_lo = slack["lower"][:,~is_target].value if "lower" in slack else 0
			slack_hi = slack["upper"][:,is_target].value if "upper" in slack else 0
			return [np.sum(slack_lo**2), np.sum(slack_hi**2)]
			
		print("Sum-of-Squares of Slacks:")
		for key, value in slack_dict.items():
			print("\t{0} (Lower, Upper):".format(key.title()), func_ss(value))

def dyn_quad_treat(A_list, alpha, beta, gamma, h_init, patient_rx, T_recov = 0, health_map = lambda h,d,t: h, d_init = None,
					auto_init = False, use_slack = False, slack_weight = 0, full_hist = False, *args, **kwargs):
	T_treat = len(A_list)
	K, n = A_list[0].shape
	alpha, beta, gamma = check_quad_vectors(alpha, beta, gamma, K, T_treat, T_recov)

	# Problem parameters.
	max_iter = kwargs.pop("max_iter", 50)  # Maximum iterations.
	ccp_eps = kwargs.pop("ccp_eps", 1e-3)  # Stopping tolerance.
	ccp_verbose = kwargs.pop("ccp_verbose", False)

	# Initialize dose.
	total_time = 0
	solve_time = 0
	if d_init is None:
		if auto_init:
			if ccp_verbose:
				print("Calculating initial dose...")
			result_init = dyn_init_dose(A_list, alpha, beta, gamma, h_init, patient_rx, T_recov, use_slack, slack_weight, *args, **kwargs)
			d_init = result_init["doses"]
			total_time += result_init["total_time"]
			solve_time += result_init["solve_time"]
		else:
			d_init = np.zeros((T_treat, K))
	if ccp_verbose:
		print("Initial dose per fraction: {0}".format(d_init[0]))
	
	# Build seq_cvx for optimization stage.
	prob, b, h, d, d_parm, h_dyn_slack = build_dyn_quad_prob(A_list, alpha, beta, gamma, h_init, patient_rx, T_recov,
															 use_slack, slack_weight)
	result = ccp_solve(prob, d, d_parm, d_init, h_dyn_slack, ccp_verbose, full_hist, max_iter = max_iter, ccp_eps = ccp_eps, 
					   *args, **kwargs)
	if result["status"] not in cvxpy_s.SOLUTION_PRESENT:
		raise RuntimeError("CCP solve failed with status {0}".format(result["status"]))
	total_time += result["total_time"]
	solve_time += result["solve_time"]

	# Construct full results.
	beams_all = pad_matrix(b.value, T_recov)
	doses_all = pad_matrix(d.value, T_recov)
	doses_parms = pad_matrix(d_parm.value, T_recov)
	alpha_pad = np.vstack([alpha, np.zeros((T_recov,K))])
	beta_pad  = np.vstack([beta, np.zeros((T_recov,K))])

	# Extend optimal health status into recovery stage using linear-quadratic model.
	health_recov = health_prog_act_range(h.value[-1], T_treat - 1, T_treat + T_recov, gamma = gamma,
										 is_target = patient_rx["is_target"], health_map = health_map)
	health_opt_recov = np.row_stack([h.value, health_recov[1:]])

	# Compute health status from optimal doses using linearized/linear-quadratic models.
	health_proj = health_prog_act(h_init, T_treat + T_recov, alpha_pad, beta_pad, gamma, doses_all, patient_rx["is_target"], health_map)
	health_est = health_prog_est(h_init, T_treat + T_recov, alpha_pad, beta_pad, gamma, doses_all, doses_parms, patient_rx["is_target"], health_map)
	
	# Compute health status in each CCP iteration.
	doses_hist = []
	health_hist = []
	if full_hist:
		for d_cur in result["doses"]:
			d_cur_all = pad_matrix(d_cur, T_recov)
			h_cur_proj = health_prog_act(h_init, T_treat + T_recov, alpha_pad, beta_pad, gamma, d_cur_all, patient_rx["is_target"], health_map)
			doses_hist.append(d_cur_all)
			health_hist.append(h_cur_proj)

	# Compute objective with final dose.
	obj = dyn_quad_obj(d, health_proj[:(T_treat+1)], patient_rx).value
	if ccp_verbose:
		print("Final objective without slack: {0}".format(obj))
	if use_slack:
		obj += slack_weight*np.sum(h_dyn_slack.value)
	run_parms = {"use_slack": use_slack, "slack_weight": slack_weight}
	return {"obj": obj, "status": result["status"], "total_time": total_time, "solve_time": solve_time, "num_iters": result["num_iters"],
			"beams": beams_all, "doses": doses_all, "health": health_proj, "health_opt": health_opt_recov, "health_est": health_est,
			"health_slacks": result["health_slacks"], "doses_hist": doses_hist, "health_hist": health_hist, "parms": run_parms}

def mpc_quad_treat(A_list, alpha, beta, gamma, h_init, patient_rx, T_recov = 0, health_map = lambda h,d,t: h, d_init = None,
				   auto_init = False, use_ccp_slack = False, ccp_slack_weight = 0, use_mpc_slack = False, mpc_slack_weights = 1,
				   mpc_verbose = False, full_hist = False, *args, **kwargs):
	T_treat = len(A_list)
	K, n = A_list[0].shape
	alpha, beta, gamma = check_quad_vectors(alpha, beta, gamma, K, T_treat, T_recov)

	# Problem parameters.
	max_iter = kwargs.pop("max_iter", 50)  # Maximum iterations.
	ccp_eps = kwargs.pop("ccp_eps", 1e-3)  # Stopping tolerance.
	ccp_verbose = kwargs.pop("ccp_verbose", False)

	# Initialize dose.
	solve_time = 0
	if d_init is None:
		if auto_init:
			if mpc_verbose:
				print("Calculating initial dose...")
			result_init = dyn_init_dose(A_list, alpha, beta, gamma, h_init, patient_rx, T_recov, use_ccp_slack,
										ccp_slack_weight, *args, **kwargs)
			d_init = result_init["doses"]
			solve_time += result_init["solve_time"]
		else:
			d_init = np.zeros((T_treat, K))
	if mpc_verbose:
		print("Initial dose per fraction: {0}".format(d_init[0]))

	# Initialize values.
	beams = np.zeros((T_treat,n))
	doses = np.zeros((T_treat,K))
	parms = np.zeros((T_treat,K))
	dyn_slacks = np.zeros((T_treat,K))
	health_opt = np.zeros((T_treat + 1,K))
	health_opt[0] = h_init

	num_iters = 0
	dose_list = []
	status_list = []
	h_cur = h_init
	for t_s in range(T_treat):
		# Drop prescription for previous periods.
		rx_cur = rx_slice(patient_rx, t_s, T_treat, squeeze = False)

		# Solve optimal control seq_cvx from current period forward.
		T_left = T_treat - t_s
		if use_mpc_slack:
			prob, b, h, d, d_parm, h_dyn_slack = build_dyn_slack_quad_prob(T_left * [A_list[t_s]], np.row_stack(T_left * [alpha[t_s]]), np.row_stack(T_left * [beta[t_s]]),
													   np.row_stack(T_left*[gamma[t_s]]), h_cur, rx_cur, T_recov, use_ccp_slack, ccp_slack_weight, mpc_slack_weights)
		else:
			prob, b, h, d, d_parm, h_dyn_slack = build_dyn_quad_prob(T_left*[A_list[t_s]], np.row_stack(T_left*[alpha[t_s]]), np.row_stack(T_left*[beta[t_s]]),
													   np.row_stack(T_left*[gamma[t_s]]), h_cur, rx_cur, T_recov, use_ccp_slack, ccp_slack_weight)
			# prob, b, h, d, d_parm, h_dyn_slack = build_dyn_quad_prob(A_list[t_s:], alpha[t_s:], beta[t_s:], gamma[t_s:],
			# 											h_cur, rx_cur, T_recov, use_ccp_slack, ccp_slack_weight)
		try:
			result = ccp_solve(prob, d, d_parm, d_init, h_dyn_slack, ccp_verbose, max_iter = max_iter, ccp_eps = ccp_eps,
							   *args, **kwargs)
			status = result["status"]
		except SolverError:
			status = "SolverError"
		if status not in cvxpy_s.SOLUTION_PRESENT:
			raise RuntimeError("Solver failed with status {0}".format(status))

		# If not optimal, re-solve with slack constraints.
		# if status not in cvxpy_s.SOLUTION_PRESENT:
		# 	if not use_mpc_slack:
		# 		raise RuntimeError("Solver failed with status {0}".format(status))
		# 	# warnings.warn("\nSolver failed with status {0}. Retrying with slack enabled...".format(status), RuntimeWarning)
		# 	print("\nSolver failed with status {0}. Retrying with slack enabled...".format(status))
		#
		# 	prob, b, h, d, d_parm, h_dyn_slack = build_dyn_slack_quad_prob(T_left*[A_list[t_s]], np.row_stack(T_left*[alpha[t_s]]), np.row_stack(T_left*[beta[t_s]]),
		# 															np.row_stack(T_left*[gamma[t_s]]), h_cur, rx_cur, T_recov, use_ccp_slack, ccp_slack_weight,
		# 															mpc_slack_weights)
		# 	result = ccp_solve(prob, d, d_parm, d_init, h_dyn_slack, *args, **kwargs)
		# 	status = result["status"]
		# 	if status not in cvxpy_s.SOLUTION_PRESENT:
		# 		raise RuntimeError("Solver failed on slack seq_cvx with status {0}".format(status))
		
		if mpc_verbose:
			print("\nStart Time:", t_s)
			print_quad_results(result, patient_rx["is_target"])

		# Save solver statistics.
		num_iters += result["num_iters"]
		solve_time += result["solve_time"]
		status_list.append(status)

		# Save entire history of doses.
		if full_hist:
			dose_list.append(d.value.copy())

		# Save beams, doses, and health statuses for current period.
		beams[t_s] = b.value[0]
		doses[t_s] = d.value[0]
		parms[t_s] = d_parm.value[0]
		dyn_slacks[t_s] = h_dyn_slack.value[0]
		health_opt[t_s + 1] = h.value[1]

		# Update health for next period.
		d_init = d.value[1:]   # Initialize next CCP at optimal dose from current period.
		h_cur = health_prog_act_range(h_cur, t_s, t_s + 1, alpha, beta, gamma, doses, patient_rx["is_target"], health_map)[1]

	# Construct full results.
	beams_all = pad_matrix(beams, T_recov)
	doses_all = pad_matrix(doses, T_recov)
	parms_all = pad_matrix(parms, T_recov)
	alpha_pad = np.vstack([alpha, np.zeros((T_recov, K))])
	beta_pad = np.vstack([beta, np.zeros((T_recov, K))])

	# Extend optimal health status into recovery stage using linear-quadratic model.
	health_recov = health_prog_act_range(health_opt[-1], T_treat - 1, T_treat + T_recov, gamma = gamma,
										 is_target = patient_rx["is_target"], health_map = health_map)
	health_opt_recov = np.row_stack([health_opt, health_recov[1:]])

	# Compute health status from optimal doses using linearized/linear-quadratic models.
	health_est = health_prog_est(h_init, T_treat + T_recov, alpha_pad, beta_pad, gamma, doses_all, parms_all,
								 patient_rx["is_target"], health_map)
	health_proj = health_prog_act(h_init, T_treat + T_recov, alpha_pad, beta_pad, gamma, doses_all, patient_rx["is_target"], health_map)

	# Compute health status in each CCP iteration.
	doses_hist = []
	health_hist = []
	if full_hist:
		for t_s in range(T_treat):
			d_cur = np.row_stack([doses[:t_s], dose_list[t_s]])   # Prepend doses delivered in sessions \tau = 1,...,t-1.
			d_cur_all = pad_matrix(d_cur, T_recov)
			h_cur_proj = health_prog_act(h_init, T_treat + T_recov, alpha_pad, beta_pad, gamma, d_cur_all, patient_rx["is_target"], health_map)
			doses_hist.append(d_cur_all)
			health_hist.append(h_cur_proj)

	# Compute objective with final dose.
	obj = dyn_quad_obj(doses, health_proj[:(T_treat + 1)], patient_rx).value
	if use_ccp_slack:
		obj += ccp_slack_weight*np.sum(dyn_slacks)
	# TODO: How should we handle constraint violations?
	status, status_count = Counter(status_list).most_common(1)[0]   # Take majority as final status.
	return {"obj": obj, "status": status, "num_iters": num_iters, "solve_time": solve_time, "beams": beams_all,
			"doses": doses_all, "health": health_proj, "health_opt": health_opt_recov, "health_est": health_est,
			"doses_hist": doses_hist, "health_hist": health_hist}
