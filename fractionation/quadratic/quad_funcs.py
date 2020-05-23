import numpy as np
import cvxpy.settings as cvxpy_s
from cvxpy import SolverError
from collections import Counter

from fractionation.ccp_funcs import ccp_solve
from fractionation.mpc_funcs import print_results
from fractionation.problem.dyn_prob import rx_slice

from fractionation.quadratic.dyn_quad_prob import build_dyn_quad_prob, dyn_quad_obj
# from fractionation.quadratic.slack_quad_prob import build_dyn_slack_quad_prob
from fractionation.utilities.data_utils import pad_matrix, check_quad_vectors, health_prog_quad

def dyn_quad_treat(A_list, alpha, beta, gamma, h_init, patient_rx, T_recov = 0, health_map = lambda h,t: h, d_init = None, *args, **kwargs):
	T_treat = len(A_list)
	K, n = A_list[0].shape
	alpha, beta, gamma = check_quad_vectors(alpha, beta, gamma, K, T_treat, T_recov)
	
	# Build problem for treatment stage.
	prob, b, h, d, d_parm = build_dyn_quad_prob(A_list, alpha, beta, gamma, h_init, patient_rx, T_recov)
	result = ccp_solve(prob, d, d_parm, d_init, *args, **kwargs)
	if result["status"] not in cvxpy_s.SOLUTION_PRESENT:
		raise RuntimeError("CCP solve failed with status {0}".format(result["status"]))
	
	# Construct full results.
	beams_all = pad_matrix(b.value, T_recov)
	doses_all = pad_matrix(d.value, T_recov)
	alpha_pad = np.vstack([alpha, np.zeros((T_recov,K))])
	beta_pad  = np.vstack([alpha, np.zeros((T_recov,K))])
	health_all = health_prog_quad(h_init, T_treat + T_recov, alpha_pad, beta_pad, gamma, doses_all, health_map)
	obj = dyn_quad_obj(d.value, health_all[:(T_treat+1)], patient_rx).value
	return {"obj": obj, "status": result["status"], "solve_time": result["solve_time"], "num_iters": result["num_iters"], \
			"beams": beams_all, "doses": doses_all, "health": health_all}

def mpc_quad_treat(A_list, alpha, beta, gamma, h_init, patient_rx, T_recov = 0, health_map = lambda h,t: h, d_init = None, \
					use_slack = True, slack_weights = None, slack_final = True, mpc_verbose = False, *args, **kwargs):
	T_treat = len(A_list)
	K, n = A_list[0].shape
	alpha, beta, gamma = check_quad_vectors(alpha, beta, gamma, K, T_treat, T_recov)
	
	# Initialize values.
	beams = np.zeros((T_treat,n))
	doses = np.zeros((T_treat,K))
	solve_time = 0
	status_list = []
	
	h_cur = h_init
	for t_s in range(T_treat):
		# Drop prescription for previous periods.
		rx_cur = rx_slice(patient_rx, t_s, T_treat, squeeze = False)

		# Solve optimal control problem from current period forward.
		T_left = T_treat - t_s
		prob, b, h, d, d_parm = build_dyn_quad_prob(T_left*[A_list[t_s]], np.tile(alpha[t_s], T_left), np.tile(beta[t_s], T_left), np.tile(gamma[t_s], T_left), h_cur, rx_cur, T_recov)
		# prob, b, h, d, d_parm = build_dyn_quad_prob(A_list[t_s:], alpha[t_s:], beta[t_s:], gamma[t_s:], h_cur, rx_cur, T_recov)
		try:
			result = ccp_solve(prob, d, d_parm, d_init, *args, **kwargs)
			status = result["status"]
		except SolverError:
			status = "SolverError"

		# If not optimal, re-solve with slack constraints.
		if status not in cvxpy_s.SOLUTION_PRESENT:
			if not use_slack:
				raise RuntimeError("Solver failed with status {0}".format(status))
			# warnings.warn("\nSolver failed with status {0}. Retrying with slack enabled...".format(status), RuntimeWarning)
			print("\nSolver failed with status {0}. Retrying with slack enabled...".format(status))

			prob, b, h, d, d_parm, s_vars = build_dyn_quad_slack_prob(T_left*[A_list[t_s]], np.tile(alpha[t_s], T_left), np.tile(beta[t_s], T_left), np.tile(gamma[t_s], T_left), \
														 				h_cur, rx_cur, T_recov, slack_weights, slack_final)
			result = ccp_solve(prob, d, d_parm, d_init, *args, **kwargs)
			status = result["status"]
			if status not in cvxpy_s.SOLUTION_PRESENT:
				raise RuntimeError("Solver failed on slack problem with status {0}".format(status))
		else:
			s_vars = dict()
		
		if mpc_verbose:
			print("\nStart Time:", t_s)
			print_results(prob, status = status, slack_dict = s_vars)

		# Save solver statistics.
		solve_time += result["solve_time"]
		status_list.append(status)

		# Save beams and doses for current period.
		beams[t_s] = b.value[0]
		doses[t_s] = d.value[0]

		# Update health for next period.
		h_cur = health_map(h_cur - alpha[t_s]*doses[t_s] - beta[t_s]*doses[t_s]**2 + gamma[t_s], t_s)

	# Construct full results.
	beams_all = pad_matrix(beams, T_recov)
	doses_all = pad_matrix(doses, T_recov)
	alpha_pad = np.vstack([alpha, np.zeros((T_recov,K))])
	beta_pad  = np.vstack([alpha, np.zeros((T_recov,K))])
	health_all = health_prog_quad(h_init, T_treat + T_recov, alpha_pad, beta_pad, gamma, doses_all, health_map)
	obj_treat = dyn_quad_obj(doses, health_all[:(T_treat+1)], patient_rx).value
	# TODO: How should we handle constraint violations?
	status, status_count = Counter(status_list).most_common(1)[0]   # Take majority as final status.
	return {"obj": obj_treat, "status": status, "solve_time": solve_time, "beams": beams_all, "doses": doses_all, "health": health_all}