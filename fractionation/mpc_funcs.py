from collections import Counter

import cvxpy.settings as cvxpy_s

from fractionation.utilities.data_utils import pad_matrix, check_dyn_matrices, health_prognosis
from fractionation.problem.dyn_prob import *
from fractionation.problem.slack_prob import build_dyn_slack_prob

def print_results(prob, slack_dict=dict()):
	print("Status:", prob.status)
	print("Objective:", prob.value)
	print("Solve Time:", prob.solver_stats.solve_time)
	if len(slack_dict.items()) > 0:
		func_ss = lambda v: [np.sum(vi.value**2) for vi in v]
		print("Sum-of-Squares of Slacks:")
		for key, value in slack_dict.items():
			print("\t{0} (Lower, Upper):".format(key.title()), func_ss(value))

def single_treatment(A, patient_rx, *args, **kwargs):
	K, n = A.shape
	if patient_rx["dose_goal"].shape not in [(K,), (K,1)]:
		raise ValueError("dose_goal must have dimensions ({0},)".format(K))
	
	b = Variable(n, pos = True)   # Beams.
	d = Variable(K, pos = True)   # Doses.
	
	obj = dose_penalty(d, patient_rx["dose_goal"], patient_rx["dose_weights"])
	# constrs = [d == A*b, b >= 0]
	constrs = [d == A*b]
	
	if "beam_constrs" in patient_rx:
		constrs += rx_to_constrs(b, patient_rx["beam_constrs"])
	
	if "dose_constrs" in patient_rx:
		constrs += rx_to_constrs(d, patient_rx["dose_constrs"])
	
	prob = Problem(Minimize(obj), constrs)
	prob.solve(*args, **kwargs)
	# h = F.dot(h_init) + G.dot(d.value)
	return {"obj": prob.value, "status": prob.status, "beams": b.value, "doses": d.value}

def dynamic_treatment(A_list, F_list, G_list, r_list, h_init, patient_rx, T_recov = 0, health_map = lambda h,t: h, *args, **kwargs):
	T_treat = len(A_list)
	K, n = A_list[0].shape
	F_list, G_list, r_list = check_dyn_matrices(F_list, G_list, r_list, K, T_treat, T_recov)
	
	# Build problem for treatment stage.
	prob, b, h, d = build_dyn_prob(A_list, F_list, G_list, r_list, h_init, patient_rx, T_recov)
	prob.solve(*args, **kwargs)
	if prob.status not in ["optimal", "optimal_inaccurate"]:
		raise RuntimeError("Solver failed with status {0}".format(prob.status))
	
	# Construct full results.
	beams_all = pad_matrix(b.value, T_recov)
	doses_all = pad_matrix(d.value, T_recov)
	G_list_pad = G_list + T_recov*[np.zeros(G_list[0].shape)]
	health_all = health_prognosis(h_init, T_treat + T_recov, F_list, G_list_pad, r_list, doses_all, health_map)
	obj = dyn_objective(d.value, health_all[:(T_treat+1)], patient_rx).value
	return {"obj": obj, "status": prob.status, "solve_time": prob.solver_stats.solve_time, "beams": beams_all, "doses": doses_all, "health": health_all}

def mpc_treatment(A_list, F_list, G_list, r_list, h_init, patient_rx, T_recov = 0, health_map = lambda h,t: h, use_slack = True, mpc_verbose = False, *args, **kwargs):
	T_treat = len(A_list)
	K, n = A_list[0].shape
	F_list, G_list, r_list = check_dyn_matrices(F_list, G_list, r_list, K, T_treat, T_recov)
	
	# Initialize values.
	beams = np.zeros((T_treat,n))
	doses = np.zeros((T_treat,K))
	solve_time = 0
	status_list = []
	
	h_cur = h_init
	# warnings.simplefilter("always", RuntimeWarning)
	for t_s in range(T_treat):
		# Drop prescription for previous periods.
		rx_cur = rx_slice(patient_rx, t_s, T_treat, squeeze = False)
		
		# Solve optimal control problem from current period forward.
		T_left = T_treat - t_s
		prob, b, h, d = build_dyn_prob(T_left*[A_list[t_s]], T_left*[F_list[t_s]], T_left*[G_list[t_s]], T_left*[r_list[t_s]], h_cur, rx_cur, T_recov)
		# prob, b, h, d = build_dyn_prob(A_list[t_s:], F_list[t_s:], G_list[t_s:], r_list[t_s:], h_cur, rx_cur, T_recov)
		prob.solve(*args, **kwargs)

		# If not optimal, re-solve with slack constraints.
		if prob.status not in cvxpy_s.SOLUTION_PRESENT:
			if not use_slack:
				raise RuntimeError("Solver failed with status {0}".format(prob.status))
			# warnings.warn("\nSolver failed with status {0}. Retrying with slack enabled...".format(prob.status), RuntimeWarning)
			print("\nSolver failed with status {0}. Retrying with slack enabled...".format(prob.status))

			prob, b, h, d, s_vars = build_dyn_slack_prob(T_left*[A_list[t_s]], T_left*[F_list[t_s]], T_left*[G_list[t_s]], T_left*[r_list[t_s]], h_cur, rx_cur, T_recov)
			prob.solve(*args, **kwargs)
			if prob.status not in cvxpy_s.SOLUTION_PRESENT:
				raise RuntimeError("Solver failed on slack problem with status {0}".format(prob.status))
		else:
			s_vars = dict()
		solve_time += prob.solver_stats.solve_time
		
		if mpc_verbose:
			print("\nStart Time:", t_s)
			print_results(prob, slack_dict=s_vars)
		
		# Save beams, doses, and penalties for current period.
		status_list.append(prob.status)
		beams[t_s] = b.value[0]
		doses[t_s] = d.value[0]
		
		# Update health for next period.
		h_cur = health_map(h.value[1], t_s)
		# h_cur = health_map(F_list[t_s].dot(h_cur) + G_list[t_s].dot(doses[t_s]) + r_list[t_s], t_s)
	
	# Construct full results.
	beams_all = pad_matrix(beams, T_recov)
	doses_all = pad_matrix(doses, T_recov)
	G_list_pad = G_list + T_recov*[np.zeros(G_list[0].shape)]
	health_all = health_prognosis(h_init, T_treat + T_recov, F_list, G_list_pad, r_list, doses_all, health_map)
	obj_treat = dyn_objective(doses, health_all[:(T_treat+1)], patient_rx).value
	status, status_count = Counter(status_list).most_common(1)[0]   # Take majority as final status.
	return {"obj": obj_treat, "status": status, "solve_time": solve_time, "beams": beams_all, "doses": doses_all, "health": health_all}
