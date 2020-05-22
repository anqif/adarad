import cvxpy.settings as cvxpy_s
from collections import Counter

from fractionation.ccp_funcs import ccp_solve
from fractionation.quadratic.dyn_quad_prob import *
from fractionation.utilities.data_utils import pad_matrix, check_quad_vectors, health_prog_quad

def dyn_treat_quad(A_list, alpha, beta, gamma, h_init, patient_rx, T_recov = 0, health_map = lambda h,t: h, d_init = None, *args, **kwargs):
	T_treat = len(A_list)
	K, n = A_list[0].shape
	alpha, beta, gamma = check_quad_vectors(alpha, beta, gamma, K, T_treat, T_recov)
	
	# Build problem for treatment stage.
	prob, b, h, d, d_parm = build_dyn_quad_prob(A_list, alpha, beta, gamma, h_init, patient_rx, T_recov)
	result = ccp_solve(prob, d_parm, d_init, *args, **kwargs)
	if result["status"] not in ["optimal", "optimal_inaccurate"]:
		raise RuntimeError("CCP solve failed with status {0}".format(result["status"]))
	
	# Construct full results.
	beams_all = pad_matrix(b.value, T_recov)
	doses_all = pad_matrix(d.value, T_recov)
	alpha_pad = np.vstack([alpha, np.zeros((T_recov,K))])
	beta_pad  = np.vstack([alpha, np.zeros((T_recov,K))])
	health_all = health_prog_quad(h_init, T_treat + T_recov, alpha_pad, beta_pad, gamma, doses_all, health_map)
	obj = dyn_quad_obj(d.value, health_all[:(T_treat+1)], patient_rx).value
	return {"obj": obj, "status": prob.status, "solve_time": prob.solver_stats.solve_time, "beams": beams_all, "doses": doses_all, "health": health_all}