import numpy as np
from warnings import warn

# Pad matrix with zeros.
def pad_matrix(A, padding, axis = 0):
	m, n = A.shape
	if axis == 0:
		A_pad = np.zeros((m + padding,n))
		A_pad[:m,:] = A
	elif axis == 1:
		A_pad = np.zeros((m, n + padding))
		A_pad[:,:n] = A
	else:
		raise ValueError("axis must be either 0 or 1.")
	return A_pad

# Check dynamics matrices are correct dimension.
def check_dyn_matrices(F_list, G_list, q_list, r_list, K, T_treat, T_recov = 0):
	T_total = T_treat + T_recov
	if not isinstance(F_list, list):
		F_list = T_total*[F_list]
	if not isinstance(G_list, list):
		G_list = T_treat*[G_list]
	if not isinstance(q_list, list):
		q_list = T_treat*[q_list]
	if not isinstance(r_list, list):
		r_list = T_total*[r_list]
	
	if len(F_list) != T_total:
		raise ValueError("F_list must be a list of length {0}".format(T_total))
	if len(G_list) != T_treat:
		raise ValueError("G_list must be a list of length {0}".format(T_treat))
	if len(q_list) != T_treat:
		raise ValueError("q_list must be a list of length {0}".format(T_treat))
	if len(r_list) != T_total:
		raise ValueError("r_list must be a list of length {0}".format(T_total))
	
	for F in F_list:
		if F.shape != (K,K):
			raise ValueError("F_t must have dimensions ({0},{0})".format(K))
	for G in G_list:
		if G.shape != (K,K):
			raise ValueError("G_t must have dimensions ({0},{0})".format(K))
	for q in q_list:
		if q.shape not in [(K,), (K,1)]:
			raise ValueError("q_t must have dimensions ({0},)".format(K))
		if np.any(q < 0):
			raise ValueError("q_t can only contain nonnegative values")
	for r in r_list:
		# if r.shape != (K,) and r.shape != (K,1):
		if r.shape not in [(K,), (K,1)]:
			raise ValueError("r_t must have dimensions ({0},)".format(K))
	return F_list, G_list, q_list, r_list

def check_row_range(v, v_name, K, T):
	if len(v.shape) != 2:
		raise ValueError("{0} must be a 2-dimensional array".format(v_name))
	if v.shape[0] < T:
		raise ValueError("{0} must have at least {1} rows".format(v_name,T))
	if v.shape[1] != K:
		raise ValueError("{0} must have exactly {1} columns".format(v_name,K))

def check_quad_vectors(alpha, beta, gamma, K, T_treat, T_recov = 0, is_range = False):
	T_total = T_treat + T_recov
	if alpha is None:
		alpha = np.zeros((T_treat,K))
	if beta is None:
		beta = np.zeros((T_treat,K))
	if gamma is None:
		gamma = np.zeros((T_total,K))

	if is_range:
		check_row_range(alpha, "alpha", K, T_treat)
		check_row_range(beta, "beta", K, T_treat)
		check_row_range(gamma, "gamma", K, T_total)
	else:
		if alpha.shape != (T_treat,K):
			raise ValueError("alpha must have dimensions ({0},{1})".format(T_treat,K))
		if beta.shape != (T_treat,K):
			raise ValueError("beta must have dimensions ({0},{1})".format(T_treat,K))
		if gamma.shape != (T_total,K):
			raise ValueError("gamma must have dimensions ({0},{1})".format(T_total,K))

	if np.any(beta < 0):
		raise ValueError("beta can only contain nonnegative values")
	return alpha, beta, gamma

def check_prog_parms(alpha, beta, gamma, doses, K, T, is_range = False):
	# Defaults to no optimization.
	if doses is None:
		if alpha is None and beta is None:
			alpha = np.zeros((T, K))
			beta = np.zeros((T, K))
			doses = np.zeros((T, K))
		else:
			raise ValueError("doses must be provided.")
	else:
		if alpha is None and beta is not None:
			alpha = np.zeros((T, K))
		elif alpha is not None and beta is None:
			beta = np.zeros((T, K))
		elif alpha is None and beta is None:
			raise ValueError("alpha or beta must be provided.")
		if is_range:
			check_row_range(doses, "doses", K, T)
		elif doses.shape != (T, K):
			raise ValueError("doses must have dimensions ({0},{1})".format(T, K))
	if gamma is None:
		gamma = np.zeros((T, K))

	alpha, beta, gamma = check_quad_vectors(alpha, beta, gamma, K, T, T_recov=0, is_range=is_range)
	return alpha, beta, gamma, doses

def check_slack_parms(use_slack, slack_weight, default_slack=1):
	if use_slack is None and slack_weight is None:
		use_slack = False
		slack_weight = 0
	elif use_slack is not None and slack_weight is None:
		if not np.isscalar(use_slack):
			raise TypeError("use_slack must be a boolean")
		use_slack = bool(use_slack)
		slack_weight = default_slack if use_slack else 0
	elif use_slack is None and slack_weight is not None:
		if not (np.isscalar(slack_weight) and slack_weight >= 0):
			raise TypeError("slack_weight must be a non-negative scalar")
		use_slack = (slack_weight > 0)
	else:
		if not np.isscalar(use_slack):
			raise TypeError("use_slack must be a boolean")
		if not (np.isscalar(slack_weight) and slack_weight >= 0):
			raise TypeError("slack_weight must be a non-negative scalar")

		use_slack = bool(use_slack)
		if use_slack and slack_weight == 0:
			warn("use_slack is True, but slack_weight is 0")
		elif not use_slack and slack_weight > 0:
			warn("use_slack is False, but slack_weight is non-zero")
	return use_slack, slack_weight
