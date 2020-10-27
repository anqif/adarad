import numpy as np

# Center and rotate counterclockwise by an angle from the x-axis.
def coord_transf(x, y, center = (0,0), angle = 0):
	x0, y0 = center
	xr = (x - x0)*np.cos(angle) - (y - y0)*np.sin(angle)
	yr = (x - x0)*np.sin(angle) + (y - y0)*np.cos(angle)
	return xr, yr
	
# Equation of a circle.
def circle(x, y, center = (0,0), radius = 1):
	return ellipse(x, y, center, (radius, radius))

# Equation of an ellipse.
def ellipse(x, y, center = (0,0), width = (1,1), angle = 0):
	x_width, y_width = width
	xr, yr = coord_transf(x, y, center, angle)
	return (xr/x_width)**2 + (yr/y_width)**2 - 1

# Equation of a cardiod.
def cardioid(x, y, a = 0.5, center = (0,0), angle = 0):
	# xr, yr = coord_transf(x, y, center, angle)
	# return (xr**2 + yr**2)**2 + 4*a*xr*(xr**2 + yr**2) - 4*a**2*yr**2
	return limacon(x, y, -2*a, 2*a, center, angle)

# Equation of a limacon.
def limacon(x, y, a = 1, b = 0, center = (0,0), angle = 0):
	xr, yr = coord_transf(x, y, center, angle)
	return (xr**2 + yr**2 - a*xr)**2 - b**2*(xr**2 + yr**2)

# Generate xy-coordinate pairs from line angles and displacements.
def line_segments(angles, d_vec, n_grid, xlim = (-1,1), ylim = (-1,1)):
	if np.any(angles < 0) or np.any(angles > np.pi):
		raise ValueError("angles must all be in [0,pi]")
	
	n_angles = len(angles)
	n_offsets = len(d_vec)
	n_lines = n_angles*n_offsets
	segments = np.zeros((n_lines,2,2))   # n_lines x n_points x n_dims = 2.
	
	xc = (xlim[1] + xlim[0])/2
	yc = (ylim[1] + ylim[0])/2
	x_scale = (xlim[1] - xlim[0])/n_grid   # (x_max - x_min)/(x_len_pixels)
	y_scale = (ylim[1] - ylim[0])/n_grid   # (y_max - y_min)/(y_len_pixels)
	dydx_scale = (ylim[1] - ylim[0])/(xlim[1] - xlim[0])
	
	k = 0
	for i in range(n_angles):
		# Slope of line.
		slope = dydx_scale*np.tan(angles[i])
		
		# Center of line.
		x0 = xc - x_scale*d_vec*np.sin(angles[i])
		y0 = yc + y_scale*d_vec*np.cos(angles[i])
		
		# Endpoints of line.
		for j in range(n_offsets):
			if slope == 0:
				segments[k,0,:] = [xlim[0], y0[j]]
				segments[k,1,:] = [xlim[1], y0[j]]
			elif np.isinf(slope):
				segments[k,0,:] = [x0[j], ylim[0]]
				segments[k,1,:] = [x0[j], ylim[1]]
			else:
				# y - y0 = slope*(x - x0).
				ys = y0[j] + slope*(xlim - x0[j])
				xs = x0[j] + (1/slope)*(ylim - y0[j])
				
				# Save points at edge of grid.
				idx_y = (ys >= ylim[0]) & (ys <= ylim[1])
				idx_x = (xs >= xlim[0]) & (xs <= xlim[1])
				e1 = np.column_stack([xlim, ys])[idx_y]
				e2 = np.column_stack([xs, ylim])[idx_x]
				edges = np.row_stack([e1, e2])
				
				segments[k,0,:] = edges[0]
				segments[k,1,:] = edges[1]
			k = k + 1
	return segments

# Construct line integral matrix.
def line_integral_mat(structures, angles = 10, n_bundle = 1, offset = 10):
	m_grid, n_grid = structures.shape
	K = np.unique(structures).size
	
	if m_grid != n_grid:
		raise NotImplementedError("Only square grids are supported")
	if np.isscalar(angles):
		angles = np.linspace(0, np.pi, angles+1)[:-1]
	if n_bundle <= 0:
		raise ValueError("n_bundle must be a positive integer")
	if offset < 0:
		raise ValueError("offset must be a nonnegative number")
	
	# A_{kj} = fraction of beam j that falls in region k.
	n_angle = len(angles)
	n_bundle = int(n_bundle)
	n = n_angle*n_bundle
	A = np.zeros((K, n))
	
	# Orthogonal offsets of line from image center given in pixel lengths.
	# Positive direction = northwest for angle <= np.pi/2, southwest for angle > np.pi/2.
	n_half = n_bundle//2
	d_vec = np.arange(-n_half, n_half+1)
	if n_bundle % 2 == 0:
		d_vec = d_vec[:-1]
	d_vec = offset*d_vec
	
	j = 0
	for i in range(n_angle):
		for d in d_vec:
			# Flip angle since we measure counterclockwise from x-axis.
			L = line_pixel_length(d, np.pi-angles[i], n_grid)
			for k in range(K):
				A[k,j] = np.sum(L[structures == k])
			j = j + 1
	return A, angles, d_vec

def line_pixel_length(d, theta, n):
	"""
	Image reconstruction from line measurements.
	
	Given a grid of n by n square pixels and a line over that grid,
	compute the length of line that goes over each pixel.
	
	Parameters
	----------
	d : displacement of line, i.e., distance of line from center of image, 
		measured in pixel lengths (and orthogonally to line).
	theta : angle of line, measured in radians clockwise from x-axis. 
			Must be between 0 and pi, inclusive.
	n : image size is n by n.
	
	Returns
	-------
	Matrix of size n by n (same as image) with length of the line over 
	each pixel. Most entries will be zero.
	"""
	# For angle in [pi/4,3*pi/4], flip along diagonal (transpose) and call recursively.
	if theta > np.pi/4 and theta < 3*np.pi/4:
		return line_pixel_length(d, np.pi/2-theta, n).T
	
	# For angle in [3*pi/4,pi], redefine line to go in opposite direction.
	if theta > np.pi/2:
		d = -d
		theta = theta - np.pi
	
	# For angle in [-pi/4,0], flip along x-axis (up/down) and call recursively.
	if theta < 0:
		return np.flipud(line_pixel_length(-d, -theta, n))
	
	if theta > np.pi/2 or theta < 0:
		raise ValueError("theta must be in [0,pi]")
	
	L = np.zeros((n,n))
	ct = np.cos(theta)
	st = np.sin(theta)
	
	x0 = n/2 - d*st
	y0 = n/2 + d*ct
	
	y = y0 - x0*st/ct
	jy = int(np.ceil(y))
	dy = (y + n) % 1
	
	for jx in range(n):
		dynext = dy + st/ct
		if dynext < 1:
			if jy >= 1 and jy <= n:
				L[n-jy, jx] = 1/ct
			dy = dynext
		else:
			if jy >= 1 and jy <= n:
				L[n-jy, jx] = (1-dy)/st
			if jy+1 >= 1 and jy + 1 <= n:
				L[n-(jy+1), jx] = (dynext-1)/st
			dy = dynext - 1
			jy = jy + 1
	return L

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

# Block average rows of dose influence matrix.
def beam_to_dose_block(A_full, indices_or_sections):
	A_blocks = np.split(A_full, indices_or_sections)
	# A_rows = [np.sum(block, axis = 0) for block in A_blocks]
	A_rows = [np.mean(block, axis = 0) for block in A_blocks]
	A = np.row_stack(A_rows)
	return A

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
	# Defaults to no treatment.
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

# Health prognosis with a given treatment.
def health_prognosis(h_init, T, F_list, G_list = None, q_list = None, r_list = None, doses = None, health_map = lambda h,d,t: h):
	K = h_init.shape[0]

	# Defaults to no treatment.
	if doses is None:
		if G_list is None and q_list is None:
			G_list = T*[np.zeros((K,K))]
			q_list = T*[np.zeros(K)]
			doses = np.zeros((T,K))
		else:
			raise ValueError("doses must be provided.")
	else:
		if G_list is None and q_list is not None:
			G_list = T*[np.zeros((K,K))]
		elif G_list is not None and q_list is None:
			q_list = T*[np.zeros(K)]
		elif G_list is None and q_list is None:
			raise ValueError("G_list or q_list must be provided.")
		if doses.shape != (T,K):
			raise ValueError("doses must have dimensions ({0},{1})".format(T,K))
	if r_list is None:
		r_list = T*[np.zeros(K)]
	
	F_list, G_list, q_list, r_list = check_dyn_matrices(F_list, G_list, q_list, r_list, K, T, T_recov = 0)

	h_prog = np.zeros((T+1,K))
	h_prog[0] = h_init
	for t in range(T):
		h_prog[t+1] = health_map(F_list[t].dot(h_prog[t]) + G_list[t].dot(doses[t]) + q_list[t]*doses[t]**2 + r_list[t], doses[t], t)
	return h_prog

def health_prog_quad(h_init, T, alpha = None, beta = None, gamma = None, doses = None, health_map = lambda h,d,t: h):
	K = h_init.shape[0]
	alpha, beta, gamma, doses = check_prog_parms(alpha, beta, gamma, doses, K, T)

	h_prog = np.zeros((T+1,K))
	h_prog[0] = h_init
	for t in range(T):
		h_prog[t+1] = health_map(h_prog[t] - alpha[t]*doses[t] - beta[t]*doses[t]**2 + gamma[t], doses[t], t)
	return h_prog

def health_prog_lin(h_init, T, alpha = None, beta = None, gamma = None, doses = None, d_parm = None, health_map = lambda h,d,t: h):
	K = h_init.shape[0]
	alpha, beta, gamma, doses = check_prog_parms(alpha, beta, gamma, doses, K, T)
	if d_parm is None:
		d_parm = np.zeros((T, K))

	h_prog = np.zeros((T + 1, K))
	h_prog[0] = h_init
	for t in range(T):
		h_prog[t+1] = health_map(h_prog[t] - alpha[t]*doses[t] - beta[t]*d_parm[t]*(2*doses[t] - d_parm[t]) + gamma[t], doses[t], t)
	return h_prog

def health_prog_act(h_init, T, alpha = None, beta = None, gamma = None, doses = None, is_target = None, health_map = lambda h,d,t: h):
	K = h_init.shape[0]
	alpha, beta, gamma, doses = check_prog_parms(alpha, beta, gamma, doses, K, T)
	return health_prog_act_range(h_init, 0, T, alpha = alpha, beta = beta, gamma = gamma, doses = doses,
								 is_target = is_target, health_map = health_map)

def health_prog_act_range(h_init, t_s, T, alpha = None, beta = None, gamma = None, doses = None, is_target = None, health_map = lambda h,d,t: h):
	K = h_init.shape[0]
	if t_s > T or t_s < 0:
		raise ValueError("t_s must be an integer in [0, {0}]".format(T))

	alpha, beta, gamma, doses = check_prog_parms(alpha, beta, gamma, doses, K, T, is_range = True)
	if is_target is None:
		is_target = np.full((K,), False)
	if is_target.shape not in [(K,), (K, 1)]:
		raise ValueError("is_target must have dimensions ({0},)".format(K))

	# Allow separate health maps for targets and organs-at-risk.
	if callable(health_map):
		health_map_ptv = health_map_oar = health_map   # TODO: Do we need to make a deep copy to avoid clashes?
	elif isinstance(health_map, dict):
		if "target" not in health_map:
			raise ValueError("health_map must contain key 'target'")
		if "organ" not in health_map:
			raise ValueError("health_map must contain key 'organ'")
		health_map_ptv = health_map["target"]
		health_map_oar = health_map["organ"]
	else:
		raise ValueError("health_map must be a function or a dictionary of functions")

	# h_prog[:, is_target] = health_prog_quad(h_init[is_target], T, alpha[:, is_target], beta[:, is_target],
	#									   gamma[:, is_target],
	#									   doses[:, is_target], d_parm[:, is_target], health_map)
	# h_prog[:, ~is_target] = health_prog_quad(h_init[~is_target], T, alpha[:, ~is_target], beta[:, ~is_target],
	#										 gamma[:, ~is_target],
	#										 doses[:, ~is_target], health_map)

	h_prog = np.zeros((T - t_s + 1, K))
	h_prog[0] = h_init
	h_idx = 0
	for t in range(t_s, T):
		h_quad_expr = h_prog[h_idx] - alpha[t]*doses[t] - beta[t]*doses[t]**2 + gamma[t]
		h_prog[h_idx+1,is_target] = health_map_ptv(h_quad_expr[is_target], doses[t,is_target], t)
		h_prog[h_idx+1,~is_target] = health_map_oar(h_quad_expr[~is_target], doses[t,~is_target], t)
		h_idx = h_idx + 1
	return h_prog

def health_prog_est(h_init, T, alpha = None, beta = None, gamma = None, doses = None, d_parm = None, is_target = None, health_map = lambda h,d,t: h):
	K = h_init.shape[0]
	alpha, beta, gamma, doses = check_prog_parms(alpha, beta, gamma, doses, K, T)
	if d_parm is None:
		d_parm = np.zeros((T, K))
	if is_target is None:
		is_target = np.full((K,), False)
	if is_target.shape not in [(K,), (K,1)]:
		raise ValueError("is_target must have dimensions ({0},)".format(K))

	# Allow separate health maps for targets and organs-at-risk.
	if callable(health_map):
		health_map_ptv = health_map_oar = health_map   # TODO: Do we need to make a deep copy to avoid clashes?
	elif isinstance(health_map, dict):
		if "target" not in health_map:
			raise ValueError("health_map must contain key 'target'")
		if "organ" not in health_map:
			raise ValueError("health_map must contain key 'organ'")
		health_map_ptv = health_map["target"]
		health_map_oar = health_map["organ"]
	else:
		raise ValueError("health_map must be a function or a dictionary of functions")

	h_prog = np.zeros((T + 1, K))
	h_prog[:,is_target] = health_prog_lin(h_init[is_target], T, alpha[:,is_target], beta[:,is_target], gamma[:,is_target],
										  doses[:,is_target], d_parm[:,is_target], health_map_ptv)
	h_prog[:,~is_target] = health_prog_quad(h_init[~is_target], T, alpha[:,~is_target], beta[:,~is_target], gamma[:,~is_target],
											doses[:,~is_target], health_map_oar)
	return h_prog
