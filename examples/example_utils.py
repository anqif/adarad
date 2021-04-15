import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from adarad.utilities.data_utils import circle, ellipse, cardioid

def save_data(result, savepath = "", fileprefix = ""):
	if "primal" in result and "dual" in result:
		residuals = np.column_stack([result["primal"], result["dual"]])
		np.save(savepath + fileprefix + "admm_residuals.npy", residuals)
	np.save(savepath + fileprefix + "beams.npy", result["beams"])
	np.save(savepath + fileprefix + "health.npy", result["health"])
	np.save(savepath + fileprefix + "doses.npy", result["doses"])

def simple_structures(m_grid, n_grid, xlim = (-1,1), ylim = (-1,1)):
	# Create polar grid.
	x = np.linspace(xlim[0], xlim[1], m_grid)
	y = np.linspace(ylim[0], ylim[1], n_grid)
	x_grid, y_grid = np.meshgrid(x, y)
	r_grid = np.sqrt(x_grid**2 + y_grid**2)
	theta_grid = np.arctan2(y_grid, x_grid)

	# Define structure regions.
	# Body voxels (s = 4).
	regions = np.full((n_grid, m_grid), 4)

	# PTV (s = 0).
	# regions[circle(x_grid, y_grid, (0,0), 0.35) <= 0] = 0
	regions[cardioid(x_grid, y_grid, 0.125, (-0.1,0), np.pi) <= 0] = 0

	# OAR (s = 1).
	r_inner = 0.6
	r_outer = 0.85
	r0 = (r_inner + r_outer)/2
	theta0_l = 3*np.pi/8
	theta0_r = np.pi/8
	r_width = (r_outer - r_inner)/2
	circle_l = circle(x_grid, y_grid, (r0*np.cos(theta0_l), r0*np.sin(theta0_l)), r_width) <= 0
	circle_r = circle(x_grid, y_grid, (r0*np.cos(theta0_r), r0*np.sin(theta0_r)), r_width) <= 0
	slice_c = (r_grid >= r_inner) & (r_grid <= r_outer) & (theta_grid <= theta0_l) & (theta_grid >= theta0_r)
	regions[circle_l | circle_r | slice_c] = 1
	# regions[(r_grid >= 0.5) & (r_grid <= 0.65) & (theta_grid <= np.pi/2)] = 1

	# OAR (s = 2).
	x0 = -0.5  # -0.375
	y0 = 0.45  # 0.65
	x_width = 0.25  # 0.1
	y_width = 0.35  # 0.18
	regions[ellipse(x_grid, y_grid, (x0, y0), (x_width, y_width), np.pi/3) <= 0] = 2
	
	# OAR (s = 3).
	x0 = -0.1
	y0 = -0.7
	x_width = 0.4
	y_width = 0.2
	regions[ellipse(x_grid, y_grid, (x0, y0), (x_width, y_width)) <= 0] = 3
	return x_grid, y_grid, regions

def simple_colormap(one_idx = False):
	K = 5
	struct_cmap = ListedColormap(['red', 'blue', 'green', 'orange', 'white'])
	struct_bounds = np.arange(K+1) - 0.5 + int(one_idx)
	# struct_cmap = truncate_cmap(plt.cm.rainbow, 0, 0.1*K, n = K)
	# struct_bounds = np.linspace(0, K, K+1)
	struct_norm = BoundaryNorm(struct_bounds, struct_cmap.N)
	struct_kw = {"cmap": struct_cmap, "norm": struct_norm, "alpha": 0.75}
	return struct_kw

# Fast proliferating tumors.
def tumor_fast_parms():
	# Tumor.
	tumor_parms = {"alpha": 0.35,           # LQ constant.
				   "alpha_over_beta": 10,   # alpha/beta ratio.
				   "T_k": 28,               # Kickoff time.
				   "T_pd": 3,               # Doubling time.
				   "tau_S": 1,              # Resensitization time.
				   "tau_R": 0.5,            # Repair time.
				   "half_sigma_sq": 0.02}   # Variance of Gaussian distribution of alpha.

	# Late-responding normal tissue.
	norm_late_parms =  {"alpha": 0.315,
						"alpha_over_beta": 3,
						"tau_R": 4}

	# Early-responding normal tissue.
	norm_early_parms = {"alpha": 0.315,
						"alpha_over_beta": 10,
						"tau_R": 0.5}

	norm_parms = {"late": norm_late_parms, "early": norm_early_parms}
	return tumor_parms, norm_parms

# Slowly proliferating tumors.
def tumor_slow_parms():
	# Tumor.
	tumor_parms = {"alpha": 0.10,            # LQ constant.
				   "alpha_over_beta": 1.5,   # alpha/beta ratio.
				   "T_k": 300,               # Kickoff time.
				   "T_pd": 40,               # Doubling time.
				   "tau_S": 2,               # Resensitization time.
				   "tau_R": 1.9,             # Repair time.
				   "half_sigma_sq": 1/30 }   # Variance of Gaussian distribution of alpha. (Here it's beta/3).

	# Late-responding normal tissue.
	norm_late_parms  = {"alpha": 0.315,
					    "alpha_over_beta": 3,
					    "tau_R": 4}

	# Early-responding normal tissue.
	norm_early_parms = {"alpha": 0.315,
						"alpha_over_beta": 10,
						"tau_R": 0.5}

	norm_parms = {"late": norm_late_parms, "early": norm_early_parms}
	return tumor_parms, norm_parms

def med_to_lq_parms(med_parms, T_tot, is_tumor = False):
	alpha_lq = med_parms["alpha"]
	beta_lq = med_parms["alpha_over_beta"]/med_parms["alpha"]
	if is_tumor:
		beta_lq = beta_lq - med_parms["half_sigma_sq"]

	alpha_vec = np.full((T_tot,), alpha_lq)
	beta_vec = np.full((T_tot,), beta_lq)
	gamma_vec = np.zeros((T_tot,))
	if is_tumor and med_parms["T_k"] <= T_tot:
		T_k_idx = int(med_parms["T_k"] - 1)
		gamma_vec[T_k_idx:] = 1/med_parms["T_pd"]

	lq_parms = {"alpha": alpha_vec, "beta": beta_vec, "gamma": gamma_vec}
	return lq_parms

def simple_parms(T = 60, fast = True):
	tumor_parms, norm_parms = tumor_fast_parms() if fast else tumor_slow_parms()
	tumor_lq = med_to_lq_parms(tumor_parms, T, is_tumor = True)
	norm_late_lq = med_to_lq_parms(norm_parms["late"], T, is_tumor = False)
	norm_early_lq = med_to_lq_parms(norm_parms["early"], T, is_tumor = False)

	K = 5
	alpha = np.zeros((T, K))
	beta = np.zeros((T, K))
	gamma = np.zeros((T, K))

	# PTV (s = 0).
	alpha[:,0] = tumor_lq["alpha"]
	beta[:,0] = tumor_lq["beta"]
	gamma[:,0] = tumor_lq["gamma"]

	# OAR (s = 1 and 3).
	alpha[:,1] = alpha[:,3] = norm_late_lq["alpha"]
	beta[:,1] = beta[:,3] = norm_late_lq["beta"]
	gamma[:,1] = gamma[:,3] = norm_late_lq["gamma"]

	# OAR (s = 2).
	alpha[:,2] = norm_early_lq["alpha"]
	beta[:,2] = norm_early_lq["beta"]
	gamma[:,2] = norm_early_lq["gamma"]

	# Body voxels (s = 4).
	alpha[:,4] = 0.1*norm_early_lq["alpha"]
	beta[:,4] = 0.01*norm_early_lq["beta"]
	gamma[:,4] = norm_early_lq["gamma"]

	return alpha, beta, gamma

def proc_parms(struct_dict, T, N0):
	for key, value in struct_dict.items():
		D_max = value["D_max"]
		D_max_per_frac = D_max/T

		if "alpha" in value.keys():
			beta = value["alpha_over_beta"]/value["alpha"]
			H_min_per_frac = -(value["alpha"]*D_max_per_frac + beta*D_max_per_frac**2)
		else:
			H_min_per_frac = -(D_max_per_frac + (1/value["alpha_over_beta"])*D_max_per_frac**2)

		struct_dict[key]["D_max_per_frac"] = D_max_per_frac
		struct_dict[key]["H_min_per_frac"] = H_min_per_frac
		struct_dict[key]["H_min"] = T*H_min_per_frac
	return struct_dict

def head_and_neck_parms(T = 35, N0 = 100):
	struct_dict = {"PTV 0-46 Gy": {"alpha": 0.35, "alpha_over_beta": 10, "D_max": 80},
	 			   "Spinal Cord": {"alpha_over_beta": 5, "D_max": 45},
	 			   "Brainstem":   {"alpha_over_beta": 5, "D_max": 50},
	 			   "Cochlea":     {"alpha_over_beta": 3, "D_max": 45},
	 			   "Parotid":     {"alpha_over_beta": 4, "D_max": 28},
	 			   "SMG":         {"alpha_over_beta": 4, "D_max": 28},
	 			   "Oral Cavity": {"alpha_over_beta": 3, "D_max": 50},
	 			   "Ring":        {"alpha_over_beta": 3, "D_max": 75},
	 			   "Larynx":      {"alpha_over_beta": 3, "D_max": 50},
	 			   "MCS":         {"alpha_over_beta": 4, "D_max": 35},
	 			   "MCM":         {"alpha_over_beta": 4, "D_max": 35},
	 			   "MCI":         {"alpha_over_beta": 4, "D_max": 35},
	 			   "MCP":         {"alpha_over_beta": 4, "D_max": 35},
	 			   "Oesophagus":  {"alpha_over_beta": 3, "D_max": 35}
	 			  }
	return proc_parms(struct_dict, T, N0)

def prostate_parms(T = 45, N0 = 100):
	struct_dict = {"Prostate":     {"alpha": 0.15, "alpha_over_beta": 3, "D_max": 80}, 
				   "Urethra":      {"alpha_over_beta": 5, "D_max": 65},
				   "Bladder":      {"alpha_over_beta": 5, "D_max": 65},
				   "Rectum":       {"alpha_over_beta": 5, "D_max": 50},
				   "Femoral Head": {"alpha_over_beta": 4, "D_max": 40},
				   "Ring":         {"alpha_over_beta": 3, "D_max": 75}
	              }
	return proc_parms(struct_dict, T, N0)