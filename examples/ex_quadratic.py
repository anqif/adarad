import matplotlib
matplotlib.use("TKAgg")
from matplotlib.colors import LogNorm

from fractionation.quadratic.quad_funcs import dyn_quad_treat
from fractionation.utilities.plot_utils import *
from fractionation.utilities.data_utils import line_integral_mat, health_prog_quad

from example_utils import simple_structures, simple_colormap, save_data

def main(figpath = "", datapath = ""):
	T = 20           # Length of treatment.
	n_grid = 1000
	offset = 5       # Displacement between beams (pixels).
	n_angle = 20     # Number of angles.
	n_bundle = 50    # Number of beams per angle.
	n = n_angle*n_bundle   # Total number of beams.

	# Display structures on a polar grid.
	x_grid, y_grid, regions = simple_structures(n_grid, n_grid)
	struct_kw = simple_colormap(one_idx = True)
	plot_structures(x_grid, y_grid, regions, title = "Anatomical Structures", one_idx = True, **struct_kw)
	# plot_structures(x_grid, y_grid, regions, one_idx = True, filename = figpath + "ex_cardioid5_structures.png", **struct_kw)

	# Problem data.
	K = np.unique(regions).size   # Number of structures.
	A, angles, offs_vec = line_integral_mat(regions, angles = n_angle, n_bundle = n_bundle, offset = offset)
	A = A/n_grid
	A_list = T*[A]

	alpha = np.array(T*[[0.01, 0.50, 0.25, 0.15, 0.0075]])
	beta = np.array(T*[[0.001, 0.05, 0.025, 0.015, 0.001]])
	gamma = np.array(T*[[1.05, 0.90, 0.75, 0.80, 0.95]])
	h_init = np.array([1] + (K-1)*[0])

    # Health prognosis.
	h_prog = health_prog_quad(h_init, T, gamma = gamma)
	curves = [{"h": h_prog, "label": "Untreated"}]

	# Penalty functions.
	w_lo = np.array([0] + (K-1)*[1])
	w_hi = np.array([1] + (K-1)*[0])
	rx_health_weights = [w_lo, w_hi]
	rx_health_goal = np.zeros((T,K))
	rx_dose_weights = np.array([0.01, 1, 1, 1, 0.001])
	rx_dose_goal = np.zeros((T,K))
	patient_rx = {"dose_goal": rx_dose_goal, "dose_weights": rx_dose_weights, \
				  "health_goal": rx_health_goal, "health_weights": rx_health_weights}

	# Beam constraints.
	beam_upper = np.full((T,n), 1.0)
	patient_rx["beam_constrs"] = {"upper": beam_upper}

	# Dose constraints.
	dose_lower = np.zeros((T,K))
	dose_upper = np.full((T,K), 25)   # Upper bound on doses.
	patient_rx["dose_constrs"] = {"lower": dose_lower, "upper": dose_upper}

	# Health constraints.
	health_lower = np.full((T,K), -np.inf)
	health_upper = np.full((T,K), np.inf)
	health_lower[:,1] = -20     # Lower bound on OARs.
	health_lower[:,2] = -10
	health_lower[:,3] = -10
	health_lower[:,4] = -30
	health_upper[:15,0] = 25    # Upper bound on PTV for t = 1,...,15.
	health_upper[15:,0] = 5     # Upper bound on PTV for t = 16,...,20.

	is_target = np.array([True] + (K-1)*[False])
	patient_rx["is_target"] = is_target
	patient_rx["health_constrs"] = {"lower": health_lower[:,~is_target], "upper": health_upper[:,is_target]}

	# Dynamic treatment.
	# res_dynamic = dyn_quad_treat(A_list, alpha, beta, gamma, h_init, patient_rx, solver = "MOSEK")
	res_dynamic = dyn_quad_treat(A_list, alpha, beta, gamma, h_init, patient_rx, max_iter = 1000, solver = "MOSEK", ccp_verbose = True)
	print("Dynamic Treatment")
	print("Status:", res_dynamic["status"])
	print("Objective:", res_dynamic["obj"])
	print("Solve Time:", res_dynamic["solve_time"])
	print("Iterations:", res_dynamic["num_iters"])

	# Set beam colors on logarithmic scale.
	b_min = np.min(res_dynamic["beams"][res_dynamic["beams"] > 0])
	b_max = np.max(res_dynamic["beams"])
	lc_norm = LogNorm(vmin = b_min, vmax = b_max)

	# Plot dynamic beam, health, and treatment curves.
	plot_beams(res_dynamic["beams"], angles = angles, offsets = offs_vec, n_grid = n_grid, stepsize = 1, cmap = transp_cmap(plt.cm.Reds, upper = 0.5), \
				title = "Beam Intensities vs. Time", one_idx = True, structures = (x_grid, y_grid, regions), struct_kw = struct_kw)
	plot_health(res_dynamic["health"], curves = curves, stepsize = 10, bounds = (health_lower, health_upper), title = "Health Status vs. Time", one_idx = True)
	plot_treatment(res_dynamic["doses"], stepsize = 10, bounds = (dose_lower, dose_upper), title = "Treatment Dose vs. Time", one_idx = True)

	# plot_beams(res_dynamic["beams"], angles = angles, offsets = offs_vec, n_grid = n_grid, stepsize = 1, cmap = transp_cmap(plt.cm.Reds, upper = 0.5), \
	#			one_idx = True, structures = (x_grid, y_grid, regions), struct_kw = struct_kw, filename = figpath + "ex_cardioid5_Dmax25_admm_beams.png")
	# plot_health(res_dynamic["health"], curves = curves, stepsize = 10, bounds = (health_lower, health_upper), one_idx = True, filename = figpath + "ex_cardioid5_Dmax25_admm_health.png")
	# plot_treatment(res_dynamic["doses"], stepsize = 10, bounds = (dose_lower, dose_upper), one_idx = True, filename = figpath + "ex_cardioid5_Dmax25_admm_doses.png")

if __name__ == '__main__':
	main(figpath = "/home/anqi/Dropbox/Research/Fractionation/Figures/", \
		 datapath = "/home/anqi/Documents/software/fractionation/examples/output/")