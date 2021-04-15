import numpy as np
import matplotlib
matplotlib.use("TKAgg")

from adarad.quad_funcs import dyn_quad_treat
from adarad.quad_admm_funcs import dyn_quad_treat_admm
from adarad.utilities.plot_utils import *
from adarad.utilities.data_utils import line_integral_mat, health_prog_act

from example_utils import simple_structures, simple_colormap

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
	# plot_structures(x_grid, y_grid, regions, title = "Anatomical Structures", one_idx = True, **struct_kw)
	# plot_structures(x_grid, y_grid, regions, one_idx = True, filename = figpath + "ex_cardioid5_structures.png", **struct_kw)

	# Problem data.
	K = np.unique(regions).size   # Number of structures.
	A, angles, offs_vec = line_integral_mat(regions, angles = n_angle, n_bundle = n_bundle, offset = offset)
	A = A/n_grid
	A_list = T*[A]

	alpha = np.array(T*[[0.01, 0.50, 0.25, 0.15, 0.005]])
	beta = np.array(T*[[0.001, 0.05, 0.025, 0.015, 0.0005]])
	gamma = np.array(T*[[0.05, 0, 0, 0, 0]])
	h_init = np.array([1] + (K-1)*[0])
	is_target = np.array([True] + (K-1)*[False])

	# Health prognosis.
	prop_cycle = plt.rcParams['axes.prop_cycle']
	colors = prop_cycle.by_key()['color']
	h_prog = health_prog_act(h_init, T, gamma = gamma)
	curves = [{"h": h_prog, "label": "Untreated", "kwargs": {"color": colors[1]}}]

	# Penalty functions.
	w_lo = np.array([0] + (K-1)*[1])
	w_hi = np.array([1] + (K-1)*[0])
	rx_health_weights = [w_lo, w_hi]
	rx_health_goal = np.zeros((T,K))
	rx_dose_weights = np.array([1, 1, 1, 1, 0.25])
	rx_dose_goal = np.zeros((T,K))
	patient_rx = {"is_target": is_target, "dose_goal": rx_dose_goal, "dose_weights": rx_dose_weights,
				  "health_goal": rx_health_goal, "health_weights": rx_health_weights}

	# Beam constraints.
	beam_upper = np.full((T,n), 1.0)
	# beam_upper = np.full((T,n), np.inf)
	patient_rx["beam_constrs"] = {"upper": beam_upper}

	# Dose constraints.
	dose_lower = np.zeros((T,K))
	dose_upper = np.full((T,K), 20)   # Upper bound on doses.
	# dose_upper = np.full((T,K), np.inf)
	patient_rx["dose_constrs"] = {"lower": dose_lower, "upper": dose_upper}

	# Health constraints.
	health_lower = np.full((T,K), -np.inf)
	health_upper = np.full((T,K), np.inf)
	health_lower[:,1] = -1.0     # Lower bound on OARs.
	health_lower[:,2] = -2.0
	health_lower[:,3] = -2.0
	health_lower[:,4] = -3.0
	health_upper[:15,0] = 2.0    # Upper bound on PTV for t = 1,...,15.
	health_upper[15:,0] = 0.05   # Upper bound on PTV for t = 16,...,20.

	patient_rx["health_constrs"] = {"lower": health_lower, "upper": health_upper}
	# patient_rx["health_constrs"] = {"lower": health_lower[:,~is_target], "upper": health_upper[:,is_target]}

	# Dynamic treatment.
	res_dynamic = dyn_quad_treat(A_list, alpha, beta, gamma, h_init, patient_rx, use_slack = True, slack_weight = 1e4,
								 max_iter = 15, solver = "MOSEK", ccp_verbose = True, auto_init = False, full_hist = True)
	# res_dynamic = dyn_quad_treat_admm(A_list, alpha, beta, gamma, h_init, patient_rx, use_slack = True, slack_weight = 1e4,
	# 								  ccp_max_iter = 15, solver = "MOSEK", rho = 5, admm_max_iter = 500, admm_verbose = True,
	#								  auto_init = True)
	print("Dynamic Treatment")
	print("Status:", res_dynamic["status"])
	print("Objective:", res_dynamic["obj"])
	print("Solve Time:", res_dynamic["solve_time"])
	print("Iterations:", res_dynamic["num_iters"])

	# Plot dynamic beam, health, and treatment curves.
	# plot_residuals(res_dynamic["primal"], res_dynamic["dual"], semilogy = True)
	# plot_beams(res_dynamic["beams"], angles = angles, offsets = offs_vec, n_grid = n_grid, stepsize = 1,
	#		   cmap = transp_cmap(plt.cm.Reds, upper = 0.5), title = "Beam Intensities vs. Time", one_idx = True,
	#		   structures = (x_grid, y_grid, regions), struct_kw = struct_kw)

	print("Plotting results for all iterations")
	for i in range(res_dynamic["num_iters"]):
		# plot_health(res_dynamic["health_hist"][i], curves = curves, stepsize = 10, bounds = (health_lower, health_upper),
		#			label = "Treated", color = colors[0], one_idx = True)
		# plot_treatment(res_dynamic["doses_hist"][i], stepsize = 10, bounds = (dose_lower, dose_upper), one_idx = True)

		# plot_health(res_dynamic["health_hist"][i], curves = curves, stepsize = 10, bounds = (health_lower, health_upper),
		#			label = "Treated", color = colors[0], one_idx = True, show = False, 
		#			filename = figpath + "ex1_health_iter_{0}.png".format(i+1))
		# plot_treatment(res_dynamic["doses_hist"][i], stepsize = 10, bounds = (dose_lower, dose_upper),
		#		    one_idx = True, show = False, filename = figpath + "ex1_doses_iter_{0}.png".format(i+1))

		h_dict = {"v": res_dynamic["health_hist"][i], "varname": "h", "curves": curves, "stepsize": 10, 
				  "bounds": (health_lower, health_upper), "label": "Treated", "color": colors[0], "one_idx": True, 
				  "one_shift": False}
		d_dict = {"v": res_dynamic["doses_hist"][i], "varname": "d", "stepsize": 10, "bounds": (dose_lower, dose_upper), 
				  "one_idx": True, "one_shift": True}
		# plot_stacked([h_dict, d_dict], title = "Health Status and Treatment Dose vs. Time", figsize = (16,10))
		plot_stacked([h_dict, d_dict], figsize = (16,10), show = False, filename = figpath + "ex1_health_doses_iter_{0}.png".format(i+1))

if __name__ == '__main__':
	main(figpath = "C:/Users/Anqi/Documents/Software/adarad/examples/output/figures/movie/",
		 datapath = "C:/Users/Anqi/Documents/Software/adarad/examples/data/")
