import numpy as np
import matplotlib
matplotlib.use("TKAgg")

from fractionation.quad_funcs import dyn_quad_treat
from fractionation.quad_admm_funcs import dyn_quad_treat_admm

from fractionation.utilities.file_utils import yaml_to_dict
from fractionation.utilities.data_utils import line_integral_mat, health_prog_act
from fractionation.utilities.plot_utils import *

from example_utils import simple_structures, simple_colormap

def main(figpath = "", datapath = ""):
	# Import data.
	patient_bio, patient_rx, visuals = yaml_to_dict(datapath + "ex_quadratic_model.yml")

	# Patient data.
	A_list = patient_bio["dose_matrices"]
	alpha = patient_bio["alpha"]
	beta = patient_bio["beta"]
	gamma = patient_bio["gamma"]
	h_init = patient_bio["health_init"]

	# Visualization data.
	angles = visuals["beam_angles"]
	offs_vec = visuals["beam_offsets"]
	x_grid = visuals["structures"]["x_grid"]
	y_grid = visuals["structures"]["y_grid"]
	regions = visuals["structures"]["regions"]
	n_grid = regions.shape[0]

	# Treatment data.
	T = len(A_list)
	K, n = A_list[0].shape
	dose_lower = patient_rx["dose_constrs"]["lower"]
	dose_upper = patient_rx["dose_constrs"]["upper"]
	health_lower = patient_rx["health_constrs"]["lower"]
	health_upper = patient_rx["health_constrs"]["upper"]

	# Display structures.
	struct_kw = simple_colormap(one_idx=True)
	plot_structures(x_grid, y_grid, regions, title="Anatomical Structures", one_idx=True, **struct_kw)
	# plot_structures(x_grid, y_grid, regions, one_idx = True, filename = figpath + "ex_cardioid5_structures.png", **struct_kw)

	# Health prognosis.
	prop_cycle = plt.rcParams["axes.prop_cycle"]
	colors = prop_cycle.by_key()["color"]
	h_prog = health_prog_act(h_init, T, gamma = gamma)
	curves = [{"h": h_prog, "label": "Untreated", "kwargs": {"color": colors[1]}}]

	# Dynamic treatment.
	res_dynamic = dyn_quad_treat(A_list, alpha, beta, gamma, h_init, patient_rx, use_slack = True, slack_weight = 1e4,
								 max_iter = 15, solver = "MOSEK", ccp_verbose = True)
	# res_dynamic = dyn_quad_treat_admm(A_list, alpha, beta, gamma, h_init, patient_rx, use_slack = True, slack_weight = 1e4,
	# 								  ccp_max_iter = 15, solver = "MOSEK", rho = 5, admm_max_iter = 50, admm_verbose = True)
	print("Dynamic Treatment")
	print("Status:", res_dynamic["status"])
	print("Objective:", res_dynamic["obj"])
	print("Solve Time:", res_dynamic["solve_time"])
	print("Iterations:", res_dynamic["num_iters"])

	# Plot total slack in health dynamics per iteration.
	# plot_slacks(res_dynamic["health_slack"], title = "Health Dynamics Slack vs. Iteration")
	# plot_slacks(res_dynamic["health_slack"], filename = figpath + "ex_cardioid_lq_ccp_slacks.png")

	# Plot dynamic beam, health, and treatment curves.
	# plot_residuals(res_dynamic["primal"], res_dynamic["dual"], semilogy = True)
	plot_beams(res_dynamic["beams"], angles = angles, offsets = offs_vec, n_grid = n_grid, stepsize = 1,
			   cmap = transp_cmap(plt.cm.Reds, upper = 0.5), title = "Beam Intensities vs. Time", one_idx = True,
			   structures = (x_grid, y_grid, regions), struct_kw = struct_kw)
	plot_health(res_dynamic["health"], curves = curves, stepsize = 10, bounds = (health_lower, health_upper),
				title = "Health Status vs. Time", label = "Treated", color = colors[0], one_idx = True)
	plot_treatment(res_dynamic["doses"], stepsize = 10, bounds = (dose_lower, dose_upper),
				   title = "Treatment Dose vs. Time", one_idx = True)

	# plot_beams(res_dynamic["beams"], angles = angles, offsets = offs_vec, n_grid = n_grid, stepsize = 1, cmap = transp_cmap(plt.cm.Reds, upper = 0.5),
	#			one_idx = True, structures = (x_grid, y_grid, regions), struct_kw = struct_kw, filename = figpath + "ex_cardioid_lq_ccp_beams.png")
	# plot_health(res_dynamic["health"], curves = curves, stepsize = 10, bounds = (health_lower, health_upper), one_idx = True, filename = figpath + "ex_cardioid_lq_ccp_health.png")
	# plot_treatment(res_dynamic["doses"], stepsize = 10, bounds = (dose_lower, dose_upper), one_idx = True, filename = figpath + "ex_cardioid_lq_ccp_doses.png")

	# Compare PTV health curves under linearized, linearized with slack, and linear-quadratic models.
	# sidx = 0
	# iters = np.array([1, 2, 5])
	# M = len(iters)
	#
	# ptv_health = np.zeros((T+1,M))
	# ptv_health_est = np.zeros((T+1,M))
	# ptv_health_opt = np.zeros((T+1,M))
	#
	# for j in range(M):
	# 	print("\nDynamic Treatment with Maximum Iterations {0}".format(iters[j]))
	# 	res_dynamic = dyn_quad_treat(A_list, alpha, beta, gamma, h_init, patient_rx, d_init = d_init, use_slack = True,
	# 								slack_weight = 1e4, max_iter = iters[j], solver = "MOSEK", ccp_verbose = True)
	# 	ptv_health[:,j] = res_dynamic["health"][:,sidx]
	# 	ptv_health_est[:,j] = res_dynamic["health_est"][:,sidx]
	# 	ptv_health_opt[:,j] = res_dynamic["health_opt"][:,sidx]
	#
	# curves = [{"h": ptv_health_est, "label": "Linearized", "kwargs": {"color": colors[3], "linestyle": "dashed"}}]
	# curves += [{"h": ptv_health_opt, "label": "Linearized with Slack", "kwargs": {"color": colors[2], "linestyle": "dashed"}}]
	# plot_health(ptv_health, curves = curves, stepsize = 10, label = "Linear-Quadratic", one_idx = True)
	# plot_health(ptv_health, curves = curves, stepsize = 10, label = "Linear-Quadratic", indices = np.array(iters), one_idx = True,
	# 			filename = figpath + "ex_cardioid_ccp_PTV_health.png")

if __name__ == '__main__':
	main(figpath = "/home/anqi/Dropbox/Research/Fractionation/Figures/",
		 datapath = "/home/anqi/Documents/software/fractionation/examples/data/")
