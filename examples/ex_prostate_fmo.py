import numpy as np
import matplotlib
# matplotlib.use("TKAgg")

from fractionation.quad_funcs import dyn_quad_treat
from fractionation.quad_admm_funcs import dyn_quad_treat_admm

from fractionation.utilities.file_utils import yaml_to_dict
from fractionation.utilities.data_utils import health_prog_act
from fractionation.utilities.plot_utils import *

from example_utils import simple_structures, simple_colormap

def main(figpath = "", datapath = ""):
	# Import data.
	figprefix = "ex_prostate_FMO_stanford-trial_05-"
	patient_bio, patient_rx, visuals = yaml_to_dict(datapath + "ex_prostate_FMO_stanford.yml")

	# Patient data.
	A_list = patient_bio["dose_matrices"]
	alpha = patient_bio["alpha"]
	beta = patient_bio["beta"]
	gamma = patient_bio["gamma"]
	h_init = patient_bio["health_init"]

	# Treatment data.
	T = len(A_list)
	K, n = A_list[0].shape
	dose_lower = patient_rx["dose_constrs"]["lower"]
	dose_upper = patient_rx["dose_constrs"]["upper"]
	health_lower = patient_rx["health_constrs"]["lower"]
	health_upper = patient_rx["health_constrs"]["upper"]
	
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
	print("Beam Max:", np.max(res_dynamic["beams"]))
	print("Beam Sum:", np.sum(res_dynamic["beams"]))

	# Plot total slack in health dynamics per iteration.
	if "health_slacks" in res_dynamic:
		plot_slacks(res_dynamic["health_slacks"], show = False, filename = figpath + figprefix + "slacks.png")

	# Plot dynamic health and treatment curves.
	if "primal" in res_dynamic and "dual" in res_dynamic:
		plot_residuals(res_dynamic["primal"], res_dynamic["dual"], semilogy = True, show = False, filename = figpath + figprefix + "residuals.png")
	plot_health(res_dynamic["health"], curves = curves, stepsize = 10, bounds = (health_lower, health_upper), label = "Treated", 
	 				color = colors[0], one_idx = True, show = False, filename = figpath + figprefix + "health.png")
	plot_treatment(res_dynamic["doses"], stepsize = 10, bounds = (dose_lower, dose_upper), one_idx = True, show = False, 
	 				filename = figpath + figprefix + "doses.png")

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
	# plot_health(ptv_health, curves = curves, stepsize = 10, label = "Linear-Quadratic", indices = np.array(iters), one_idx = True,
	# 			filename = figpath + "ex_prostate_FMO_PTV_health.png")

if __name__ == '__main__':
	main(figpath = "/home/anqif/fractionation/examples/output/figures/",
		 datapath = "/home/anqif/fractionation/examples/data/")
