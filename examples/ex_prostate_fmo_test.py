import numpy as np
import matplotlib
matplotlib.use("TKAgg")

from fractionation.init_funcs import dyn_init_dose
from fractionation.quad_funcs import dyn_quad_treat
from fractionation.quad_admm_funcs import dyn_quad_treat_admm

from fractionation.utilities.file_utils import yaml_to_dict
from fractionation.utilities.data_utils import line_integral_mat, health_prog_act
from fractionation.utilities.plot_utils import *

from example_utils import simple_structures, simple_colormap

def main(figpath = "", datapath = ""):
	# Import data.
	figprefix = "ex_prostate_FMO_stanford_test-admm-"
	patient_bio, patient_rx, visuals = yaml_to_dict(datapath + "ex_prostate_FMO_stanford_test.yml")

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
	h_curves = [{"h": h_prog, "label": "Untreated", "kwargs": {"color": colors[1]}}]

	# TODO: Should we add slack to PTV health dynamics constraint or upper/lower health bounds?
	b_zeros = np.zeros(beta.shape)
	res_init = dyn_init_dose(A_list, alpha, b_zeros, gamma, h_init, patient_rx, use_slack = True, slack_weight = 1e4, solver = "MOSEK")
	d_init = res_init["doses"]
	h_equal = health_prog_act(h_init, T, alpha, b_zeros, gamma, d_init, patient_rx["is_target"])

	# Plot initial health, and treatment curves.
	# plot_health(h_equal, curves = h_curves, stepsize = 10, bounds = (health_lower, health_upper), title = "Initial Stage: Health Status vs. Time", 
	#			label = "Treated", color = colors[0], one_idx = True)
	# plot_treatment(d_init, stepsize = 10, bounds = (dose_lower, dose_upper), title = "Initial Stage: Treatment Dose vs. Time", one_idx = True)

	# Dynamic treatment.
	# res_dynamic = dyn_quad_treat(A_list, alpha, beta, gamma, h_init, patient_rx, use_slack = True, slack_weight = 1e4,
	#							 max_iter = 30, solver = "MOSEK", ccp_verbose = True, d_init = d_init)
	res_dynamic = dyn_quad_treat_admm(A_list, alpha, beta, gamma, h_init, patient_rx, use_slack = False, # use_slack = True, slack_weight = 1e4,
	 								  ccp_max_iter = 15, solver = "MOSEK", rho = 5, admm_max_iter = 100, admm_verbose = True,
	 								  d_init = d_init)
	print("Dynamic Treatment")
	print("Status:", res_dynamic["status"])
	print("Objective:", res_dynamic["obj"])
	print("Solve Time:", res_dynamic["solve_time"])
	print("Iterations:", res_dynamic["num_iters"])
	print("Beam Max:", np.max(res_dynamic["beams"]))
	print("Beam Sum:", np.sum(res_dynamic["beams"]))

	# Plot total slack in health dynamics per iteration.
	# if "health_slack" in res_dynamic:
	#	plot_slacks(res_dynamic["health_slack"], filename = figpath + figprefix + "slacks.png")

	# Plot dynamic health and treatment curves.
	if "primal" in res_dynamic and "dual" in res_dynamic:
		plot_residuals(res_dynamic["primal"], res_dynamic["dual"], semilogy = True, filename = figpath + figprefix + "residuals.png")

	h_curves += [{"h": h_equal, "label": "Initial", "kwargs": {"color": colors[2]}}]
	d_curves  = [{"d": d_init, "label": "Initial", "kwargs": {"color": colors[2]}}]
	plot_health(res_dynamic["health"], curves = h_curves, stepsize = 10, bounds = (health_lower, health_upper), label = "Treated", 
	 				color = colors[0], one_idx = True, filename = figpath + figprefix + "health.png")
	plot_treatment(res_dynamic["doses"], curves = d_curves, stepsize = 10, bounds = (dose_lower, dose_upper), one_idx = True, 
	 				filename = figpath + figprefix + "doses.png")

if __name__ == '__main__':
	main(figpath = "C:/Users/Anqi/Documents/Software/fractionation/examples/output/figures/",
		 datapath = "C:/Users/Anqi/Documents/Software/fractionation/examples/data/")
