import numpy as np
import matplotlib
matplotlib.use("TKAgg")

from adarad.quad_funcs import dyn_quad_treat
from adarad.quad_admm_funcs import dyn_quad_treat_admm

from adarad.utilities.file_utils import yaml_to_dict
from adarad.utilities.data_utils import line_integral_mat, health_prog_act
from adarad.utilities.plot_utils import *

from example_utils import simple_structures, simple_colormap

def main(figpath = "", datapath = ""):
	# Import data.
	figprefix = "ex_head_and_neck_VMAT_TROT_test"
	patient_bio, patient_rx, visuals = yaml_to_dict(datapath + "ex_head_and_neck_VMAT_TROT_test.yml")

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
	# res_dynamic = dyn_quad_treat(A_list, alpha, beta, gamma, h_init, patient_rx, use_slack = True, slack_weight = 1e4, 
	#							   max_iter = 15, solver = "MOSEK", ccp_verbose = True)
	res_dynamic = dyn_quad_treat_admm(A_list, alpha, beta, gamma, h_init, patient_rx, use_slack = True, slack_weight = 1e4,
								      ccp_max_iter = 30, solver = "ECOS", rho = 5, admm_max_iter = 1000, admm_verbose = True,
								      max_iters = 1000, verbose = True)
	print("Dynamic Treatment")
	print("Status:", res_dynamic["status"])
	print("Iterations:", res_dynamic["num_iters"])
	print("Solve Time:", res_dynamic["solve_time"])
	print("Objective:", res_dynamic["obj"])
	print("Beam Max:", np.max(res_dynamic["beams"]))
	print("Beam Sum:", np.sum(res_dynamic["beams"]))

	# Plot total slack in health dynamics per iteration.
	if "health_slacks" in res_dynamic:
		plot_slacks(res_dynamic["health_slacks"], filename = figpath + "ex_head_and_neck_VMAT_TROT_test-admm-slacks.png")

	# Plot dynamic health and treatment curves.
	if "primal" in res_dynamic and "dual" in res_dynamic:
		plot_residuals(res_dynamic["primal"], res_dynamic["dual"], semilogy = True, filename = figpath + figprefix + "-admm-residuals.png")
	plot_treatment(res_dynamic["doses"], stepsize = 10, bounds = (dose_lower, dose_upper), one_idx = True, 
	 			   filename = figpath + "ex_head_and_neck_VMAT_TROT_test-doses.png")
	plot_health(res_dynamic["health"], curves = curves, stepsize = 10, bounds = (health_lower, health_upper), label = "Treated", 
	 			color = colors[0], one_idx = True, filename = figpath + figprefix + "-admm-health.png")

	# Plot health curve with rescaled axes (ignoring upper/lower bounds).
	h_all_vec = np.concatenate([res_dynamic["health"].ravel(), h_prog.ravel()])
	h_min = np.min(h_all_vec)
	h_max = np.max(h_all_vec)
	h_eps = 0.01*(h_max - h_min)
	h_min = np.floor(h_min - h_eps)
	h_max = np.ceil(h_max + h_eps)
	plot_health(res_dynamic["health"], curves = curves, stepsize = 10, bounds = (health_lower, health_upper), label = "Treated", 
	 			color = colors[0], one_idx = True, ylim = (h_min, h_max), filename = figpath + figprefix + "-admm-health_ylim.png")

	# Plot health curve of PTV only.
	fig = plt.figure(figsize = (12,8))
	plt.plot(range(T+1), res_dynamic["health"][:,0], label = "Treated", color = colors[0])
	plt.plot(range(T+1), h_prog[:,0], label = "Untreated", color = colors[1])
	plt.plot(range(1,T+1), health_lower[:,0], lw = 1, ls = "--", color = "cornflowerblue")
	plt.plot(range(1,T+1), health_upper[:,0], lw = 1, ls = "--", color = "cornflowerblue")
	# plt.title(patient_bio["names"][0])
	plt.title("Planning Target Volume (PTV)")
	plt.legend()
	plt.show()
	fig.savefig(figpath + figprefix + "-admm-health_ptv.png", bbox_inches = "tight", dpi = 300)

	# Plot health curve of surrounding OARs on same figure.
	# indices = [1, 2, 4, 5, 9, 11, 16, 10]
	indices = [1, 2, 4, 5, 11, 10]
	fig = plt.figure(figsize = (12,8))
	for i in indices:
		plt.plot(range(T+1), res_dynamic["health"][:,i], label = patient_bio["names"][i])
	plt.plot(range(1,T+1), np.zeros(T), lw = 1, ls = "--", color = "cornflowerblue")
	plt.title("Surrounding Organs-at-Risk (OARs)")
	plt.legend()
	plt.show()
	fig.savefig(figpath + figprefix + "-health_oar.png", bbox_inches = "tight", dpi = 300)

if __name__ == '__main__':
	main(figpath = "C:/Users/Anqi/Documents/Software/adarad/examples/output/figures/",
		 datapath = "C:/Users/Anqi/Documents/Software/adarad/examples/data/")
