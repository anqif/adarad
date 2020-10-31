import numpy as np
import matplotlib
# matplotlib.use("TKAgg")

from fractionation.quad_funcs import dyn_quad_treat, mpc_quad_treat
from fractionation.quad_admm_funcs import dyn_quad_treat_admm, mpc_quad_treat_admm

from fractionation.utilities.file_utils import yaml_to_dict
from fractionation.utilities.data_utils import line_integral_mat, health_prog_act
from fractionation.utilities.plot_utils import *

from example_utils import simple_structures, simple_colormap

def main(figpath = "", datapath = ""):
	# Import data.
	figprefix = "ex_prostate_FMO_stanford_noisy-"
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
	is_target = patient_rx["is_target"]

	# Actual health status transition function.
	mu = 0
	sigma = 0.05
	d_noise = mu + sigma*np.random.randn(T,K)

	def health_map_ptv(h, d, t):
		d_eps = d_noise[t,is_target]
		h_delta_lin = np.multiply(alpha[t,is_target], d_eps)
		h_delta_quad = np.multiply(beta[t,is_target], d_eps**2 + 2*np.multiply(d, d_eps))
		h_delta = h_delta_lin + h_delta_quad
		h_delta[d + d_eps <= 0] = 0
		return h - h_delta

	def health_map_oar(h, d, t):
		d_eps = d_noise[t,~is_target]
		h_delta_lin = np.multiply(alpha[t,~is_target], d_eps)
		h_delta_quad = np.multiply(beta[t,~is_target], d_eps**2 + 2*np.multiply(d, d_eps))
		h_delta = h_delta_lin + h_delta_quad
		h_delta[d + d_eps <= 0] = 0
		return h - h_delta
	health_map = {"target": health_map_ptv, "organ": health_map_oar}

	# Health prognosis.
	prop_cycle = plt.rcParams["axes.prop_cycle"]
	colors = prop_cycle.by_key()["color"]
	h_prog = health_prog_act(h_init, T, gamma = gamma, is_target = is_target)
	h_curves = [{"h": h_prog, "label": "Untreated", "kwargs": {"color": colors[1]}}]

	# Dynamic treatment.
	res_dynamic = dyn_quad_treat(A_list, alpha, beta, gamma, h_init, patient_rx, health_map = health_map, use_slack = True, 
								 slack_weight = 1e4, max_iter = 15, solver = "MOSEK", ccp_verbose = True)
	# res_dynamic = dyn_quad_treat_admm(A_list, alpha, beta, gamma, h_init, patient_rx, health_map = health_map,
	# 							use_slack = True, slack_weight = 1e4, ccp_max_iter = 15, solver = "MOSEK", rho = 10,
	# 							admm_max_iter = 500, admm_verbose = True)
	print("Dynamic Treatment")
	print("Status:", res_dynamic["status"])
	print("Objective:", res_dynamic["obj"])
	print("Solve Time:", res_dynamic["solve_time"])
	print("Iterations:", res_dynamic["num_iters"])
	print("Beam Max:", np.max(res_dynamic["beams"]))
	print("Beam Sum:", np.sum(res_dynamic["beams"]))

	# Plot dynamic health and treatment curves.
	if "primal" in res_dynamic and "dual" in res_dynamic:
		plot_residuals(res_dynamic["primal"], res_dynamic["dual"], semilogy = True, show = False, filename = figpath + figprefix + "residuals.png")
	dyn_doses_act = np.maximum(res_dynamic["doses"] + d_noise, 0)   # Compute actual doses (including random error).
	plot_health(res_dynamic["health"], curves = h_curves, stepsize = 10, bounds = (health_lower, health_upper), label = "Treated", 
	 				color = colors[0], one_idx = True, show = False, filename = figpath + figprefix + "health.png")
	plot_treatment(dyn_doses_act, stepsize = 10, bounds = (dose_lower, dose_upper), one_idx = True, 
	 				show = False, filename = figpath + figprefix + "doses.png")

	# Dynamic treatment with MPC.
	print("\nStarting MPC algorithm...")
	res_mpc = mpc_quad_treat(A_list, alpha, beta, gamma, h_init, patient_rx, health_map = health_map, use_ccp_slack = True,
							 ccp_slack_weight = 1e4, use_mpc_slack = True, mpc_slack_weights = 1e4, max_iter = 100,   # max_iter = 15,
							 solver = "MOSEK", mpc_verbose = True)
	# res_mpc = mpc_quad_treat_admm(A_list, alpha, beta, gamma, h_init, patient_rx, health_map = health_map, use_ccp_slack = True,
	#						ccp_slack_weight = 1e4, ccp_max_iter = 15, use_mpc_slack = True, mpc_slack_weights = 1e4,
	#						solver = "MOSEK", rho = 10, admm_max_iter = 100, mpc_verbose = True)
	print("\nMPC Treatment")
	print("Status:", res_mpc["status"])
	print("Objective:", res_mpc["obj"])
	print("Solve Time:", res_mpc["solve_time"])
	print("Iterations:", res_mpc["num_iters"])
	print("Beam Max:", np.max(res_dynamic["beams"]))
	print("Beam Sum:", np.sum(res_dynamic["beams"]))

	# Compare one-shot dynamic and MPC treatment results.
	d_curves = [{"d": dyn_doses_act, "label": "Naive", "kwargs": {"color": colors[0]}}]
	h_naive = [{"h": res_dynamic["health"], "label": "Treated (Naive)", "kwargs": {"color": colors[0]}}]
	h_curves = h_naive + h_curves

	mpc_doses_act = np.maximum(res_mpc["doses"] + d_noise, 0)	# Compute actual doses (including random error).
	plot_health(res_mpc["health"], curves = h_curves, stepsize = 10, bounds = (health_lower, health_upper),
				label = "Treated (MPC)", color = colors[2], one_idx = True, show =  False, filename = figpath + figprefix + "mpc-health.png")
	plot_treatment(mpc_doses_act, curves = d_curves, stepsize = 10, bounds = (dose_lower, dose_upper),
				label = "MPC", color = colors[2], one_idx = True, show = False, filename = figpath + figprefix + "mpc-doses.png")

if __name__ == "__main__":
	main(figpath = "/home/anqif/fractionation/examples/output/figures/",
		 datapath = "/home/anqif/fractionation/examples/data/")
