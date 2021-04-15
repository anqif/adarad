import matplotlib
matplotlib.use("TKAgg")
from matplotlib.colors import LogNorm

from adarad.quad_funcs import dyn_quad_treat, mpc_quad_treat
from adarad.quad_admm_funcs import dyn_quad_treat_admm, mpc_quad_treat_admm
from adarad.utilities.plot_utils import *
from adarad.utilities.data_utils import line_integral_mat, health_prog_act

from example_utils import simple_structures, simple_colormap

def main(figpath = "", datapath = ""):
	np.random.seed(1)

	T = 20           # Length of treatment.
	n_grid = 1000
	offset = 5       # Displacement between beams (pixels).
	n_angle = 20     # Number of angles.
	n_bundle = 50    # Number of beams per angle.
	n = n_angle*n_bundle   # Total number of beams.

	# Display structures.
	x_grid, y_grid, regions = simple_structures(n_grid, n_grid)
	struct_kw = simple_colormap(one_idx = True)
	plot_structures(x_grid, y_grid, regions, title = "Anatomical Structures", one_idx = True, **struct_kw)
	# plot_structures(x_grid, y_grid, regions, one_idx = True, filename = figpath + "ex_cardioid5_structures.png", **struct_kw)

	# Problem data.
	K = np.unique(regions).size   # Number of structures.
	A, angles, offs_vec = line_integral_mat(regions, angles = n_angle, n_bundle = n_bundle, offset = offset)
	A = A/n_grid
	A_list = T*[A]

	alpha = np.array(T * [[0.01, 0.50, 0.25, 0.15, 0.005]])
	beta = np.array(T * [[0.001, 0.05, 0.025, 0.015, 0.0005]])
	gamma = np.array(T * [[0.05, 0, 0, 0, 0]])
	h_init = np.array([1] + (K-1)*[0])
	is_target = np.array([True] + (K - 1) * [False])

	# Actual health status transition function.
	mu = 0
	sigma = 0.25
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

	# Health violation.
	def health_viol(h, bounds, is_target):
		viol_oar = np.maximum(bounds["lower"][:,~is_target] - h[:,~is_target], 0)
		viol_ptv = np.maximum(h[:,is_target] - bounds["upper"][:,is_target], 0)
		return np.sum(viol_oar + viol_ptv)/T

	# Dose violation.
	def dose_viol(d, bounds):
		viol_lo = np.maximum(bounds["lower"] - d, 0)
		viol_hi = np.maximum(d - bounds["upper"], 0)
		return np.sum(viol_lo + viol_hi)/T

	# Health prognosis.
	prop_cycle = plt.rcParams['axes.prop_cycle']
	colors = prop_cycle.by_key()['color']
	h_prog = health_prog_act(h_init, T, gamma = gamma, is_target = is_target)
	h_curves = [{"h": h_prog, "label": "Untreated", "kwargs": {"color": colors[1]}}]

	# Penalty functions.
	w_lo = np.array([0] + (K-1)*[1])
	w_hi = np.array([1] + (K-1)*[0])
	rx_health_weights = [w_lo, w_hi]
	rx_health_goal = np.zeros((T,K))
	# rx_dose_weights = np.array([0.01, 1, 1, 1, 0.001])
	rx_dose_weights = np.array([1, 1, 1, 1, 0.25])
	rx_dose_goal = np.zeros((T,K))
	patient_rx = {"is_target": is_target, "dose_goal": rx_dose_goal, "dose_weights": rx_dose_weights,
				  "health_goal": rx_health_goal, "health_weights": rx_health_weights}

	# Beam constraints.
	beam_upper = np.full((T,n), 1.0)
	patient_rx["beam_constrs"] = {"upper": beam_upper}

	# Dose constraints.
	dose_lower = np.zeros((T,K))
	dose_upper = np.full((T,K), 20)   # Upper bound on doses.
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

	# TODO: Check lower bound for PTV always -Inf and upper bound for OAR always Inf.
	patient_rx["health_constrs"] = {"lower": health_lower, "upper": health_upper}
	# patient_rx["health_constrs"] = {"lower": health_lower[:,~is_target], "upper": health_upper[:,is_target]}

	# Dynamic treatment.
	res_dynamic = dyn_quad_treat(A_list, alpha, beta, gamma, h_init, patient_rx, health_map = health_map, use_slack = True,
								 slack_weight = 1e4, max_iter = 15, solver = "MOSEK", ccp_verbose = True)
	# res_dynamic = dyn_quad_treat_admm(A_list, alpha, beta, gamma, h_init, patient_rx, health_map = health_map,
	#							use_slack = True, slack_weight = 1e4, ccp_max_iter = 15, solver = "MOSEK", rho = 10,
	#							admm_max_iter = 500, admm_verbose = True)
	print("Dynamic Treatment")
	print("Status:", res_dynamic["status"])
	print("Objective:", res_dynamic["obj"])
	print("Solve Time:", res_dynamic["solve_time"])
	print("Iterations:", res_dynamic["num_iters"])

	# Calculate actual health/dose constraint violation.
	h_viol_dyn = health_viol(res_dynamic["health"][1:], patient_rx["health_constrs"], is_target)
	d_viol_dyn = dose_viol(res_dynamic["doses"], patient_rx["dose_constrs"])
	print("Average Health Violation:", h_viol_dyn)
	print("Average Dose Violation:", d_viol_dyn)

	# Compute actual doses (including random error).
	dyn_doses_act = np.maximum(res_dynamic["doses"] + d_noise, 0)

	# Plot dynamic beam, health, and treatment curves.
	plot_beams(res_dynamic["beams"], angles = angles, offsets = offs_vec, n_grid = n_grid, stepsize = 1, cmap = transp_cmap(plt.cm.Reds, upper = 0.5),
				title = "Beam Intensities vs. Time", one_idx = True, structures = (x_grid, y_grid, regions), struct_kw = struct_kw)
	plot_health(res_dynamic["health"], curves = h_curves, stepsize = 10, bounds = (health_lower, health_upper), title = "Health Status vs. Time", one_idx = True)
	plot_treatment(dyn_doses_act, stepsize = 10, bounds = (dose_lower, dose_upper), title = "Treatment Dose vs. Time", one_idx = True)

	# plot_beams(res_dynamic["beams"], angles = angles, offsets = offs_vec, n_grid = n_grid, stepsize = 1, cmap = transp_cmap(plt.cm.Reds, upper = 0.5),
	#			one_idx = True, structures = (x_grid, y_grid, regions), struct_kw = struct_kw, filename = figpath + "ex_quadratic_noisy2-admm-beams.png")
	# plot_health(res_dynamic["health"], curves = h_curves, stepsize = 10, bounds = (health_lower, health_upper), one_idx = True, 
	#			filename = figpath + "ex_quadratic_noisy2-admm-health.png")
	# plot_treatment(dyn_doses_act, stepsize = 10, bounds = (dose_lower, dose_upper), one_idx = True, filename = figpath + "ex_quadratic_noisy2-admm-doses.png")

	# Dynamic treatment with MPC.
	print("\nStarting MPC algorithm...")
	res_mpc = mpc_quad_treat(A_list, alpha, beta, gamma, h_init, patient_rx, health_map = health_map, use_ccp_slack = True,
							 ccp_slack_weight = 1e4, use_mpc_slack = True, mpc_slack_weights = 1e4, max_iter = 100, # max_iter = 15,
							 solver = "MOSEK", mpc_verbose = True)
	# res_mpc = mpc_quad_treat_admm(A_list, alpha, beta, gamma, h_init, patient_rx, health_map = health_map, use_ccp_slack = True,
	#							ccp_slack_weight = 1e4, ccp_max_iter = 15, use_mpc_slack = True, mpc_slack_weights = 1e4,
	#							solver = "MOSEK", rho = 10, admm_max_iter = 500, mpc_verbose = True)
	# res_mpc = mpc_quad_treat(A_list, alpha, beta, gamma, h_init, patient_rx, health_map = health_map, use_ccp_slack = True,
	#						 ccp_slack_weight = 1e4, max_iter = 15, solver = "MOSEK", mpc_verbose = True)
	# res_mpc = mpc_quad_treat_admm(A_list, alpha, beta, gamma, h_init, patient_rx, health_map = health_map, use_ccp_slack = True,
	#							  ccp_slack_weight = 1e4, ccp_max_iter = 15, solver = "MOSEK", rho = 5, admm_max_iter = 50, mpc_verbose = True)
	print("\nMPC Treatment")
	print("Status:", res_mpc["status"])
	print("Objective:", res_mpc["obj"])
	print("Solve Time:", res_mpc["solve_time"])
	print("Iterations:", res_mpc["num_iters"])

	# Calculate actual health constraint violation.
	h_viol_mpc = health_viol(res_mpc["health"][1:], patient_rx["health_constrs"], is_target)
	d_viol_mpc = dose_viol(res_mpc["doses"], patient_rx["dose_constrs"])
	print("Average Health Violation:", h_viol_mpc)
	print("Average Dose Violation:", d_viol_mpc)

	# Compute actual doses (including random error).
	mpc_doses_act = np.maximum(res_mpc["doses"] + d_noise, 0)

	# Compare one-shot dynamic and MPC treatment results.
	d_curves = [{"d": dyn_doses_act, "label": "Naive", "kwargs": {"color": colors[0]}}]
	h_naive = [{"h": res_dynamic["health"], "label": "Treated (Naive)", "kwargs": {"color": colors[0]}}]
	h_curves = h_naive + h_curves

	# Plot dynamic MPC beam, health, and treatment curves.
	plot_beams(res_mpc["beams"], angles = angles, offsets = offs_vec, n_grid = n_grid, stepsize = 1,
			   cmap = transp_cmap(plt.cm.Reds, upper = 0.5), title = "Beam Intensities vs. Time", one_idx = True,
			   structures = (x_grid, y_grid, regions), struct_kw = struct_kw)
	plot_health(res_mpc["health"], curves = h_curves, stepsize = 10, bounds = (health_lower, health_upper),
				title = "Health Status vs. Time", label = "Treated (MPC)", color = colors[2], one_idx = True)
	plot_treatment(mpc_doses_act, curves = d_curves, stepsize = 10, bounds = (dose_lower, dose_upper),
				title = "Treatment Dose vs. Time", label = "MPC", color = colors[2], one_idx = True)

	# plot_beams(res_mpc["beams"], angles = angles, offsets = offs_vec, n_grid = n_grid, stepsize = 1,
 	#			cmap = transp_cmap(plt.cm.Reds, upper = 0.5), one_idx = True, structures = (x_grid, y_grid, regions),
	#			struct_kw = struct_kw, filename = figpath + "ex_quadratic_noisy2_mpc-admm-beams.png")
	# plot_health(res_mpc["health"], curves = h_curves, stepsize = 10, bounds = (health_lower, health_upper),
	#			label = "Treated (MPC)", color = colors[2], one_idx = True, filename = figpath + "ex_quadratic_noisy2_mpc-admm-health.png")
	# plot_treatment(mpc_doses_act, curves = d_curves, stepsize = 10, bounds = (dose_lower, dose_upper),
	#			label = "MPC", color = colors[2], one_idx = True, filename = figpath + "ex_quadratic_noisy2_mpc-admm-doses.png")

if __name__ == '__main__':
	main(figpath = "C:/Users/Anqi/Documents/Software/adarad/examples/output/figures/",
		 datapath = "C:/Users/Anqi/Documents/Software/adarad/examples/data/")