import matplotlib
matplotlib.use("TKAgg")

from adarad.optimization.seq_cvx.quad_funcs import dyn_quad_treat, mpc_quad_treat
from adarad.visualization.plot_funcs import *
from adarad.utilities.beam_utils import line_integral_mat
from adarad.medicine.prognosis import health_prog_act

from examples.utilities.simple_utils import simple_structures, simple_colormap

def main(figpath = "", datapath = ""):
	np.random.seed(1)

	T = 20           # Length of optimization.
	n_grid = 1000
	offset = 5       # Displacement between beams (pixels).
	n_angle = 20     # Number of angles.
	n_bundle = 50    # Number of beams per angle.
	n = n_angle*n_bundle   # Total number of beams.

	# Display structures.
	x_grid, y_grid, regions = simple_structures(n_grid, n_grid)
	struct_kw = simple_colormap(one_idx = True)
	# plot_structures(x_grid, y_grid, regions, title = "Anatomical Structures", one_idx = True, **struct_kw)
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
	sigma = 0.1
	h_noise = mu + sigma*np.random.randn(T,K)
	# health_map = lambda h,t: h
	# health_map = lambda h,t: h + h_noise[t]

	# PTV: h_t >= 0.
	def health_map_ptv(h, d, t):
		h_jitter = h + h_noise[t,is_target]
		return np.maximum(h_jitter, 0)
	# OAR: h_t <= 0.
	def health_map_oar(h, d, t):
		h_jitter = h + h_noise[t,~is_target]
		return np.minimum(h_jitter, 0)
	health_map = {"target": health_map_ptv, "organ": health_map_oar}

	# Health violation.
	def health_viol(h, bounds, is_target):
		viol_oar = np.maximum(bounds["lower"][:,~is_target] - h[:,~is_target], 0)
		viol_ptv = np.maximum(h[:,is_target] - bounds["upper"][:,is_target], 0)
		return np.sum(viol_oar + viol_ptv)/T

	# Health prognosis.
	prop_cycle = plt.rcParams['axes.prop_cycle']
	colors = prop_cycle.by_key()['color']
	h_prog = health_prog_act(h_init, T, gamma = gamma, is_target = is_target, health_map = health_map)
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

	# Dynamic optimization.
	res_dynamic = dyn_quad_treat(A_list, alpha, beta, gamma, h_init, patient_rx, health_map = health_map, use_slack = True,
								 slack_weight = 1e4, max_iter = 15, solver = "MOSEK", ccp_verbose = True, auto_init = False,
								 full_hist = True)
	print("Dynamic Treatment")
	print("Status:", res_dynamic["status"])
	print("Objective:", res_dynamic["obj"])
	print("Solve Time:", res_dynamic["solve_time"])
	print("Iterations:", res_dynamic["num_iters"])

	# Dynamic optimization with MPC.
	print("\nStarting MPC algorithm...")
	res_mpc = mpc_quad_treat(A_list, alpha, beta, gamma, h_init, patient_rx, health_map = health_map, use_slack= True,
							 slack_weight= 1e4, use_mpc_slack = True, mpc_slack_weights = 1e4, max_iter = 100,
							 solver = "MOSEK", mpc_verbose = True, auto_init = False, full_hist = True)
	print("\nMPC Treatment")
	print("Status:", res_mpc["status"])
	print("Objective:", res_mpc["obj"])
	print("Solve Time:", res_mpc["solve_time"])
	print("Iterations:", res_mpc["num_iters"])

	# Calculate actual health constraint violation.
	h_viol_mpc = health_viol(res_mpc["health"][1:], patient_rx["health_constrs"], is_target)
	print("Average Health Violation:", h_viol_mpc)

	# Compare one-shot dynamic and MPC optimization results.
	d_curves = [{"d": res_dynamic["doses"], "label": "Naive", "kwargs": {"color": colors[0]}}]
	h_naive = [{"h": res_dynamic["health"], "label": "Treated (Naive)", "kwargs": {"color": colors[0]}}]
	h_curves = h_naive + h_curves

	# Plot dynamic MPC beam, health, and optimization curves.
	# plot_beams(res_mpc["beams"], angles = angles, offsets = offs_vec, n_grid = n_grid, stepsize = 1,
	#		   cmap = transp_cmap(plt.cm.Reds, upper = 0.5), title = "Beam Intensities vs. Time", one_idx = True,
	#		   structures = (x_grid, y_grid, regions), struct_kw = struct_kw)

	print("Plotting results for all iterations")
	for i in range(T):
		# plot_health(res_mpc["health_hist"][i], curves = h_curves, stepsize = 10, bounds = (health_lower, health_upper),
		#		label = "Treated (MPC)", color = colors[2], one_idx = True)
		# plot_treatment(res_mpc["doses_hist"][i], curves = d_curves, stepsize = 10, bounds = (dose_lower, dose_upper),
		#		label = "MPC", color = colors[2], one_idx = True)

		# plot_health(res_mpc["health_hist"][i], curves = h_curves, stepsize = 10, bounds = (health_lower, health_upper),
		#			label = "Treated (MPC)", color = colors[2], one_idx = True, filename = figpath + "ex2_mpc_health_iter_{0}.png".format(i+1))
		# plot_treatment(res_mpc["doses_hist"][i], curves = d_curves, stepsize = 10, bounds = (dose_lower, dose_upper),
		#			label = "MPC", color = colors[2], one_idx = True, filename = figpath + "ex2_mpc_doses_iter_{0}.png".format(i+1))

		h_dict = {"v": res_mpc["health_hist"][i], "varname": "h", "curves": h_curves, "stepsize": 10, 
				  "bounds": (health_lower, health_upper), "label": "Treated (MPC)", "color": colors[2], "one_idx": True, "one_shift": False}
		d_dict = {"v": res_mpc["doses_hist"][i], "varname": "d", "curves": d_curves, "stepsize": 10, 
				  "bounds": (dose_lower, dose_upper), "label": "MPC", "color": colors[2], "one_idx": True, "one_shift": True}
		# plot_stacked([h_dict, d_dict], title = "Health Status and Treatment Dose vs. Time", figsize = (16,10))
		plot_stacked([h_dict, d_dict], figsize = (16,10), show = False, filename = figpath + "ex2_mpc_health_doses_iter_{0}.png".format(i+1))

if __name__ == '__main__':
	main(figpath = "C:/Users/Anqi/Documents/Software/adarad/examples/output/figures/movie/",
		 datapath = "C:/Users/Anqi/Documents/Software/adarad/examples/data/")