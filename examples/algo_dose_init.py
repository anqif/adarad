import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TKAgg")

import cvxpy
from cvxpy import *
from cvxpy.settings import SOLUTION_PRESENT

from adarad.init_funcs import *
from adarad.quadratic.dyn_quad_prob import build_dyn_quad_prob
from adarad.utilities.plot_utils import *
from adarad.utilities.data_utils import line_integral_mat, health_prog_act

from example_utils import simple_structures, simple_colormap

# output_path = "C:/Users/Anqi/Documents/Software/adarad/examples/output/"
output_path = "/home/anqi/Documents/software/adarad/examples/output/"
output_prefix = output_path + "ex1_simple_"
init_prefix = output_prefix + "init_"
final_prefix = output_prefix + "ccp_"

def form_step_xy(x, y, buf = 0, shift = 0):
	x_shift = x - shift
	x_buf = np.zeros(x_shift.shape[0] + 2)
	x_buf[0] = x_shift[0] - buf
	x_buf[-1] = x_shift[-1] + buf
	x_buf[1:-1] = x_shift

	y_buf = np.zeros(y.shape[0] + 2)
	y_buf[0] = y[0]
	y_buf[-1] = y[-1]
	y_buf[1:-1] = y

	return x_buf, y_buf

def main():
	# Problem data.
	T = 20           # Length of treatment.
	n_grid = 1000
	offset = 5       # Displacement between beams (pixels).
	n_angle = 20     # Number of angles.
	n_bundle = 50    # Number of beams per angle.
	n = n_angle*n_bundle   # Total number of beams.
	t_s = 0   # Static session.

	# Anatomical structures.
	x_grid, y_grid, regions = simple_structures(n_grid, n_grid)
	struct_kw = simple_colormap(one_idx = True)
	K = np.unique(regions).size   # Number of structures.

	A, angles, offs_vec = line_integral_mat(regions, angles = n_angle, n_bundle = n_bundle, offset = offset)
	A = A/n_grid
	A_list = T*[A]

	alpha = np.array(T*[[0.01, 0.50, 0.25, 0.15, 0.005]])
	beta = np.array(T*[[0.001, 0.05, 0.025, 0.015, 0.0005]])
	gamma = np.array(T*[[0.05, 0, 0, 0, 0]])
	h_init = np.array([1] + (K-1)*[0])

	is_target = np.array([True] + (K-1)*[False])
	num_ptv = np.sum(is_target)
	num_oar = K - num_ptv

	# Health prognosis.
	prop_cycle = plt.rcParams['axes.prop_cycle']
	colors = prop_cycle.by_key()['color']
	h_prog = health_prog_act(h_init, T, gamma = gamma)
	h_curves = [{"h": h_prog, "label": "Untreated", "kwargs": {"color": colors[1]}}]

	# Beam constraints.
	beam_upper = np.full((T,n), 1.0)

	# Dose constraints.
	dose_lower = np.zeros((T,K))
	dose_upper = np.full((T,K), 20)

	# Health constraints.
	health_lower = np.full((T,K), -np.inf)
	health_upper = np.full((T,K), np.inf)
	health_lower[:,1] = -1.0     # Lower bound on OARs.
	health_lower[:,2] = -2.0
	health_lower[:,3] = -2.0
	health_lower[:,4] = -3.0
	health_upper[:15,0] = 2.0    # Upper bound on PTV for t = 1,...,15.
	health_upper[15:,0] = 0.05   # Upper bound on PTV for t = 16,...,20.

	patient_rx = {"is_target": is_target,
				  "dose_goal": np.zeros((T,K)),
				  "dose_weights": np.array((K-1)*[1] + [0.25]),
				  "health_goal": np.zeros((T,K)),
				  "health_weights": [np.array([0] + (K-1)*[0.25]), np.array([1] + (K-1)*[0])],
				  "beam_constrs": {"upper": beam_upper},
				  "dose_constrs": {"lower": dose_lower, "upper": dose_upper},
				  "health_constrs": {"lower": health_lower, "upper": health_upper}}

	# Stage 1: Static beam problem.
	# Define variables.
	b = Variable((n,), nonneg=True)
	d = A @ b

	h_lin = h_init - multiply(alpha[t_s], d) + gamma[t_s]
	h_quad = h_init - multiply(alpha[t_s], d) - multiply(beta[t_s], square(d)) + gamma[t_s]
	h_ptv = h_lin[is_target]
	h_oar = h_quad[~is_target]
	h = multiply(h_lin, is_target) + multiply(h_quad, ~is_target)

	# Form objective.
	d_penalty = sum_squares(d[:-1]) + 0.25*square(d[-1])
	h_penalty_ptv = sum(pos(h_ptv))
	h_penalty_oar = 0.25*sum(neg(h_oar))
	h_penalty = h_penalty_ptv + h_penalty_oar

	# Add slack to health bounds.
	# h_hi_slack_weight = 1e4
	# h_hi_slack = Variable(nonneg=True)
	# s_hi_penalty = h_hi_slack_weight*h_hi_slack

	h_lo_slack_weight = 0.25
	h_lo_slack = Variable(h_oar.shape, nonneg=True)
	s_lo_penalty = h_lo_slack_weight*sum(h_lo_slack)

	# s_penalty = s_hi_penalty + s_lo_penalty
	s_penalty = s_lo_penalty
	obj = d_penalty + h_penalty + s_penalty

	# Additional constraints.
	# constrs = [b <= beam_upper[t_s,:], d <= dose_upper[t_s,:], d >= dose_lower[t_s,:], h_ptv <= health_upper[t_s,0], h_oar >= health_lower[t_s,1:]]
	# constrs = [h_ptv <= health_upper[-1,0] + h_hi_slack, h_oar >= health_lower[-1,1:] - h_lo_slack]
	# constrs = [h_ptv <= health_upper[-1,0], h_oar >= health_lower[-1,1:] - h_lo_slack]
	constrs = [b <= np.sum(beam_upper, axis=0), h_ptv <= health_upper[-1,0], h_oar >= health_lower[-1,1:] - h_lo_slack]

	# Solve problem.
	print("Stage 1: Solving problem...")
	prob_1 = Problem(Minimize(obj), constrs)
	prob_1.solve(solver = "MOSEK")
	if prob_1.status not in SOLUTION_PRESENT:
		raise RuntimeError("Stage 1: Solver failed with status {0}".format(prob_1.status))
	solve_time = prob_1.solver_stats.solve_time

	# Save results.
	b_static = b.value   # Save optimal static beams for stage 2.
	d_static = np.vstack([A_list[t] @ b_static for t in range(T)])
	d_stage_1 = d.value
	# h_stage_1 = h.value
	h_stage_1 = h_init - alpha[t_s]*d_stage_1 - beta[t_s]*d_stage_1**2 + gamma[t_s]

	print("Stage 1 Results")
	print("Objective:", prob_1.value)
	print("Optimal Beam (Max):", np.max(b_static))
	print("Optimal Dose:", d_stage_1)
	print("Optimal Health:", h_stage_1)
	print("Solve Time:", prob_1.solver_stats.solve_time)

	# Compare with AdaRad package.
	# prob_1_ada, b_1_ada, h_1_ada, d_1_ada, h_actual_1_ada, h_slack_1_ada = \
	# 	build_stat_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx, t_static = 0, slack_oar_weight = h_lo_slack_weight)
	# prob_1_ada.solve(solver = "MOSEK")
	# if prob_1_ada.status not in SOLUTION_PRESENT:
	# 	raise RuntimeError("AdaRad Stage 1: Solver failed with status {0}".format(prob_1_ada.status))
	#
	# print("Compare with AdaRad")
	# print("Difference in Objective:", np.abs(prob_1.value - prob_1_ada.value))
	# print("Normed Difference in Beam:", np.linalg.norm(b_static - b_1_ada.value))
	# print("Normed Difference in Dose:", np.linalg.norm(d_stage_1 - d_1_ada.value))
	# print("Normed Difference in Health:", np.linalg.norm(h.value - h_1_ada.value))
	# print("Normed Difference in Health Slack:", np.linalg.norm(h_lo_slack.value - h_slack_1_ada[1:].value))
	# print("AdaRad Solve Time:", prob_1_ada.solver_stats.solve_time)

	# Plot optimal dose and health per structure.
	xlim_eps = 0.5
	plt.bar(range(K), d_stage_1, width = 0.8)
	plt.step(*form_step_xy(np.arange(K), dose_lower[-1,:], buf = 0.5), where = "mid", lw = 1, ls = "--", color = colors[1])
	plt.step(*form_step_xy(np.arange(K), dose_upper[-1,:], buf = 0.5), where = "mid", lw = 1, ls = "--", color = colors[1])
	plt.title("Treatment Dose vs. Structure")
	plt.xlim(-xlim_eps, K-1+xlim_eps)
	plt.show()

	health_bounds_fin = np.concatenate(([health_upper[-1,0]], health_lower[-1,1:]))
	plt.bar(range(K), h_stage_1, width=0.8)
	plt.step(*form_step_xy(np.arange(K), health_bounds_fin, buf = 0.5), where = "mid", lw = 1, ls = "--", color = colors[1])
	plt.title("Health Status vs. Structure")
	plt.xlim(-xlim_eps, K-1+xlim_eps)
	plt.show()

	# raise RuntimeError("Stop 0")

	# Stage 2a: Dynamic scaling problem with constant factor.
	u = Variable(nonneg=True)
	b = u*b_static
	# d = vstack([A_list[t] @ b for t in range(T)])
	d = u*d_static
	h = Variable((T+1,K))

	# Used in Taylor expansion of PTV health dynamics.
	h_tayl_slack_weight = 1e4
	h_tayl_slack = Variable((T,), nonneg=True)      # Slack in approximation.
	# h_tayl_slack = Parameter((T,), value=np.zeros(T))

	# Form objective.
	# d_penalty = sum_squares(d[:,:-1]) + 0.25*sum_squares(d[:,-1])
	d_penalty = square(u)*(sum_squares(d_static[:,:-1]) + 0.25*sum_squares(d_static[:,-1])).value
	h_penalty = sum(pos(h[1:,0])) + 0.25*sum(neg(h[1:,1:]))
	s_tayl_penalty = h_tayl_slack_weight*sum(h_tayl_slack)

	# Add slack to lower health bounds.
	# TODO: Should we continue to add slack to lower health bound on OARs?
	h_lo_slack_weight = 0.25
	h_lo_slack = Variable((T,num_oar), nonneg=True)
	s_lo_penalty = h_lo_slack_weight*sum(h_lo_slack)

	s_penalty = s_tayl_penalty + s_lo_penalty
	obj = d_penalty + h_penalty + s_penalty

	# Health dynamics.
	constrs = [h[0] == h_init]
	for t in range(T):
		# For PTV, use simple linear model (beta_t = 0).
		# constrs += [h[t+1,0] == h[t,0] - alpha[t,0]*d[t,0] + gamma[t,0] - h_tayl_slack[t]]
		constrs += [h[t+1,0] == h[t,0] - alpha[t,0]*u*d_static[t,0] + gamma[t,0] - h_tayl_slack[t]]

		# For OAR, use linear-quadratic model with lossless relaxation.
		# constrs += [h[t+1,1:] <= h[t,1:] - multiply(alpha[t,1:], d[t,1:]) - multiply(beta[t,1:], square(d[t,1:])) + gamma[t,1:]]
		constrs += [h[t+1,1:] <= h[t,1:] - u*multiply(alpha[t,1:], d_static[t,1:]).value - square(u)*multiply(beta[t,1:], square(d_static[t,1:])).value + gamma[t,1:]]

	# Additional constraints.
	# constrs += [b <= np.min(beam_upper, axis=0), d <= dose_upper, d >= dose_lower, h[1:,0] <= health_upper[:,0], h[1:,1:] >= health_lower[:,1:]]
	# constrs += [b <= np.min(beam_upper, axis=0), d <= dose_upper, d >= dose_lower, h[1:,0] <= health_upper[:,0], h[1:,1:] >= health_lower[:,1:] - h_lo_slack]
	constrs += [b <= np.min(beam_upper, axis=0), u*d_static <= dose_upper, u*d_static >= dose_lower, h[1:,0] <= health_upper[:,0], h[1:,1:] >= health_lower[:,1:] - h_lo_slack]
	# constrs += [d <= dose_upper, d >= dose_lower, h[1:,0] <= health_upper[:,0], h[1:,1:] >= health_lower[:,1:] - h_lo_slack]

	# Warm start.
	u.value = 1

	# Solve problem.
	print("Stage 2: Solving initial problem...")
	prob_2a = Problem(Minimize(obj), constrs)
	prob_2a.solve(solver = "MOSEK", warm_start = True)
	if prob_2a.status not in SOLUTION_PRESENT:
		raise RuntimeError("Stage 2 Initialization: Solver failed with status {0}".format(prob_2a.status))
	solve_time += prob_2a.solver_stats.solve_time

	# Save results.
	u_stage_2_init = u.value
	d_stage_2_init = d.value   # Save optimal doses derived from constant factor for stage 2b.
	# h_stage_2_init = h.value
	h_stage_2_init = health_prog_act(h_init, T, alpha, beta, gamma, d_stage_2_init, is_target)
	s_stage_2_init = h_tayl_slack.value

	print("Stage 2 Initialization")
	print("Objective:", prob_2a.value)
	print("Optimal Beam Weight:", u_stage_2_init)
	print("Optimal Beam (Max):", np.max(b.value))
	# print("Optimal Dose:", d_stage_2_init)
	# print("Optimal Health:", h_stage_2_init)
	# print("Optimal Health Slack:", s_stage_2_init)
	print("Solve Time:", prob_2a.solver_stats.solve_time)

	# Compare with AdaRad package.
	# prob_2a_ada, u_2a_ada, b_2a_ada, h_2a_ada, d_2a_ada, h_lin_dyn_slack_2a_ada, h_lin_bnd_slack_2a_ada = \
	# 	build_scale_lin_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx, b_1_ada.value, use_dyn_slack = True,
	# 		slack_dyn_weight = h_tayl_slack_weight, use_bnd_slack = True, slack_bnd_weight = h_lo_slack_weight)
	# prob_2a_ada.solve(solver="MOSEK")
	# if prob_2a_ada.status not in SOLUTION_PRESENT:
	# 	raise RuntimeError("AdaRad Stage 2a: Solver failed with status {0}".format(prob_2a_ada.status))
	#
	# print("Compare with AdaRad")
	# print("Difference in Objective:", np.abs(prob_2a.value - prob_2a_ada.value))
	# print("Normed Difference in Beam:", np.linalg.norm(u_stage_2_init*b_static - b_2a_ada.value))
	# print("Normed Difference in Dose:", np.linalg.norm(d_stage_2_init - d_2a_ada.value))
	# print("Normed Difference in Health:", np.linalg.norm(h.value - h_2a_ada.value))
	# print("Normed Difference in Health Slack (Dynamics):", np.linalg.norm(s_stage_2_init - h_lin_dyn_slack_2a_ada[:,0].value))
	# print("Normed Difference in Health Slack (Bound):", np.linalg.norm(h_lo_slack.value - h_lin_bnd_slack_2a_ada[:,1:].value))
	# print("AdaRad Solve Time:", prob_2a_ada.solver_stats.solve_time)

	# Plot optimal dose and health over time.
	plot_treatment(d_stage_2_init, stepsize = 10, bounds = (dose_lower, dose_upper), title="Treatment Dose vs. Time",
				   color = colors[0], one_idx = True)
	plot_health(h_stage_2_init, curves = h_curves, stepsize = 10, bounds = (health_lower, health_upper),
				title = "Health Status vs. Time", label = "Treated", color = colors[0], one_idx = True)

	# raise RuntimeError("Stop 1")

	# Stage 2b: Dynamic scaling problem with time-varying factors.
	# Define variables.
	u = Variable((T,), nonneg=True)
	b = vstack([u[t]*b_static for t in range(T)])
	# d = vstack([A_list[t] @ b[t] for t in range(T)])
	d = vstack([u[t]*d_static[t,:] for t in range(T)])
	h = Variable((T+1,K))

	# Used in Taylor expansion of PTV health dynamics.
	h_tayl_slack_weight = 1e4
	h_tayl_slack = Variable((T,), nonneg=True)      # Slack in approximation.
	# d_parm = Parameter(d.shape, nonneg=True)   # Dose point around which to linearize.
	d_parm = Parameter((T,), nonneg=True)

	# Form objective.
	d_penalty = sum_squares(d[:,:-1]) + 0.25*sum_squares(d[:,-1])
	h_penalty = sum(pos(h[1:,0])) + 0.25*sum(neg(h[1:,1:]))
	s_tayl_penalty = h_tayl_slack_weight*sum(h_tayl_slack)

	# Add slack to lower health bounds.
	# TODO: Should we continue to add slack to lower health bound on OARs?
	h_lo_slack_weight = 0.25
	h_lo_slack = Variable((T,num_oar), nonneg=True)
	s_lo_penalty = h_lo_slack_weight*sum(h_lo_slack)

	s_penalty = s_tayl_penalty + s_lo_penalty
	obj = d_penalty + h_penalty + s_penalty

	# Health dynamics.
	constrs = [h[0] == h_init]
	for t in range(T):
		# For PTV, use first-order Taylor expansion of dose around d_parm.
		# constrs += [h[t+1,0] == h[t,0] - alpha[t,0]*d[t,0] - (2*d[t,0] - d_parm[t,0])*beta[t,0]*d_parm[t,0] + gamma[t,0] - h_tayl_slack[t]]
		# constrs += [h[t+1,0] == h[t,0] - alpha[t,0]*d[t,0] - (2*d[t,0] - d_parm[t])*beta[t,0]*d_parm[t] + gamma[t,0] - h_tayl_slack[t]]
		constrs += [h[t+1,0] == h[t,0] - alpha[t,0]*u[t]*d_static[t,0] - (2*u[t]*d_static[t,0] - d_parm[t])*beta[t,0]*d_parm[t] + gamma[t,0] - h_tayl_slack[t]]

		# For OAR, use linear-quadratic model with lossless relaxation.
		# constrs += [h[t+1,1:] <= h[t,1:] - multiply(alpha[t,1:], d[t,1:]) - multiply(beta[t,1:], square(d[t,1:])) + gamma[t,1:]]
		constrs += [h[t+1,1:] <= h[t,1:] - u[t]*multiply(alpha[t,1:], d_static[t,1:]).value - square(u[t])*multiply(beta[t,1:], square(d_static[t,1:])).value + gamma[t,1:]]
		# alpha_d_static = multiply(alpha[t,1:], d_static[t,1:]).value
		# beta_d_static_sq = multiply(beta[t,1:], square(d_static[t,1:])).value
		# constrs += [h[t+1,1:] <= h[t,1:] - u[t]*alpha_d_static - square(u[t])*beta_d_static_sq + gamma[t,1:]]

	# Additional constraints.
	# constrs += [b <= beam_upper, d <= dose_upper, d >= dose_lower, h[1:,0] <= health_upper[:,0], h[1:,1:] >= health_lower[:,1:]]
	constrs += [b <= beam_upper, d <= dose_upper, d >= dose_lower, h[1:,0] <= health_upper[:,0], h[1:,1:] >= health_lower[:,1:] - h_lo_slack]
	prob_2b = Problem(Minimize(obj), constrs)

	# Solve using CCP.
	print("Stage 2: Solving dynamic problem with CCP...")
	ccp_max_iter = 20
	ccp_eps = 1e-3
	ccp_2b_solve_time = 0

	# Warm start.
	u.value = np.array(T*[u_stage_2_init])
	h.value = h_stage_2_init
	h_tayl_slack.value = s_stage_2_init

	obj_old = np.inf
	# d_parm.value = d_stage_2_init
	d_parm.value = d_stage_2_init[:,0]
	for k in range(ccp_max_iter):
		# Solve linearized problem.
		prob_2b.solve(solver = "MOSEK", warm_start = True)
		if prob_2b.status not in SOLUTION_PRESENT:
			raise RuntimeError("Stage 2 CCP: Solver failed on iteration {0} with status {1}".format(k, prob_2b.status))
		ccp_2b_solve_time += prob_2b.solver_stats.solve_time

		# Terminate if change in objective is small.
		obj_diff = obj_old - prob_2b.value
		print("CCP Iteration {0}, Objective Difference: {1}".format(k, obj_diff))
		if obj_diff <= ccp_eps:
			break

		obj_old = prob_2b.value
		# d_parm.value = d.value
		d_parm.value = d.value[:,0]
	solve_time += ccp_2b_solve_time

	# Save results.
	u_stage_2 = u.value
	b_stage_2 = b.value
	d_stage_2 = d.value
	# h_stage_2 = h.value
	h_stage_2 = health_prog_act(h_init, T, alpha, beta, gamma, d_stage_2, is_target)
	s_stage_2 = h_tayl_slack.value

	print("Stage 2 Results")
	print("Objective:", prob_2b.value)
	# print("Optimal Beam Weight:", u_stage_2)
	print("Optimal Beam Weight (Median):", np.median(u_stage_2))
	# print("Optimal Dose:", d_stage_2)
	# print("Optimal Health:", h_stage_2)
	# print("Optimal Health Slack:", s_stage_2)
	print("Solve Time:", ccp_2b_solve_time)

	# Compare with AdaRad package.
	# prob_2b_ada, u_2b_ada, b_2b_ada, h_2b_ada, d_2b_ada, d_parm_2b_ada, h_dyn_slack_2b_ada, h_bnd_slack_2b_ada = \
	# 	build_scale_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx, b_static, use_dyn_slack = True,
	# 		slack_dyn_weight = h_tayl_slack_weight, use_bnd_slack = True, slack_bnd_weight = h_lo_slack_weight)
	#
	# result_2b_ada = ccp_solve(prob_2b_ada, d_2b_ada, d_parm_2b_ada, d_2a_ada.value, h_dyn_slack_2b_ada, max_iter = ccp_max_iter,
	# 					ccp_eps = ccp_eps, solver = "MOSEK", warm_start = True)
	# if result_2b_ada["status"] not in SOLUTION_PRESENT:
	# 	raise RuntimeError("Stage 2b: CCP solve failed with status {0}".format(result_2b_ada["status"]))
	#
	# print("Compare with AdaRad")
	# print("Difference in Objective:", np.abs(prob_2b.value - prob_2b_ada.value))
	# print("Normed Difference in Beam:", np.linalg.norm(b_stage_2 - b_2b_ada.value))
	# print("Normed Difference in Dose:", np.linalg.norm(d_stage_2 - d_2b_ada.value))
	# print("Normed Difference in Health:", np.linalg.norm(h.value - h_2b_ada.value))
	# print("Normed Difference in Health Slack (Dynamics):", np.linalg.norm(s_stage_2 - h_dyn_slack_2b_ada[:,0].value))
	# print("Normed Difference in Health Slack (Bound):", np.linalg.norm(h_lo_slack.value - h_bnd_slack_2b_ada[:,1:].value))
	# print("AdaRad Solve Time:", result_2b_ada["solve_time"])

	# Save to file.
	np.save(init_prefix + "beams.npy", b_stage_2)
	np.save(init_prefix + "doses.npy", d_stage_2)
	np.save(init_prefix + "health.npy", h_stage_2)
	np.save(init_prefix + "health_slack.npy", s_stage_2)

	# Plot optimal beam, dose, and health over time.
	plot_beams(b_stage_2, angles = angles, offsets = offs_vec, n_grid = n_grid, stepsize = 1, cmap = transp_cmap(plt.cm.Reds, upper = 0.5), 
			   title = "Beam Intensities vs. Time", one_idx = True, structures = (x_grid, y_grid, regions), struct_kw = struct_kw, 
			   filename = init_prefix + "beams.png")
	plot_treatment(d_stage_2, stepsize = 10, bounds = (dose_lower, dose_upper), title = "Treatment Dose vs. Time", 
				color = colors[0], one_idx = True, filename = init_prefix + "doses.png")
	plot_health(h_stage_2, curves = h_curves, stepsize = 10, bounds = (health_lower, health_upper), title = "Health Status vs. Time", label = "Treated", 
				color = colors[0], one_idx = True, filename = init_prefix + "health.png")

	# raise RuntimeError("Stop 2")

	# Main Stage: Dynamic optimal control problem.
	# Define variables.
	b = Variable((T,n), nonneg=True)
	d = vstack([A_list[t] @ b[t] for t in range(T)])
	h = Variable((T+1,K))

	# Used in Taylor expansion of PTV health dynamics.
	h_slack_weight = 1e4
	h_slack = Variable((T,), nonneg=True)      # Slack in approximation.
	# d_parm = Parameter(d.shape, nonneg=True)   # Dose point around which to linearize.
	d_parm = Parameter((T,), nonneg=True)

	# Form objective.
	d_penalty = sum_squares(d[:,:-1]) + 0.25*sum_squares(d[:,-1])
	h_penalty = sum(pos(h[1:,0])) + 0.25*sum(neg(h[1:,1:]))
	s_penalty = h_slack_weight*sum(h_slack)
	obj = d_penalty + h_penalty + s_penalty

	# Health dynamics.
	constrs = [h[0] == h_init]
	for t in range(T):
		# For PTV, use first-order Taylor expansion of dose around d_parm.
		# constrs += [h[t+1,0] == h[t,0] - alpha[t,0]*d[t,0] - (2*d[t,0] - d_parm[t,0])*beta[t,0]*d_parm[t,0] + gamma[t,0] - h_slack[t]]
		constrs += [h[t+1,0] == h[t,0] - alpha[t,0]*d[t,0] - (2*d[t,0] - d_parm[t])*beta[t,0]*d_parm[t] + gamma[t,0] - h_slack[t]]

		# For OAR, use linear-quadratic model with lossless relaxation.
		constrs += [h[t+1,1:] <= h[t,1:] - multiply(alpha[t,1:], d[t,1:]) - multiply(beta[t,1:], square(d[t,1:])) + gamma[t,1:]]

	# Additional constraints.
	constrs += [b <= beam_upper, d <= dose_upper, d >= dose_lower, h[1:,0] <= health_upper[:,0], h[1:,1:] >= health_lower[:,1:]]
	prob_main = Problem(Minimize(obj), constrs)

	# Solve using CCP.
	print("Main Stage: Solving dynamic problem with CCP...")
	ccp_max_iter = 20
	ccp_eps = 1e-3
	ccp_main_solve_time = 0

	# Warm start.
	b.value = b_stage_2
	h.value = h_stage_2
	h_slack.value = s_stage_2

	obj_old = np.inf
	# d_parm.value = d_stage_2   # Initialize using optimal dose from stage 2.
	d_parm.value = d_stage_2[:,0]
	for k in range(ccp_max_iter):
		# Solve linearized problem.
		prob_main.solve(solver = "MOSEK", warm_start = True)
		if prob_main.status not in SOLUTION_PRESENT:
			raise RuntimeError("Main Stage CCP: Solver failed on iteration {0} with status {1}".format(k, prob_main.status))
		ccp_main_solve_time += prob_main.solver_stats.solve_time

		# Terminate if change in objective is small.
		obj_diff = obj_old - prob_main.value
		print("CCP Iteration {0}, Objective Difference: {1}".format(k, obj_diff))
		if obj_diff <= ccp_eps:
			break

		obj_old = prob_main.value
		# d_parm.value = d.value
		d_parm.value = d.value[:,0]
	solve_time += ccp_main_solve_time

	# Save results.
	b_main = b.value
	d_main = d.value
	# h_main = h.value
	h_main = health_prog_act(h_init, T, alpha, beta, gamma, d_main, is_target)
	s_main = h_slack.value

	print("Main Stage Results")
	print("Objective:", prob_main.value)
	# print("Optimal Dose:", d_main)
	# print("Optimal Health:", h_main)
	# print("Optimal Health Slack:", s_main)
	print("Solve Time:", ccp_main_solve_time)
	print("Total Solve Time:", solve_time)

	# Compare with AdaRad package.
	# prob_main_ada, b_main_ada, h_main_ada, d_main_ada, d_parm_main_ada, h_dyn_slack_main_ada = \
	# 	build_dyn_quad_prob(A_list, alpha, beta, gamma, h_init, patient_rx, use_slack = True, slack_weight = h_tayl_slack_weight)
	# result_main_ada = ccp_solve(prob_main_ada, d_main_ada, d_parm_main_ada, d_stage_2, h_dyn_slack_main_ada, max_iter = ccp_max_iter,
	# 							ccp_verbose = True, ccp_eps = ccp_eps, solver = "MOSEK", warm_start = True)
	# if result_main_ada["status"] not in SOLUTION_PRESENT:
	# 	raise RuntimeError("Main Stage: CCP solve failed with status {0}".format(result_main_ada["status"]))
	#
	# print("Compare with AdaRad")
	# print("Difference in Objective:", np.abs(prob_main.value - prob_main_ada.value))
	# print("Normed Difference in Beam:", np.linalg.norm(b_main - b_main_ada.value))
	# print("Normed Difference in Dose:", np.linalg.norm(d_main - d_main_ada.value))
	# print("Normed Difference in Health:", np.linalg.norm(h.value - h_main_ada.value))
	# print("Normed Difference in Health Slack (Dynamics):", np.linalg.norm(s_main - h_dyn_slack_main_ada[:,0].value))
	# print("AdaRad Solve Time:", result_main_ada["solve_time"])

	# Save to file.
	np.save(final_prefix + "beams.npy", b_main)
	np.save(final_prefix + "doses.npy", d_main)
	np.save(final_prefix + "health.npy", h_main)
	np.save(final_prefix + "health_slack.npy", s_main)

	# Plot optimal beam, dose, and health over time.
	plot_beams(b_main, angles = angles, offsets = offs_vec, n_grid = n_grid, stepsize = 1, cmap = transp_cmap(plt.cm.Reds, upper = 0.5), 
			   title = "Beam Intensities vs. Time", one_idx = True, structures = (x_grid, y_grid, regions), struct_kw = struct_kw,
			   filename = final_prefix + "beams.png")
	plot_treatment(d_main, stepsize = 10, bounds = (dose_lower, dose_upper), title = "Treatment Dose vs. Time", one_idx = True, 
			   filename = final_prefix + "doses.png")
	plot_health(h_main, curves = h_curves, stepsize = 10, bounds = (health_lower, health_upper), title = "Health Status vs. Time", 
			  label = "Treated", color = colors[0], one_idx = True, filename = final_prefix + "health.png")

if __name__ == "__main__":
	main()