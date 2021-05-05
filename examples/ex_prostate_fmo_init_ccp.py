import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TKAgg")
from time import time

import cvxpy
from cvxpy import *
from cvxpy.settings import SOLUTION_PRESENT

from adarad.init_funcs import *
from adarad.utilities.plot_utils import *
from adarad.utilities.file_utils import yaml_to_dict
from adarad.utilities.data_utils import health_prog_act

# input_path = "C:/Users/Anqi/Documents/Software/adarad/examples/data/"
# output_path = "C:/Users/Anqi/Documents/Software/adarad/examples/output/"
input_path = "/home/anqi/Documents/software/adarad/examples/data/"
output_path = "/home/anqi/Documents/software/adarad/examples/output/"
fig_path = output_path + "figures/"

# output_prefix = output_path + "ex3_prostate_fmo_"
output_prefix = output_path + "ex3_prostate_fmo_full_"
init_prefix = output_prefix + "init_"
final_prefix = output_prefix + "ccp_"

fig_prefix = fig_path + "ex3_prostate_fmo_full_"
init_fig_prefix = fig_prefix + "init_"
final_fig_prefix = fig_prefix + "ccp_"

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
	patient_bio, patient_rx, visuals = yaml_to_dict(input_path + "ex_prostate_FMO_stanford.yml")

	# Patient data.
	A_list = patient_bio["dose_matrices"]
	alpha = patient_bio["alpha"]
	beta = patient_bio["beta"]
	gamma = patient_bio["gamma"]
	h_init = patient_bio["health_init"]

	# Treatment data.
	t_s = 0   # Static session.
	T = len(A_list)
	K, n = A_list[0].shape

	is_target = patient_rx["is_target"]
	num_ptv = np.sum(is_target)
	num_oar = K - num_ptv

	beam_lower = patient_rx["beam_constrs"]["lower"]
	beam_upper = patient_rx["beam_constrs"]["upper"]
	dose_lower = patient_rx["dose_constrs"]["lower"]
	dose_upper = patient_rx["dose_constrs"]["upper"]
	health_lower = patient_rx["health_constrs"]["lower"]
	health_upper = patient_rx["health_constrs"]["upper"]

	# Health prognosis.
	prop_cycle = plt.rcParams["axes.prop_cycle"]
	colors = prop_cycle.by_key()["color"]
	h_prog = health_prog_act(h_init, T, gamma = gamma)
	h_curves = [{"h": h_prog, "label": "Untreated", "kwargs": {"color": colors[1]}}]

	patient_rx_ada = {"is_target": is_target,
				  	  "dose_goal": np.zeros((T,K)),
				  	  "dose_weights": np.array((K-1)*[1] + [0.25]),
				  	  "health_goal": np.zeros((T,K)),
				  	  "health_weights": [np.array([0] + (K-1)*[num_ptv/num_oar]), np.array([1] + (K-1)*[0])],
				  	  "beam_constrs": {"lower": beam_lower, "upper": beam_upper},
				  	  "dose_constrs": {"lower": dose_lower, "upper": dose_upper},
				  	  "health_constrs": {"lower": health_lower, "upper": health_upper}}

	# Stage 1: Static beam problem.
	# Define variables.
	b = Variable((n,), nonneg=True)
	d = A_list[t_s] @ b

	h_lin = h_init - multiply(alpha[t_s], d) + gamma[t_s]
	h_quad = h_init - multiply(alpha[t_s], d) - multiply(beta[t_s], square(d)) + gamma[t_s]
	h_ptv = h_lin[is_target]
	h_oar = h_quad[~is_target]
	h = multiply(is_target, h_lin) + multiply(~is_target, h_quad)

	# Form objective.
	d_penalty = sum_squares(d[:-1]) + 0.25*square(d[-1])   # Lower penalty on generic body voxels.
	h_penalty_ptv = sum(pos(h_ptv))
	h_penalty_oar = (num_ptv/num_oar)*sum(neg(h_oar))
	h_penalty = h_penalty_ptv + h_penalty_oar

	# Add slack to health bounds.
	# h_hi_slack_weight = 1e4
	# h_hi_slack = Variable(h_ptv.shape, nonneg=True)
	# s_hi_penalty = h_hi_slack_weight*sum(h_hi_slack)

	h_lo_slack_weight = 1/(K-1)   # 0.25
	h_lo_slack = Variable(h_oar.shape, nonneg=True)
	s_lo_penalty = h_lo_slack_weight*sum(h_lo_slack)

	# s_penalty = s_hi_penalty + s_lo_penalty
	s_penalty = s_lo_penalty
	obj = d_penalty + h_penalty + s_penalty

	# Additional constraints.
	# constrs = [b <= beam_upper[t_s,:], d <= dose_upper[t_s,:], d >= dose_lower[t_s,:], h_ptv <= health_upper[t_s,is_target],
	# 			 h_oar >= health_lower[t_s,~is_target]]
	# constrs = [h_ptv <= health_upper[-1,is_target] + h_hi_slack, h_oar >= health_lower[-1,~is_target] - h_lo_slack]
	# constrs = [h_ptv <= health_upper[-1,is_target], h_oar >= health_lower[-1,~is_target] - h_lo_slack]
	# constrs = [b <= np.sum(beam_upper, axis=0), h_ptv <= health_upper[-1,is_target], h_oar >= health_lower[-1,~is_target] - h_lo_slack]
	constrs = [b <= np.sum(beam_upper, axis=0), d <= np.sum(dose_upper, axis=0), h_ptv <= health_upper[-1,is_target],
			   h_oar >= health_lower[-1,~is_target] - h_lo_slack]

	# Solve problem.
	print("Stage 1: Solving problem...")
	prob_1 = Problem(Minimize(obj), constrs)
	prob_1.solve(solver = "MOSEK")
	if prob_1.status not in SOLUTION_PRESENT:
		raise RuntimeError("Stage 1: Solver failed with status {0}".format(prob_1.status))
	setup_time = 0 if prob_1.solver_stats.setup_time is None else prob_1.solver_stats.setup_time
	solve_time = prob_1.solver_stats.solve_time
	run_time = (0 if prob_1.solver_stats.setup_time is None else prob_1.solver_stats.setup_time) + prob_1.solver_stats.solve_time

	# Save results.
	b_static = b.value   # Save optimal static beams for stage 2.
	d_static = np.vstack([A_list[t] @ b_static for t in range(T)])
	d_stage_1 = d.value
	# h_stage_1 = h.value
	h_stage_1 = h_init - alpha[t_s]*d_stage_1 - beta[t_s]*d_stage_1**2 + gamma[t_s]

	print("Stage 1 Results")
	print("Objective:", prob_1.value)
	print("Optimal Dose:", d_stage_1)
	print("Optimal Beam (Max):", np.max(b_static))
	print("Optimal Health:", h_stage_1)
	print("Setup Time:", prob_1.solver_stats.setup_time)
	print("Solve Time:", prob_1.solver_stats.solve_time)

	# Compare with AdaRad package.
	# prob_1_ada, b_1_ada, h_1_ada, d_1_ada, h_actual_1_ada, h_slack_1_ada = \
	# 	build_stat_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx_ada, t_static = 0, slack_oar_weight = h_lo_slack_weight)
	# prob_1_ada.solve(solver = "MOSEK")
	# if prob_1_ada.status not in SOLUTION_PRESENT:
	# 	raise RuntimeError("AdaRad Stage 1: Solver failed with status {0}".format(prob_1_ada.status))
	#
	# print("Compare with AdaRad")
	# print("Difference in Objective:", np.abs(prob_1.value - prob_1_ada.value))
	# print("Normed Difference in Beam:", np.linalg.norm(b_static - b_1_ada.value))
	# print("Normed Difference in Dose:", np.linalg.norm(d_stage_1 - d_1_ada.value))
	# print("Normed Difference in Health:", np.linalg.norm(h.value - h_1_ada.value))
	# print("Normed Difference in Health Slack:", np.linalg.norm(h_lo_slack.value - h_slack_1_ada[~is_target].value))
	# print("AdaRad Solve Time:", prob_1_ada.solver_stats.solve_time)

	# Plot optimal dose and health per structure.
	xlim_eps = 0.5
	plt.bar(range(K), d_stage_1, width = 0.8)
	plt.step(*form_step_xy(np.arange(K), dose_lower[-1,:], buf = 0.5), where = "mid", lw = 1, ls = "--", color = colors[1])
	plt.step(*form_step_xy(np.arange(K), dose_upper[-1,:], buf = 0.5), where = "mid", lw = 1, ls = "--", color = colors[1])
	plt.title("Treatment Dose vs. Structure")
	plt.xlim(-xlim_eps, K-1+xlim_eps)
	plt.show()

	health_bounds_fin = np.zeros(K)
	health_bounds_fin[is_target] = health_upper[-1,is_target]
	health_bounds_fin[~is_target] = health_lower[-1,~is_target]
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
	h_tayl_slack = Variable((T,K), nonneg=True)      # Slack in approximation.

	# Form objective.
	# d_penalty = sum_squares(d[:,:-1]) + 0.25*sum_squares(d[:,-1])   # Lower penalty on generic body voxels.
	d_penalty = square(u)*(sum_squares(d_static[:,:-1]) + 0.25*sum_squares(d_static[:,-1])).value
	h_penalty = sum(pos(h[1:,is_target])) + (num_ptv/num_oar)*sum(neg(h[1:,~is_target]))
	s_tayl_penalty = h_tayl_slack_weight*sum(h_tayl_slack)

	# Add slack to lower health bounds.
	# TODO: Should we continue to add slack to lower health bound on OARs?
	h_lo_slack_weight = 1/(K-1)   # 0.25
	h_lo_slack = Variable((T,num_oar), nonneg=True)
	s_lo_penalty = h_lo_slack_weight*sum(h_lo_slack)

	# s_penalty = s_tayl_penalty
	s_penalty = s_tayl_penalty + s_lo_penalty
	obj = d_penalty + h_penalty + s_penalty

	# Health dynamics.
	constrs = [h[0] == h_init]
	for t in range(T):
		# For PTV, use simple linear model (beta_t = 0).
		# constrs += [h[t+1,is_target] == h[t,is_target] - multiply(alpha[t,is_target], d[t,is_target]) + gamma[t,is_target] - h_tayl_slack[t,is_target]]
		constrs += [h[t+1,is_target] == h[t,is_target] - u*multiply(alpha[t,is_target], d_static[t,is_target]).value + gamma[t,is_target] - h_tayl_slack[t,is_target]]

		# For OAR, use linear-quadratic model with lossless relaxation.
		# constrs += [h[t+1,~is_target] <= h[t,~is_target] - multiply(alpha[t,~is_target], d[t,~is_target]) - multiply(beta[t,~is_target], square(d[t,~is_target])) + gamma[t,~is_target]]
		constrs += [h[t+1,~is_target] <= h[t,~is_target] - u*multiply(alpha[t,~is_target], d_static[t,~is_target]).value
														- square(u)*multiply(beta[t,~is_target], square(d_static[t,~is_target])).value + gamma[t,~is_target]]

	# Additional constraints.
	# constrs += [b <= np.min(beam_upper, axis=0), d <= dose_upper, d >= dose_lower, h[1:,is_target] <= health_upper[:,is_target], h[1:,~is_target] >= health_lower[:,~is_target]]
	# constrs += [b <= np.min(beam_upper, axis=0), d <= dose_upper, d >= dose_lower,
	#			h[1:,is_target] <= health_upper[:,is_target], h[1:,~is_target] >= health_lower[:,~is_target] - h_lo_slack]
	constrs += [b <= np.min(beam_upper, axis=0), u*d_static <= dose_upper, u*d_static >= dose_lower,
				h[1:,is_target] <= health_upper[:,is_target], h[1:,~is_target] >= health_lower[:,~is_target] - h_lo_slack]

	# Warm start.
	u.value = 1

	# Solve problem.
	print("Stage 2: Solving initial problem...")
	prob_2a = Problem(Minimize(obj), constrs)
	prob_2a.solve(solver = "MOSEK", warm_start = True)
	if prob_2a.status not in SOLUTION_PRESENT:
		raise RuntimeError("Stage 2 Initialization: Solver failed with status {0}".format(prob_2a.status))
	setup_time += 0 if prob_2a.solver_stats.setup_time is None else prob_2a.solver_stats.setup_time
	solve_time += prob_2a.solver_stats.solve_time
	run_time += (0 if prob_2a.solver_stats.setup_time is None else prob_2a.solver_stats.setup_time) + prob_2a.solver_stats.solve_time

	# Save results.
	u_stage_2_init = u.value
	d_stage_2_init = d.value   # Save optimal doses derived from constant factor for stage 2b.
	# h_stage_2_init = h.value
	h_stage_2_init = health_prog_act(h_init, T, alpha, beta, gamma, d_stage_2_init, is_target)
	s_stage_2_init = h_tayl_slack.value

	print("Stage 2 Initialization")
	print("Objective:", prob_2a.value)
	print("Optimal Beam Weight:", u_stage_2_init)
	# print("Optimal Dose:", d_stage_2_init)
	# print("Optimal Health:", h_stage_2_init)
	# print("Optimal Health Slack:", s_stage_2_init)
	print("Setup Time:", prob_2a.solver_stats.setup_time)
	print("Solve Time:", prob_2a.solver_stats.solve_time)

	# Compare with AdaRad package.
	# prob_2a_ada, u_2a_ada, b_2a_ada, h_2a_ada, d_2a_ada, h_lin_dyn_slack_2a_ada, h_lin_bnd_slack_2a_ada = \
	# 	build_scale_lin_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx_ada, b_1_ada.value, use_dyn_slack = True,
	# 		slack_dyn_weight = h_tayl_slack_weight, use_bnd_slack = True, slack_bnd_weight = h_lo_slack_weight)
	# prob_2a_ada.solve(solver = "MOSEK")
	# if prob_2a_ada.status not in SOLUTION_PRESENT:
	# 	raise RuntimeError("AdaRad Stage 2a: Solver failed with status {0}".format(prob_2a_ada.status))
	#
	# print("Compare with AdaRad")
	# print("Difference in Objective:", np.abs(prob_2a.value - prob_2a_ada.value))
	# print("Normed Difference in Beam:", np.linalg.norm(u_stage_2_init*b_static - b_2a_ada.value))
	# print("Normed Difference in Dose:", np.linalg.norm(d_stage_2_init - d_2a_ada.value))
	# print("Normed Difference in Health:", np.linalg.norm(h.value - h_2a_ada.value))
	# print("Normed Difference in Health Slack (Dynamics):", np.linalg.norm(s_stage_2_init - h_lin_dyn_slack_2a_ada.value))
	# print("Normed Difference in Health Slack (Bound):", np.linalg.norm(h_lo_slack.value - h_lin_bnd_slack_2a_ada[:,~is_target].value))
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
	h_tayl_slack = Variable((T,K), nonneg=True)      # Slack in approximation.
	d_parm = Parameter(d.shape, nonneg=True)   # Dose point around which to linearize.

	# Form objective.
	d_penalty = sum_squares(d[:,:-1]) + 0.25*sum_squares(d[:,-1])   # Lower penalty on generic body voxels.
	h_penalty = sum(pos(h[1:,is_target])) + (num_ptv/num_oar)*sum(neg(h[1:,~is_target]))
	s_tayl_penalty = h_tayl_slack_weight*sum(h_tayl_slack[:,is_target])

	# Add slack to lower health bounds.
	# TODO: Should we continue to add slack to lower health bound on OARs?
	h_lo_slack_weight = 1/(K-1)   # 0.25
	h_lo_slack = Variable((T, num_oar), nonneg=True)
	s_lo_penalty = h_lo_slack_weight * sum(h_lo_slack)

	# s_penalty = s_tayl_penalty
	s_penalty = s_tayl_penalty + s_lo_penalty
	obj = d_penalty + h_penalty + s_penalty

	# Health dynamics.
	constrs = [h[0] == h_init]
	for t in range(T):
		# For PTV, use first-order Taylor expansion of dose around d_parm.
		# constrs += [h[t+1,is_target] == h[t,is_target] - multiply(alpha[t,is_target], d[t,is_target]) - multiply(2*d[t,is_target] - d_parm[t,is_target], multiply(beta[t,is_target], d_parm[t,is_target])) \
		#											   + gamma[t,is_target] - h_tayl_slack[t,is_target]]
		constrs += [h[t+1,is_target] == h[t,is_target] - u[t]*multiply(alpha[t,is_target], d_static[t,is_target]).value
													   - multiply(2*u[t]*d_static[t,is_target] - d_parm[t,is_target], multiply(beta[t,is_target], d_parm[t,is_target]))
													   + gamma[t,is_target] - h_tayl_slack[t,is_target]]

		# For OAR, use linear-quadratic model with lossless relaxation.
		# constrs += [h[t+1,~is_target] <= h[t,~is_target] - multiply(alpha[t,~is_target], d[t,~is_target]) - multiply(beta[t,~is_target], square(d[t,~is_target])) + gamma[t,~is_target]]
		constrs += [h[t+1,~is_target] <= h[t,~is_target] - u[t]*multiply(alpha[t,~is_target], d_static[t,~is_target]).value
														 - square(u[t])*multiply(beta[t,~is_target], square(d_static[t,~is_target])).value + gamma[t,~is_target]]

	# Additional constraints.
	# constrs += [b <= beam_upper, d <= dose_upper, d >= dose_lower, h[1:,is_target] <= health_upper[:,is_target], h[1:,~is_target] >= health_lower[:,~is_target]]
	constrs += [b >= beam_lower, b <= beam_upper, d <= dose_upper, d >= dose_lower, h[1:,is_target] <= health_upper[:,is_target], h[1:,~is_target] >= health_lower[:,~is_target] - h_lo_slack]
	prob_2b = Problem(Minimize(obj), constrs)

	# Solve using CCP.
	print("Stage 2: Solving dynamic problem with CCP...")
	ccp_max_iter = 20
	ccp_eps = 1e-3

	# Warm start.
	u.value = np.array(T*[u_stage_2_init])
	h.value = h_stage_2_init
	h_tayl_slack.value = s_stage_2_init

	obj_old = np.inf
	d_parm.value = d_stage_2_init
	prob_2b_setup_time = 0
	prob_2b_solve_time = 0

	start = time()
	for k in range(ccp_max_iter):
		# Solve linearized problem.
		prob_2b.solve(solver = "MOSEK", warm_start = True)
		if prob_2b.status not in SOLUTION_PRESENT:
			raise RuntimeError("Stage 2 CCP: Solver failed on iteration {0} with status {1}".format(k, prob_2b.status))
		prob_2b_setup_time += 0 if prob_2b.solver_stats.setup_time is None else prob_2b.solver_stats.setup_time
		prob_2b_solve_time += prob_2b.solver_stats.solve_time

		# Terminate if change in objective is small.
		obj_diff = obj_old - prob_2b.value
		print("CCP Iteration {0}, Objective Difference: {1}".format(k, obj_diff))
		if obj_diff <= ccp_eps:
			break

		obj_old = prob_2b.value
		d_parm.value = d.value
	end = time()
	prob_2b_runtime = end - start

	setup_time += prob_2b_setup_time
	solve_time += prob_2b_solve_time
	run_time += prob_2b_runtime

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
	print("Setup Time:", prob_2b_setup_time)
	print("Solve Time:", prob_2b_solve_time)
	print("Runtime:", prob_2b_runtime)

	print("\nSolver Stats: Initialization")
	print("Total Setup Time:", setup_time)
	print("Total Solve Time:", solve_time)
	print("Total (Setup + Solve) Time:", setup_time + solve_time)
	print("Total Runtime:", run_time)

	# Compare with AdaRad package.
	# prob_2b_ada, u_2b_ada, b_2b_ada, h_2b_ada, d_2b_ada, d_parm_2b_ada, h_dyn_slack_2b_ada, h_bnd_slack_2b_ada = \
	# 	build_scale_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx_ada, b_static, use_dyn_slack = True,
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
	# print("Normed Difference in Health Slack (Dynamics):", np.linalg.norm(s_stage_2 - h_dyn_slack_2b_ada.value))
	# print("Normed Difference in Health Slack (Bound):", np.linalg.norm(h_lo_slack.value - h_bnd_slack_2b_ada[:,~is_target].value))
	# print("AdaRad Solve Time:", result_2b_ada["solve_time"])

	# Save to file.
	np.save(init_prefix + "beams.npy", b_stage_2)
	np.save(init_prefix + "doses.npy", d_stage_2)
	np.save(init_prefix + "health.npy", h_stage_2)
	np.save(init_prefix + "health_slack.npy", s_stage_2)

	# Plot optimal dose and health over time.
	plot_treatment(d_stage_2, stepsize = 10, bounds = (dose_lower, dose_upper), title = "Treatment Dose vs. Time", 
				color = colors[0], one_idx = True, filename = init_fig_prefix + "doses.png")
	plot_health(h_stage_2, curves = h_curves, stepsize = 10, bounds = (health_lower, health_upper), title = "Health Status vs. Time",
				label = "Treated", color = colors[0], one_idx = True, filename = init_fig_prefix + "health.png")

	# raise RuntimeError("Stop 2")

	# Main Stage: Dynamic optimal control problem.
	# Define variables.
	b = Variable((T,n), nonneg=True)
	d = vstack([A_list[t] @ b[t] for t in range(T)])
	h = Variable((T+1,K))

	# Used in Taylor expansion of PTV health dynamics.
	h_tayl_slack_weight = 1e4
	h_tayl_slack = Variable((T,K), nonneg=True)      # Slack in approximation.
	d_parm = Parameter(d.shape, nonneg=True)   # Dose point around which to linearize.

	# Form objective.
	d_penalty = sum_squares(d[:,:-1]) + 0.25*sum_squares(d[:,-1])
	h_penalty = sum(pos(h[1:,is_target])) + (num_ptv/num_oar)*sum(neg(h[1:,~is_target]))
	s_penalty = h_tayl_slack_weight*sum(h_tayl_slack[:,is_target])
	obj = d_penalty + h_penalty + s_penalty

	# Health dynamics.
	constrs = [h[0] == h_init]
	for t in range(T):
		# For PTV, use first-order Taylor expansion of dose around d_parm.
		constrs += [h[t+1,is_target] == h[t,is_target] - multiply(alpha[t,is_target], d[t,is_target]) - multiply(2*d[t,is_target] - d_parm[t,is_target], multiply(beta[t,is_target], d_parm[t,is_target])) \
													   + gamma[t,is_target] - h_tayl_slack[t,is_target]]

		# For OAR, use linear-quadratic model with lossless relaxation.
		constrs += [h[t+1,~is_target] <= h[t,~is_target] - multiply(alpha[t,~is_target], d[t,~is_target]) - multiply(beta[t,~is_target], square(d[t,~is_target])) + gamma[t,~is_target]]

	# Additional constraints.
	constrs += [b >= beam_lower, b <= beam_upper, d <= dose_upper, d >= dose_lower, h[1:,is_target] <= health_upper[:,is_target], h[1:,~is_target] >= health_lower[:,~is_target]]
	prob_main = Problem(Minimize(obj), constrs)

	# Solve using CCP.
	print("Main Stage: Solving dynamic problem with CCP...")
	ccp_max_iter = 20
	ccp_eps = 1e-3

	# Warm start.
	b.value = b_stage_2
	h.value = h_stage_2
	h_tayl_slack.value = s_stage_2

	obj_old = np.inf
	d_parm.value = d_stage_2   # Initialize using optimal dose from stage 2.
	prob_main_setup_time = 0
	prob_main_solve_time = 0

	start = time()
	for k in range(ccp_max_iter):
		# Solve linearized problem.
		prob_main.solve(solver = "MOSEK", warm_start = True)
		if prob_main.status not in SOLUTION_PRESENT:
			raise RuntimeError("Main Stage CCP: Solver failed on iteration {0} with status {1}".format(k, prob_main.status))
		prob_main_setup_time += 0 if prob_main.solver_stats.setup_time is None else prob_main.solver_stats.setup_time
		prob_main_solve_time += prob_main.solver_stats.solve_time

		# Terminate if change in objective is small.
		obj_diff = obj_old - prob_main.value
		print("CCP Iteration {0}, Objective Difference: {1}".format(k, obj_diff))
		if obj_diff <= ccp_eps:
			break

		obj_old = prob_main.value
		d_parm.value = d.value
	end = time()
	prob_main_runtime = end - start

	setup_time += prob_main_setup_time
	solve_time += prob_main_solve_time
	run_time += prob_main_runtime

	# Save results.
	b_main = b.value
	d_main = d.value
	# h_main = h.value
	h_main = health_prog_act(h_init, T, alpha, beta, gamma, d_main, is_target)
	s_main = h_tayl_slack.value

	print("Main Stage Results")
	print("Objective:", prob_main.value)
	# print("Optimal Dose:", d_main)
	# print("Optimal Health:", h_main)
	# print("Optimal Health Slack:", s_main)
	print("Setup Time:", prob_main_setup_time)
	print("Solve Time:", prob_main_solve_time)
	print("Total (Setup + Solve) Time:", prob_main_setup_time + prob_main_solve_time)
	print("Runtime:", prob_main_runtime)

	print("\nSolver Stats: All Stages")
	print("Total Setup Time:", setup_time)
	print("Total Solve Time:", solve_time)
	print("Total (Setup + Solve) Time:", setup_time + solve_time)
	print("Total Runtime:", run_time)

	# Save to file.
	np.save(final_prefix + "beams.npy", b_main)
	np.save(final_prefix + "doses.npy", d_main)
	np.save(final_prefix + "health.npy", h_main)
	np.save(final_prefix + "health_slack.npy", s_main)

	# Plot optimal dose and health over time.
	plot_treatment(d_main, stepsize = 10, bounds = (dose_lower, dose_upper), title = "Treatment Dose vs. Time", one_idx = True, 
			   filename = final_fig_prefix + "doses.png")
	plot_health(h_main, curves = h_curves, stepsize = 10, bounds = (health_lower, health_upper), title = "Health Status vs. Time", 
			  label = "Treated", color = colors[0], one_idx = True, filename = final_fig_prefix + "health.png")

if __name__ == "__main__":
	main()