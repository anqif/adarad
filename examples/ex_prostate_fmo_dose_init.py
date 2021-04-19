import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TKAgg")

import cvxpy
from cvxpy import *
from cvxpy.settings import SOLUTION_PRESENT

from fractionation.init_funcs import *
from fractionation.utilities.plot_utils import *
from fractionation.utilities.file_utils import yaml_to_dict
from fractionation.utilities.data_utils import health_prog_act

input_path = "C:/Users/Anqi/Documents/Software/fractionation/examples/data/"
output_path = "C:/Users/Anqi/Documents/Software/fractionation/examples/output/"
fig_path = output_path + "figures/"

output_prefix = output_path + "ex3_prostate_fmo_"
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

def fin_upper_constr(x, bound):
	if np.isscalar(bound):
		if not np.isinf(bound):
			return [x <= bound]
	else:
		if not np.all(x.shape == bound.shape):
			raise ValueError("bound must have dimensions {0}".format(x.shape))
		is_finite = ~np.isinf(bound)
		if np.all(is_finite):
			return [x <= bound]
		elif np.any(is_finite):
			return [x[is_finite] <= bound[is_finite]]
	return []

def fin_lower_constr(x, bound):
	return fin_upper_constr(-x, -bound)

def main():
	# Problem data.
	# patient_bio, patient_rx, visuals = yaml_to_dict(input_path + "ex_prostate_FMO_stanford.yml")
	patient_bio, patient_rx, visuals = yaml_to_dict(input_path + "ex_prostate_FMO_stanford_test.yml")

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
	h_penalty_ptv = pos(h_ptv)
	h_penalty_oar = (num_ptv/num_oar)*sum(neg(h_oar))
	h_penalty = h_penalty_ptv + h_penalty_oar

	# Add slack to health bounds.
	# h_hi_slack_weight = 1e4
	# h_hi_slack = Variable(h_ptv.shape, nonneg=True)
	# s_hi_penalty = h_hi_slack_weight*sum(h_hi_slack)

	h_lo_slack_weight = 0.25
	h_lo_slack = Variable(h_oar.shape, nonneg=True)
	s_lo_penalty = h_lo_slack_weight*sum(h_lo_slack)

	# s_penalty = s_hi_penalty + s_lo_penalty
	s_penalty = s_lo_penalty
	obj = d_penalty + h_penalty + s_penalty

	# Additional constraints.
	# constrs = [b <= beam_upper[t_s,:], d <= dose_upper[t_s,:], d >= dose_lower[t_s,:], h_ptv <= health_upper[t_s,is_target], h_oar >= health_lower[t_s,~is_target]]
	# constrs = [h_ptv <= health_upper[-1,is_target] + h_hi_slack, h_oar >= health_lower[-1,~is_target] - h_lo_slack]
	constrs = [h_ptv <= health_upper[-1,is_target], h_oar >= health_lower[-1,~is_target] - h_lo_slack]

	# Solve problem.
	print("Stage 1: Solving problem...")
	prob_1 = Problem(Minimize(obj), constrs)
	prob_1.solve(solver = "MOSEK")
	if prob_1.status not in SOLUTION_PRESENT:
		raise RuntimeError("Stage 1: Solver failed with status {0}".format(prob_1.status))
	solve_time = prob_1.solver_stats.solve_time

	# Save results.
	b_static = b.value   # Save optimal static beams for stage 2.
	d_stage_1 = d.value
	# h_stage_1 = h.value
	h_stage_1 = h_init - alpha[t_s]*d_stage_1 - beta[t_s]*d_stage_1**2 + gamma[t_s]

	print("Stage 1 Results")
	print("Objective:", prob_1.value)
	print("Optimal Dose:", d_stage_1)
	print("Optimal Health:", h_stage_1)
	print("Solve Time:", prob_1.solver_stats.solve_time)

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

	# Stage 2a: Dynamic scaling problem with constant factor.
	u = Variable(nonneg=True)
	b = u*b_static
	d = vstack([A_list[t] @ b for t in range(T)])
	h = Variable((T+1,K))

	# Used in Taylor expansion of PTV health dynamics.
	h_slack_weight = 1e4
	h_slack = Variable((T,K), nonneg=True)      # Slack in approximation.

	# Form objective.
	d_penalty = sum_squares(d[:,:-1]) + 0.25*sum_squares(d[:,-1])   # Lower penalty on generic body voxels.
	h_penalty = sum(pos(h[1:,is_target])) + 0.25*sum(neg(h[1:,~is_target]))
	s_penalty = h_slack_weight*sum(h_slack[:,is_target])
	obj = d_penalty + h_penalty + s_penalty

	# Health dynamics.
	constrs = [h[0] == h_init]
	for t in range(T):
		# For PTV, use simple linear model (beta_t = 0).
		constrs += [h[t+1,is_target] == h[t,is_target] - multiply(alpha[t,is_target], d[t,is_target]) + gamma[t,is_target] - h_slack[t,is_target]]

		# For OAR, use linear-quadratic model with lossless relaxation.
		constrs += [h[t+1,~is_target] <= h[t,~is_target] - multiply(alpha[t,~is_target], d[t,~is_target]) - multiply(beta[t,~is_target], square(d[t,~is_target])) + gamma[t,~is_target]]

	# Additional constraints.
	constrs += [b <= np.min(beam_upper, axis=0), b >= np.max(beam_lower, axis=0)]
	constrs += [h[1:,is_target] <= health_upper[:,is_target], h[1:,~is_target] >= health_lower[:,~is_target]]
	constrs += fin_upper_constr(d, dose_upper) + fin_lower_constr(d, dose_lower)

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
	s_stage_2_init = h_slack.value

	print("Stage 2 Initialization")
	print("Objective:", prob_2a.value)
	print("Optimal Beam Weight:", u_stage_2_init)
	# print("Optimal Dose:", d_stage_2_init)
	# print("Optimal Health:", h_stage_2_init)
	# print("Optimal Health Slack:", s_stage_2_init)
	print("Solve Time:", prob_2a.solver_stats.solve_time)

	# Stage 2b: Dynamic scaling problem with time-varying factors.
	# Define variables.
	u = Variable((T,), nonneg=True)
	b = vstack([u[t]*b_static for t in range(T)])
	d = vstack([A_list[t] @ b[t] for t in range(T)])
	h = Variable((T+1,K))

	# Used in Taylor expansion of PTV health dynamics.
	h_slack_weight = 1e4
	h_slack = Variable((T,K), nonneg=True)      # Slack in approximation.
	d_parm = Parameter(d.shape, nonneg=True)   # Dose point around which to linearize.

	# Form objective.
	d_penalty = sum_squares(d[:,:-1]) + 0.25*sum_squares(d[:,-1])   # Lower penalty on generic body voxels.
	h_penalty = sum(pos(h[1:,is_target])) + 0.25*sum(neg(h[1:,~is_target]))
	s_penalty = h_slack_weight*sum(h_slack[:,is_target])
	obj = d_penalty + h_penalty + s_penalty

	# Health dynamics.
	constrs = [h[0] == h_init]
	for t in range(T):
		# For PTV, use first-order Taylor expansion of dose around d_parm.
		constrs += [h[t+1,is_target] == h[t,is_target] - multiply(alpha[t,is_target], d[t,is_target]) - multiply(2*d[t,is_target] - d_parm[t,is_target], multiply(beta[t,is_target], d_parm[t,is_target])) \
													   + gamma[t,is_target] - h_slack[t,is_target]]

		# For OAR, use linear-quadratic model with lossless relaxation.
		constrs += [h[t+1,~is_target] <= h[t,~is_target] - multiply(alpha[t,~is_target], d[t,~is_target]) - multiply(beta[t,~is_target], square(d[t,~is_target])) + gamma[t,~is_target]]

	# Additional constraints.
	constrs += [b <= beam_upper, h[1:,is_target] <= health_upper[:,is_target], h[1:,~is_target] >= health_lower[:,~is_target]]
	constrs += fin_upper_constr(d, dose_upper) + fin_lower_constr(d, dose_lower)
	prob_2b = Problem(Minimize(obj), constrs)

	# Solve using CCP.
	print("Stage 2: Solving dynamic problem with CCP...")
	ccp_max_iter = 20
	ccp_eps = 1e-3

	# Warm start.
	u.value = np.array(T*[u_stage_2_init])
	h.value = h_stage_2_init
	h_slack.value = s_stage_2_init

	obj_old = np.inf
	d_parm.value = d_stage_2_init
	for k in range(ccp_max_iter):
		# Solve linearized problem.
		prob_2b.solve(solver = "MOSEK", warm_start = True)
		if prob_2b.status not in SOLUTION_PRESENT:
			raise RuntimeError("Stage 2 CCP: Solver failed on iteration {0} with status {1}".format(k, prob_2b.status))

		# Terminate if change in objective is small.
		obj_diff = obj_old - prob_2b.value
		print("CCP Iteration {0}, Objective Difference: {1}".format(k, obj_diff))
		if obj_diff <= ccp_eps:
			break

		obj_old = prob_2b.value
		d_parm.value = d.value
	solve_time += prob_2b.solver_stats.solve_time

	# Save results.
	u_stage_2 = u.value
	b_stage_2 = b.value
	d_stage_2 = d.value
	# h_stage_2 = h.value
	h_stage_2 = health_prog_act(h_init, T, alpha, beta, gamma, d_stage_2, is_target)
	s_stage_2 = h_slack.value

	print("Stage 2 Results")
	print("Objective:", prob_2b.value)
	print("Optimal Beam Weight:", u_stage_2)
	# print("Optimal Dose:", d_stage_2)
	# print("Optimal Health:", h_stage_2)
	# print("Optimal Health Slack:", s_stage_2)
	print("Solve Time:", prob_2b.solver_stats.solve_time)

	# Save to file.
	np.save(init_prefix + "beams.npy", b_stage_2)
	np.save(init_prefix + "doses.npy", d_stage_2)
	np.save(init_prefix + "health.npy", h_stage_2)
	np.save(init_prefix + "health_slack.npy", s_stage_2)

	# Plot optimal dose and health over time.
	plot_treatment(d_stage_2, stepsize = 10, bounds = (dose_lower, dose_upper), title = "Treatment Dose vs. Time", 
				color = colors[0], one_idx = True, filename = init_prefix + "doses.png")
	plot_health(h_stage_2, curves = h_curves, stepsize = 10, bounds = (health_lower, health_upper), title = "Health Status vs. Time", label = "Treated", 
				color = colors[0], one_idx = True, filename = init_prefix + "health.png")

	# Main Stage: Dynamic optimal control problem.
	# Define variables.
	b = Variable((T,n), nonneg=True)
	d = vstack([A_list[t] @ b[t] for t in range(T)])
	h = Variable((T+1,K))

	# Used in Taylor expansion of PTV health dynamics.
	h_slack_weight = 1e4
	h_slack = Variable((T,K), nonneg=True)      # Slack in approximation.
	d_parm = Parameter(d.shape, nonneg=True)   # Dose point around which to linearize.

	# Form objective.
	d_penalty = sum_squares(d[:,:-1]) + 0.25*sum_squares(d[:,-1])
	h_penalty = sum(pos(h[1:,is_target])) + 0.25*sum(neg(h[1:,~is_target]))
	s_penalty = h_slack_weight*sum(h_slack[:,is_target])
	obj = d_penalty + h_penalty + s_penalty

	# Health dynamics.
	constrs = [h[0] == h_init]
	for t in range(T):
		# For PTV, use first-order Taylor expansion of dose around d_parm.
		constrs += [h[t+1,is_target] == h[t,is_target] - multiply(alpha[t,is_target], d[t,is_target]) - multiply(2*d[t,is_target] - d_parm[t,is_target], multiply(beta[t,is_target], d_parm[t,is_target])) \
													   + gamma[t,is_target] - h_slack[t,is_target]]

		# For OAR, use linear-quadratic model with lossless relaxation.
		constrs += [h[t+1,~is_target] <= h[t,~is_target] - multiply(alpha[t,~is_target], d[t,~is_target]) - multiply(beta[t,~is_target], square(d[t,~is_target])) + gamma[t,~is_target]]

	# Additional constraints.
	constrs += [b <= beam_upper, h[1:,is_target] <= health_upper[:,is_target], h[1:,~is_target] >= health_lower[:,~is_target]]
	constrs += fin_upper_constr(d, dose_upper) + fin_lower_constr(d, dose_lower)
	prob_main = Problem(Minimize(obj), constrs)

	# Solve using CCP.
	print("Main Stage: Solving dynamic problem with CCP...")
	ccp_max_iter = 20
	ccp_eps = 1e-3

	# Warm start.
	b.value = b_stage_2
	h.value = h_stage_2
	h_slack.value = s_stage_2

	obj_old = np.inf
	d_parm.value = d_stage_2   # Initialize using optimal dose from stage 2.
	for k in range(ccp_max_iter):
		# Solve linearized problem.
		prob_main.solve(solver = "MOSEK", warm_start = True)
		if prob_main.status not in SOLUTION_PRESENT:
			raise RuntimeError("Main Stage CCP: Solver failed on iteration {0} with status {1}".format(k, prob_main.status))

		# Terminate if change in objective is small.
		obj_diff = obj_old - prob_main.value
		print("CCP Iteration {0}, Objective Difference: {1}".format(k, obj_diff))
		if obj_diff <= ccp_eps:
			break

		obj_old = prob_main.value
		d_parm.value = d.value
	solve_time += prob_main.solver_stats.solve_time

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
	print("Solve Time:", prob_main.solver_stats.solve_time)
	print("Total Solve Time:", solve_time)

	# Save to file.
	np.save(final_prefix + "beams.npy", b_main)
	np.save(final_prefix + "doses.npy", d_main)
	np.save(final_prefix + "health.npy", h_main)
	np.save(final_prefix + "health_slack.npy", s_main)

	# Plot optimal dose and health over time.
	plot_treatment(d_main, stepsize = 10, bounds = (dose_lower, dose_upper), title = "Treatment Dose vs. Time", one_idx = True, 
			   filename = final_prefix + "doses.png")
	plot_health(h_main, curves = h_curves, stepsize = 10, bounds = (health_lower, health_upper), title = "Health Status vs. Time", 
			  label = "Treated", color = colors[0], one_idx = True, filename = final_prefix + "health.png")

if __name__ == "__main__":
    main()