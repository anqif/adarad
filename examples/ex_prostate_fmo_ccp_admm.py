import numpy as np
import numpy.linalg as LA
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TKAgg")

import cvxpy
from cvxpy import *
from cvxpy.settings import SOLUTION_PRESENT
from multiprocessing import Process, Pipe

from adarad.utilities.plot_utils import *
from adarad.utilities.file_utils import yaml_to_dict
from adarad.utilities.data_utils import health_prog_act

INIT_FROM_FILE = True
input_path = "/home/anqi/Documents/software/adarad/examples/data/"
output_path = "/home/anqi/Documents/software/adarad/examples/output/"

output_prefix = output_path + "ex3_prostate_fmo_"
init_file = output_prefix + "init_doses.npy"
final_ccp_prefix = output_prefix + "ccp_"
final_admm_prefix = output_prefix + "admm_"

# Beam subproblems.
def run_beam_proc(pipe, A, beam_lower, beam_upper, dose_upper, dose_lower, rho_init):
	K, n = A.shape

	# Define variables.
	b = Variable((n,), nonneg=True)
	d = A @ b

	# Initialize parameters.
	rho = Parameter(pos=True, value=rho_init)
	u = Parameter((K,), value=np.zeros(K))
	d_tld = Parameter((K,), nonneg=True, value=np.zeros(K))

	# Form objective.
	d_penalty = sum_squares(d[:-1]) + 0.25*square(d[-1])
	c_penalty = (rho/2.0)*sum_squares(d - d_tld - u)
	obj = d_penalty + c_penalty

	constrs = [b >= beam_lower, b <= beam_upper, d <= dose_upper, d >= dose_lower]
	prob = Problem(Minimize(obj), constrs)

	# ADMM loop.
	finished = False
	while not finished:
		d_tld.value, u.value = pipe.recv()
		prob.solve(solver = "MOSEK", warm_start = True)
		if prob.status not in SOLUTION_PRESENT:
			raise RuntimeError("ADMM: Solver failed on beam subproblem with status {0}".format(prob.status))
		pipe.send((d.value, prob.solver_stats.solve_time))
		finished = pipe.recv()

	# Send final beams and doses.
	pipe.send((b.value, d.value))

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
	t_s = 0  # Static session.
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
	h_prog = health_prog_act(h_init, T, gamma=gamma)
	h_curves = [{"h": h_prog, "label": "Untreated", "kwargs": {"color": colors[1]}}]

	# CCP: Dynamic optimal control problem.
	# Define variables.
	b = Variable((T,n), nonneg=True)
	d = vstack([A_list[t] @ b[t] for t in range(T)])
	h = Variable((T+1,K))

	# Used in Taylor expansion of PTV health dynamics.
	h_slack_weight = 1e4
	h_slack = Variable((T,K), nonneg=True)   # Slack in approximation.
	d_parm = Parameter((T,K), nonneg=True)   # Dose point around which to linearize.
	# d_init_ccp = np.zeros((T,K))
	d_init_ccp = np.load(init_file) if INIT_FROM_FILE else np.zeros((T,K))

	# Form objective.
	d_penalty = sum_squares(d[:,:-1]) + 0.25*sum_squares(d[:,-1])
	h_penalty = sum(pos(h[1:,is_target])) + 0.25*sum(neg(h[1:,~is_target]))
	s_penalty = h_slack_weight*sum(h_slack[:,is_target])
	obj = d_penalty + h_penalty + s_penalty

	# Health dynamics.
	constrs = [h[0] == h_init]
	for t in range(T):
		# For PTV, use first-order Taylor expansion of dose around d_parm.
		constrs += [h[t+1,is_target] == h[t,is_target] - multiply(alpha[t,is_target], d[t,is_target])
						- multiply(2*d[t,is_target] - d_parm[t,is_target], multiply(beta[t,is_target], d_parm[t,is_target]))
						+ gamma[t,is_target] - h_slack[t,is_target]]

		# For OAR, use linear-quadratic model with lossless relaxation.
		constrs += [h[t+1,~is_target] <= h[t,~is_target] - multiply(alpha[t,~is_target], d[t,~is_target])
						- multiply(beta[t,~is_target], square(d[t,~is_target])) + gamma[t,~is_target]]

	# Additional constraints.
	constrs += [b >= beam_lower, b <= beam_upper, d <= dose_upper, d >= dose_lower,
				h[1:,is_target] <= health_upper[:,is_target], h[1:,~is_target] >= health_lower[:,~is_target]]
	prob_ccp = Problem(Minimize(obj), constrs)

	# Solve using CCP.
	max_iter_ccp = 15
	eps_ccp = 1e-3

	print("CCP: Solving dynamic problem...")
	k = 0
	solve_time_ccp = 0
	obj_old = np.inf
	d_parm.value = d_init_ccp
	while k < max_iter_ccp:
		# Solve linearized problem.
		prob_ccp.solve(solver = "MOSEK", warm_start = True)
		if prob_ccp.status not in SOLUTION_PRESENT:
			raise RuntimeError("CCP: Solver failed on iteration {0} with status {1}".format(k, prob_ccp.status))

		# Terminate if change in objective is small.
		obj_diff = obj_old - prob_ccp.value
		print("CCP Iteration {0}, Objective Difference: {1}".format(k, obj_diff))
		k = k + 1
		if obj_diff <= eps_ccp:
			break

		obj_old = prob_ccp.value
		d_parm.value = d.value
		solve_time_ccp += prob_ccp.solver_stats.solve_time

	# Save results.
	b_ccp = b.value
	d_ccp = d.value
	# h_ccp = h.value
	h_ccp = health_prog_act(h_init, T, alpha, beta, gamma, d_ccp, is_target)
	h_slack_ccp = h_slack.value

	obj_ccp = prob_ccp.value
	iters_ccp = k

	print("CCP Results")
	print("Objective:", obj_ccp)
	# print("Optimal Dose:", d_ccp)
	# print("Optimal Health:", h_ccp)
	# print("Optimal Health Slack:", h_slack_ccp)
	print("Solve Time:", solve_time_ccp)
	print("Iterations:", iters_ccp)

	# Save to file.
	np.save(final_ccp_prefix + "beams.npy", b_ccp)
	np.save(final_ccp_prefix + "doses.npy", d_ccp)
	np.save(final_ccp_prefix + "health.npy", h_ccp)
	np.save(final_ccp_prefix + "health_slack.npy", h_slack_ccp)

	# Plot optimal health and dose over time.
	# plot_treatment(d_ccp, stepsize = 10, bounds = (dose_lower, dose_upper), title = "CCP: Treatment Dose vs. Time", one_idx = True,
	#				filename = final_ccp_prefix + "doses.png")
	# plot_health(h_ccp, curves = h_curves, stepsize = 10, bounds = (health_lower, health_upper), title = "CCP: Health Status vs. Time",
	# 			label = "Treated", color = colors[0], one_idx = True, filename = final_ccp_prefix + "health.png")

	# ADMM: Dynamic optimal control problem.
	rho = Parameter(pos=True)
	u = Parameter((T,K))

	# Health subproblem.
	h = Variable((T+1,K))
	d_tld = Variable((T,K), nonneg=True)
	d_cons_parm = Parameter((T,K), nonneg=True)
	d_tayl_parm = Parameter((T,K), nonneg=True)

	# Used in Taylor expansion of PTV health dynamics.
	h_slack_weight = 1e4
	h_slack = Variable((T,K), nonneg=True)   # Slack in approximation.

	h_penalty = sum(pos(h[1:,is_target])) + 0.25*sum(neg(h[1:,~is_target]))
	s_penalty = h_slack_weight*sum(h_slack[:,is_target])
	c_penalty = (rho/2.0)*sum_squares(d_tld - d_cons_parm + u)
	obj = h_penalty + s_penalty + c_penalty

	# Health dynamics.
	constrs = [h[0] == h_init]
	for t in range(T):
		# For PTV, use first-order Taylor expansion of dose around d_parm.
		constrs += [h[t+1,is_target] == h[t,is_target] - multiply(alpha[t,is_target], d_tld[t,is_target])
						- multiply(2*d_tld[t,is_target] - d_tayl_parm[t,is_target], multiply(beta[t,is_target], d_tayl_parm[t,is_target]))
						+ gamma[t,is_target] - h_slack[t,is_target]]

		# For OAR, use linear-quadratic model with lossless relaxation.
		constrs += [h[t+1,~is_target] <= h[t,~is_target] - multiply(alpha[t,~is_target], d_tld[t,~is_target])
						- multiply(beta[t,~is_target], square(d_tld[t,~is_target])) + gamma[t,~is_target]]
	constrs += [d_tld <= dose_upper, d_tld >= dose_lower,
				h[1:,is_target] <= health_upper[:,is_target], h[1:,~is_target] >= health_lower[:,~is_target]]

	prob_h = Problem(Minimize(obj), constrs)
	prob_h_dict = {"prob": prob_h, "h": h, "h_slack": h_slack, "d_tld": d_tld, "d_cons_parm": d_cons_parm, "d_tayl_parm": d_tayl_parm}

	# Initialize main loop.
	rho_init = 80.0
	u_init = np.zeros(u.shape)
	# d_init_admm = np.zeros(d_tld.shape)
	d_init_admm = np.load(init_file) if INIT_FROM_FILE else np.zeros(d_tld.shape)

	# Set up beam subproblem processes.
	pipes = []
	procs = []
	for t in range(T):
		local, remote = Pipe()
		pipes += [local]
		procs += [Process(target=run_beam_proc, args=(remote, A_list[t], beam_lower[t], beam_upper[t],
													  dose_upper[t], dose_lower[t], rho_init))]
		procs[-1].start()

	# Solve using ADMM.
	admm_max_iter = 1000
	eps_abs = 1e-6   # Absolute stopping tolerance.
	eps_rel = 1e-3   # Relative stopping tolerance.

	print("ADMM: Solving dynamic problem...")
	rho.value = rho_init
	u.value = u_init
	d_tld_var_val = d_init_admm
	d_tld_var_val_old = d_init_admm

	k = 0
	solve_time_admm = 0
	r_prim_norms = np.zeros(admm_max_iter)
	r_dual_norms = np.zeros(admm_max_iter)
	finished = (k >= admm_max_iter)
	while not finished:
		if k % 10 == 0:
			print("ADMM Iteration {0}".format(k))

		# Solve beam subproblems in parallel.
		for t in range(T):
			pipes[t].send((d_tld_var_val[t], u.value[t]))
		dt_update = [pipe.recv() for pipe in pipes]
		d_rows, d_times = map(list, zip(*dt_update))
		d_var_val = np.row_stack(d_rows)
		solve_time_admm += np.max(d_times)   # Take max of all solve times, since subproblems solved in parallel.

		# Solve health subproblem using CCP.
		obj_old = np.inf
		prob_h_dict["d_cons_parm"].value = d_var_val
		prob_h_dict["d_tayl_parm"].value = d_tld_var_val_old   # TODO: What dose point should we linearize PTV health dynamics around?

		for l in range(max_iter_ccp):
			# Solve linearized problem.
			prob_h_dict["prob"].solve(solver = "MOSEK", warm_start = True)
			if prob_h_dict["prob"].status not in SOLUTION_PRESENT:
				raise RuntimeError("ADMM CCP: Solver failed on ADMM iteration {0}, CCP iteration {1} with status {2}".format(k, l, prob_h_dict["prob"].status))
			solve_time_admm += prob_h_dict["prob"].solver_stats.solve_time

			# Terminate if change in objective is small.
			obj_diff = obj_old - prob_h_dict["prob"].value
			# print("ADMM CCP Iteration {0}, Objective Difference: {1}".format(l, obj_diff))
			if obj_diff <= eps_ccp:
				break
			obj_old = prob_h_dict["prob"].value
			prob_h_dict["d_tayl_parm"].value = prob_h_dict["d_tld"].value

		d_tld_var_val_old = d_tld_var_val
		d_tld_var_val = prob_h_dict["d_tld"].value

		# Update dual values.
		u.value = u.value + d_tld_var_val - d_var_val

		# Calculate residuals.
		r_prim = d_var_val - d_tld_var_val
		r_dual = rho.value*(d_tld_var_val - d_tld_var_val_old)

		# Check stopping criteria.
		r_prim_norms[k] = LA.norm(r_prim)
		r_dual_norms[k] = LA.norm(r_dual)
		eps_prim = eps_abs*np.sqrt(T*K) + eps_rel*np.max([LA.norm(d_var_val), LA.norm(d_tld_var_val)])
		eps_dual = eps_abs*np.sqrt(T*K) + eps_rel*LA.norm(u.value)

		finished = (k + 1 >= admm_max_iter) or (r_prim_norms[k] <= eps_prim and r_dual_norms[k] <= eps_dual)
		k = k + 1
		for t in range(T):
			pipes[t].send(finished)

	# Get final beams and doses from beam subproblem.
	bd_update = [pipe.recv() for pipe in pipes]
	b_rows, d_rows = map(list, zip(*bd_update))
	[p.terminate() for p in procs]

	# Save results.
	iters_admm = k
	r_prim_admm = np.array(r_prim_norms[:iters_admm])
	r_dual_admm = np.array(r_dual_norms[:iters_admm])

	b_admm = np.row_stack(b_rows)
	d_var_val = np.row_stack(d_rows)
	d_admm = (d_var_val + d_tld_var_val)/2.0   # Final dose = average of original and consensus dose values.
	# h_admm = prob_h_dict["h"].value
	h_admm = health_prog_act(h_init, T, alpha, beta, gamma, d_admm, is_target)
	h_slack_admm = prob_h_dict["h_slack"].value

	# Calculate true objective.
	d_penalty_admm = np.sum(d_admm[:,:-1]**2) + 0.25*np.sum(d_admm[:,-1]**2)
	h_penalty_admm = np.sum(np.maximum(h_admm[1:,is_target], 0)) + 0.25*np.sum(np.maximum(-h_admm[1:,~is_target], 0))
	s_penalty_admm = h_slack_weight*np.sum(h_slack_admm[:,is_target])
	obj_admm = d_penalty_admm + h_penalty_admm + s_penalty_admm

	print("ADMM Results")
	print("Objective:", obj_admm)
	# print("Optimal Dose:", d_admm)
	# print("Optimal Health:", h_admm)
	# print("Optimal Health Slack:", h_slack_admm)
	print("Solve Time:", solve_time_admm)
	print("Iterations:", iters_admm)
	print("Primal Residual (Norm):", r_prim_admm[-1])
	print("Dual Residual (Norm):", r_dual_admm[-1])

	# Save to file.
	np.save(final_admm_prefix + "beams.npy", b_admm)
	np.save(final_admm_prefix + "doses.npy", d_admm)
	np.save(final_admm_prefix + "health.npy", h_admm)
	np.save(final_admm_prefix + "health_slack.npy", h_slack_admm)
	np.save(final_admm_prefix + "primal_residuals.npy", r_prim_admm)
	np.save(final_admm_prefix + "dual_residuals.npy", r_dual_admm)

	# Plot primal and dual residual norms over time.
	plot_residuals(r_prim_admm, r_dual_admm, semilogy = True)

	# Plot optimal health and dose over time.
	h_curves += [{"h": h_ccp, "label": "Treated (CCP)", "kwargs": {"color": colors[2]}}]
	d_curves  = [{"d": d_ccp, "label": "Dose Plan (CCP)", "kwargs": {"color": colors[2]}}]
	plot_treatment(d_admm, curves = d_curves, stepsize = 10, bounds = (dose_lower, dose_upper), 
					title = "Treatment Dose vs. Time", label = "Dose Plan (ADMM)", color = colors[0], one_idx = True,
					filename = final_admm_prefix + "doses.png")
	plot_health(h_admm, curves = h_curves, stepsize = 10, bounds = (health_lower, health_upper),
				title = "Health Status vs. Time", label = "Treated (ADMM)", color = colors[0], one_idx = True,
				filename = final_admm_prefix + "health.png")

	print("Compare CCP and ADMM Results")
	print("Difference in Objective:", np.abs(obj_ccp - obj_admm))
	print("Normed Difference in Beams:", LA.norm(b_ccp - b_admm))
	print("Normed Difference in Dose:", LA.norm(d_ccp - d_admm))
	print("Normed Difference in Health:", LA.norm(h_ccp - h_admm))
	print("Normed Difference in Health Slack:", LA.norm(h_slack_ccp - h_slack_admm))
	print("CCP Solve Time - ADMM Solve Time:", solve_time_ccp - solve_time_admm)

if __name__ == "__main__":
	main()
