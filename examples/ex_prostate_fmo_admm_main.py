import numpy as np
import numpy.linalg as LA
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TKAgg")

import cvxpy
from cvxpy import *
from cvxpy.settings import SOLUTION_PRESENT
from multiprocessing import Process, Pipe
from warnings import warn

from fractionation.init_funcs import *
from fractionation.utilities.plot_utils import *
from fractionation.utilities.file_utils import yaml_to_dict
from fractionation.utilities.data_utils import health_prog_act

input_path = "C:/Users/Anqi/Documents/Software/fractionation/examples/data/"
output_path = "C:/Users/Anqi/Documents/Software/fractionation/examples/output/"
fig_path = output_path + "figures/"

output_prefix = output_path + "ex3_prostate_fmo_"
init_prefix = output_prefix + "init_"
final_prefix = output_prefix + "admm_"

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

# Beam subproblems.
def run_beam_proc(pipe, A, beam_upper, beam_lower, dose_upper, dose_lower, rho_init):
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

	constrs  = fin_upper_constr(b, beam_upper) + fin_lower_constr(b, beam_lower)
	constrs += fin_upper_constr(d, dose_upper) + fin_lower_constr(d, dose_lower)
	prob = Problem(Minimize(obj), constrs)

	# ADMM loop.
	finished = False
	while not finished:
		d_tld.value, u.value = pipe.recv()
		prob.solve(solver = "MOSEK")
		if prob.status not in SOLUTION_PRESENT:
			raise RuntimeError("ADMM: Solver failed on beam subproblem with status {0}".format(prob.status))
		pipe.send((d.value, prob.solver_stats.solve_time))
		finished = pipe.recv()

	# Send final beams and doses.
	pipe.send((b.value, d.value))

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
		constrs += [h[t+1,is_target] == h[t,is_target] - multiply(alpha[t,is_target], d_tld[t,is_target]) \
													   - multiply(2*d_tld[t,is_target] - d_tayl_parm[t,is_target], multiply(beta[t,is_target], d_tayl_parm[t,is_target])) \
													   + gamma[t,is_target] - h_slack[t,is_target]]

		# For OAR, use linear-quadratic model with lossless relaxation.
		constrs += [h[t+1,~is_target] <= h[t,~is_target] - multiply(alpha[t,~is_target], d_tld[t,~is_target]) - multiply(beta[t,~is_target], square(d_tld[t,~is_target])) + gamma[t,~is_target]]
	constrs += [h[1:,0] <= health_upper[:,0], h[1:,1:] >= health_lower[:,1:]]
	constrs += fin_upper_constr(d_tld, dose_upper) + fin_lower_constr(d_tld, dose_lower)

	prob_h = Problem(Minimize(obj), constrs)
	prob_h_dict = {"prob": prob_h, "h": h, "h_slack": h_slack, "d_tld": d_tld, "d_cons_parm": d_cons_parm, "d_tayl_parm": d_tayl_parm}

	# Initialize main loop.
	rho_init = 5.0
	u_init = np.zeros(u.shape)
	try:
		d_init_admm = np.load(init_prefix + "doses.npy")
	except IOError:
		# raise RuntimeError("{0} does not exist. Initializing dose to zero.".format(init_prefix + "doses.npy"))
		warn("{0} does not exist. Initializing dose to zero.".format(init_prefix + "doses.npy"))
		d_init_admm = np.zeros(d_tld.shape)

	# Set up beam subproblem processes.
	pipes = []
	procs = []
	for t in range(T):
		local, remote = Pipe()
		pipes += [local]
		procs += [Process(target=run_beam_proc, args=(remote, A_list[t], beam_upper[t], beam_lower[t], dose_upper[t], dose_lower[t], rho_init))]
		procs[-1].start()

	# Solve using ADMM.
	admm_max_iter = 15   # 100
	eps_abs = 1e-6   # Absolute stopping tolerance.
	eps_rel = 1e-3   # Relative stopping tolerance.

	ccp_max_iter = 15
	ccp_eps = 1e-3

	print("ADMM: Solving dynamic problem...")
	rho.value = rho_init
	u.value = u_init
	d_tld_var_val = d_init_admm
	d_tld_var_val_old = d_init_admm

	k = 0
	solve_time_admm = 0
	r_prim_admm = np.zeros(admm_max_iter)
	r_dual_admm = np.zeros(admm_max_iter)
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
		solve_time_admm = np.max(d_times)   # Take max of all solve times, since subproblems solved in parallel.

		# Solve health subproblem using CCP.
		obj_old = np.inf
		prob_h_dict["d_cons_parm"].value = d_var_val
		prob_h_dict["d_tayl_parm"].value = d_tld_var_val_old   # TODO: What dose point should we linearize PTV health dynamics around?

		for l in range(ccp_max_iter):
			# Solve linearized problem.
			prob_h_dict["prob"].solve(solver = "MOSEK")
			if prob_h_dict["prob"].status not in SOLUTION_PRESENT:
				raise RuntimeError("ADMM CCP: Solver failed on ADMM iteration {0}, CCP iteration {1} with status {2}".format(k, l, prob_h_dict["prob"].status))
			solve_time_admm += prob_h_dict["prob"].solver_stats.solve_time

			# Terminate if change in objective is small.
			obj_diff = obj_old - prob_h_dict["prob"].value
			# print("ADMM CCP Iteration {0}, Objective Difference: {1}".format(l, obj_diff))
			if obj_diff <= ccp_eps:
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

		r_prim_norm = LA.norm(r_prim)
		r_dual_norm = LA.norm(r_dual)
		r_prim_admm[k] = r_prim_norm
		r_dual_admm[k] = r_dual_norm

		# Check stopping criteria.
		eps_prim = eps_abs*np.sqrt(T*K) + eps_rel*np.max([LA.norm(d_var_val), LA.norm(d_tld_var_val)])
		eps_dual = eps_abs*np.sqrt(T*K) + eps_rel*LA.norm(u.value)

		k = k + 1
		finished = (k >= admm_max_iter) or (r_prim_norm <= eps_prim and r_dual_norm <= eps_dual)
		for t in range(T):
			pipes[t].send(finished)

	# Get final beams and doses from beam subproblem.
	bd_update = [pipe.recv() for pipe in pipes]
	b_rows, d_rows = map(list, zip(*bd_update))
	[p.terminate() for p in procs]

	# Save results.
	iters_admm = k
	r_prim_admm = r_prim_admm[:k]
	r_dual_admm = r_dual_admm[:k]
	
	b_admm = np.row_stack(b_rows)
	d_var_val = np.row_stack(d_rows)
	d_admm = (d_var_val + d_tld_var_val)/2.0
	h_admm = prob_h_dict["h"].value
	s_admm = prob_h_dict["h_slack"].value

	# Calculate true objective.
	d_penalty_admm = np.sum(d_admm[:,:-1]**2) + 0.25*np.sum(d_admm[:,-1]**2)
	h_penalty_admm = np.sum(np.maximum(h_admm[1:,is_target], 0)) + 0.25*np.sum(np.maximum(-h_admm[1:,~is_target], 0))
	s_penalty_admm = h_slack_weight*np.sum(s_admm[:,is_target])
	obj_admm = d_penalty_admm + h_penalty_admm + s_penalty_admm

	print("ADMM Results")
	print("Objective:", obj_admm)
	# print("Optimal Dose:", d_admm)
	# print("Optimal Health:", h_admm)
	# print("Optimal Health Slack:", s_admm)
	print("Solve Time:", solve_time_admm)
	print("Iterations:", iters_admm)

	# Save to file.
	np.save(final_prefix + "beams.npy", b_admm)
	np.save(final_prefix + "doses.npy", d_admm)
	np.save(final_prefix + "health.npy", h_admm)
	np.save(final_prefix + "health_slack.npy", s_admm)
	np.save(final_prefix + "residuals_primal.npy", r_prim_admm)
	np.save(final_prefix + "residuals_dual.npy", r_dual_admm)

	# Plot optimal health and dose over time.
	plot_residuals(r_prim_admm, r_dual_admm, semilogy = True)
	plot_treatment(d_admm, stepsize = 10, bounds = (dose_lower, dose_upper), title = "Treatment Dose vs. Time", one_idx = True, 
			   filename = final_prefix + "doses.png")
	plot_health(h_admm, curves = h_curves, stepsize = 10, bounds = (health_lower, health_upper), title = "Health Status vs. Time", 
			  label = "Treated", color = colors[0], one_idx = True, filename = final_prefix + "health.png")

if __name__ == "__main__":
    main()