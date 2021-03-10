import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TKAgg")

import cvxpy
from cvxpy import *
from cvxpy.settings import SOLUTION_PRESENT

from fractionation.init_funcs import *
from fractionation.utilities.plot_utils import *
from fractionation.utilities.data_utils import line_integral_mat, health_prog_act

from example_utils import simple_structures, simple_colormap

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

# Stage 1: Static beam problem.
# Define variables.
b = Variable((n,), nonneg=True)
d = A @ b
h_ptv = h_init[0] - alpha[t_s,0] * d[0] + gamma[t_s,0]
h_oar = h_init[1:] - multiply(alpha[t_s,1:], d[1:]) - multiply(beta[t_s,1:], square(d[1:])) + gamma[t_s,1:]
h = hstack([h_ptv, h_oar])

# Form objective.
d_penalty = sum_squares(d[:-1]) + 0.25*square(d[-1])
h_penalty_ptv = pos(h_ptv)
h_penalty_oar = 0.25*sum(neg(h_oar))
h_penalty = h_penalty_ptv + h_penalty_oar

# Add slack for constraints.
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
constrs = [h_ptv <= health_upper[-1,0], h_oar >= health_lower[-1,1:] - h_lo_slack]

# Solve problem.
print("Stage 1: Solving problem...")
prob_1 = Problem(Minimize(obj), constrs)
prob_1.solve(solver = "MOSEK")
if prob_1.status not in SOLUTION_PRESENT:
	raise RuntimeError("Stage 1: Solver failed with status {0}".format(prob_1.status))
solve_time = prob_1.solver_stats.solve_time

print("Stage 1 Results")
print("Objective:", prob_1.value)
print("Optimal Dose:", d.value)
print("Optimal Health:", h.value)
print("Solve Time:", prob_1.solver_stats.solve_time)

# Save results.
b_static = b.value   # Save optimal static beams for stage 2.
d_stage_1 = d.value
h_stage_1 = h.value

# Plot optimal dose and health per structure.
xlim_eps = 0.5
plt.bar(range(K), d_stage_1, width = 0.8)
plt.step(*form_step_xy(np.arange(K), dose_lower[-1,:], buf = 0.5), where = "mid", lw = 1, ls = "--", color = colors[1])
plt.step(*form_step_xy(np.arange(K), dose_upper[-1,:], buf = 0.5), where = "mid", lw = 1, ls = "--", color = colors[1])
plt.title("Treatment Dose vs. Structure")
plt.xlim(-xlim_eps, K-1+xlim_eps)
plt.show()

plt.bar(range(K), h_stage_1, width=0.8)
health_bounds_fin = np.concatenate(([health_upper[-1,0]], health_lower[-1,1:]))
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
h_slack = Variable((T,), nonneg=True)      # Slack in approximation.

# Form objective.
d_penalty = sum_squares(d[:,:-1]) + 0.25*sum_squares(d[:,-1])
h_penalty = sum(pos(h[1:,0])) + 0.25*sum(neg(h[1:,1:]))
s_penalty = h_slack_weight*sum(h_slack)
obj = d_penalty + h_penalty + s_penalty

# Health dynamics.
constrs = [h[0] == h_init]
for t in range(T):
	# For PTV, use simple linear model (beta_t = 0).
	constrs += [h[t+1,0] == h[t,0] - alpha[t,0]*d[t,0] + gamma[t,0] - h_slack[t]]

	# For OAR, use linear-quadratic model with lossless relaxation.
	constrs += [h[t+1,1:] <= h[t,1:] - multiply(alpha[t,1:], d[t,1:]) - multiply(beta[t,1:], square(d[t,1:])) + gamma[t,1:]]

# Additional constraints.
constrs += [b <= np.min(beam_upper, axis=0), d <= dose_upper, d >= dose_lower, h[1:,0] <= health_upper[:,0], h[1:,1:] >= health_lower[:,1:]]

# Warm start.
u.value = 1

# Solve problem.
print("Stage 2: Solving initial problem...")
prob_2a = Problem(Minimize(obj), constrs)
prob_2a.solve(solver = "MOSEK", warm_start = True)
if prob_2a.status not in SOLUTION_PRESENT:
	raise RuntimeError("Stage 2 Initialization: Solver failed with status {0}".format(prob_2a.status))
solve_time += prob_2a.solver_stats.solve_time

print("Stage 2 Initialization")
print("Objective:", prob_2a.value)
print("Optimal Beam Weight:", u.value)
# print("Optimal Dose:", d.value)
# print("Optimal Health:", h.value)
# print("Optimal Health Slack:", h_slack.value)
print("Solve Time:", prob_2a.solver_stats.solve_time)

# Save results.
u_stage_2_init = u.value
d_stage_2_init = d.value   # Save optimal doses derived from constant factor for stage 2b.
h_stage_2_init = h.value
s_stage_2_init = h_slack.value

# Stage 2b: Dynamic scaling problem with time-varying factors.
# Define variables.
u = Variable((T,), nonneg=True)
b = vstack([u[t]*b_static for t in range(T)])
d = vstack([A_list[t] @ b[t] for t in range(T)])
h = Variable((T+1,K))

# Used in Taylor expansion of PTV health dynamics.
h_slack_weight = 1e4
h_slack = Variable((T,), nonneg=True)      # Slack in approximation.
d_parm = Parameter(d.shape, nonneg=True)   # Dose point around which to linearize.

# Form objective.
d_penalty = sum_squares(d[:,:-1]) + 0.25*sum_squares(d[:,-1])
h_penalty = sum(pos(h[1:,0])) + 0.25*sum(neg(h[1:,1:]))
s_penalty = h_slack_weight*sum(h_slack)
obj = d_penalty + h_penalty + s_penalty

# Health dynamics.
constrs = [h[0] == h_init]
for t in range(T):
	# For PTV, use first-order Taylor expansion of dose around d_parm.
	constrs += [h[t+1,0] == h[t,0] - alpha[t,0]*d[t,0] - (2*d[t,0] - d_parm[t,0])*beta[t,0]*d_parm[t,0] + gamma[t,0] - h_slack[t]]

	# For OAR, use linear-quadratic model with lossless relaxation.
	constrs += [h[t+1,1:] <= h[t,1:] - multiply(alpha[t,1:], d[t,1:]) - multiply(beta[t,1:], square(d[t,1:])) + gamma[t,1:]]

# Additional constraints.
constrs += [b <= beam_upper, d <= dose_upper, d >= dose_lower, h[1:,0] <= health_upper[:,0], h[1:,1:] >= health_lower[:,1:]]
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

print("Stage 2 Results")
print("Objective:", prob_2b.value)
print("Optimal Beam Weight:", u.value)
# print("Optimal Dose:", d.value)
# print("Optimal Health:", h.value)
# print("Optimal Health Slack:", h_slack.value)
print("Solve Time:", prob_2b.solver_stats.solve_time)

# Save results.
b_stage_2 = b.value
d_stage_2 = d.value
h_stage_2 = h.value
s_stage_2 = h_slack.value

# Plot optimal dose and health over time.
plot_treatment(d_stage_2, stepsize = 10, bounds = (dose_lower, dose_upper),
			title = "Treatment Dose vs. Time", color = colors[0], one_idx = True)
plot_health(h_stage_2, curves = h_curves, stepsize = 10, bounds = (health_lower, health_upper),
			title = "Health Status vs. Time", label = "Treated", color = colors[0], one_idx = True)

# Main Stage: Dynamic optimal control problem.
# Define variables.
b = Variable((T,n), nonneg=True)
d = vstack([A_list[t] @ b[t] for t in range(T)])
h = Variable((T+1,K))

# Used in Taylor expansion of PTV health dynamics.
h_slack_weight = 1e4
h_slack = Variable((T,), nonneg=True)      # Slack in approximation.
d_parm = Parameter(d.shape, nonneg=True)   # Dose point around which to linearize.

# Form objective.
d_penalty = sum_squares(d[:,:-1]) + 0.25*sum_squares(d[:,-1])
h_penalty = sum(pos(h[1:,0])) + 0.25*sum(neg(h[1:,1:]))
s_penalty = h_slack_weight*sum(h_slack)
obj = d_penalty + h_penalty + s_penalty

# Health dynamics.
constrs = [h[0] == h_init]
for t in range(T):
	# For PTV, use first-order Taylor expansion of dose around d_parm.
	constrs += [h[t+1,0] == h[t,0] - alpha[t,0]*d[t,0] - (2*d[t,0] - d_parm[t,0])*beta[t,0]*d_parm[t,0] + gamma[t,0] - h_slack[t]]

	# For OAR, use linear-quadratic model with lossless relaxation.
	constrs += [h[t+1,1:] <= h[t,1:] - multiply(alpha[t,1:], d[t,1:]) - multiply(beta[t,1:], square(d[t,1:])) + gamma[t,1:]]

# Additional constraints.
constrs += [b <= beam_upper, d <= dose_upper, d >= dose_lower, h[1:,0] <= health_upper[:,0], h[1:,1:] >= health_lower[:,1:]]
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

print("Main Stage Results")
print("Objective:", prob_main.value)
# print("Optimal Dose:", d.value)
# print("Optimal Health:", h.value)
# print("Optimal Health Slack:", h_slack.value)
print("Solve Time:", prob_main.solver_stats.solve_time)
print("Total Solve Time:", solve_time)

# Save results.
b_main = b.value
d_main = d.value
h_main = h.value

# Plot optimal beam, health, and dose over time.
plot_beams(b_main, angles = angles, offsets = offs_vec, n_grid = n_grid, stepsize = 1, cmap = transp_cmap(plt.cm.Reds, upper = 0.5), 
		   title = "Beam Intensities vs. Time", one_idx = True, structures = (x_grid, y_grid, regions), struct_kw = struct_kw)
plot_treatment(d_main, stepsize = 10, bounds = (dose_lower, dose_upper), title = "Treatment Dose vs. Time", one_idx = True)
plot_health(h_main, curves = h_curves, stepsize = 10, bounds = (health_lower, health_upper),
			title = "Health Status vs. Time", label = "Treated", color = colors[0], one_idx = True)
