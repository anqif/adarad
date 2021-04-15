import numpy as np
import cvxpy
from cvxpy import *

from adarad.init_funcs import *
from adarad.utilities.data_utils import line_integral_mat, health_prog_act
from example_utils import simple_structures, simple_colormap

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
K = np.unique(regions).size   # Number of structures.

A, angles, offs_vec = line_integral_mat(regions, angles = n_angle, n_bundle = n_bundle, offset = offset)
A = A/n_grid
A_list = T*[A]

alpha = np.array(T*[[0.01, 0.50, 0.25, 0.15, 0.005]])
beta = np.array(T*[[0.001, 0.05, 0.025, 0.015, 0.0005]])
gamma = np.array(T*[[0.05, 0, 0, 0, 0]])
h_init = np.array([1] + (K-1)*[0])
is_target = np.array([True] + (K-1)*[False])

# Static beam problem in CVXPY.
# Define variables.
b = Variable((n,), nonneg=True)
d = A @ b
h_ptv = h_init[0] - alpha[t_s,0] * d[0] + gamma[t_s,0]
h_oar = h_init[1:] - multiply(alpha[t_s,1:], d[1:]) - multiply(beta[t_s,1:], square(d[1:])) + gamma[t_s,1:]
h = hstack([h_ptv, h_oar])

# Form objective.
d_penalty = sum_squares(d[:-1]) + 0.25*square(d[-1])
h_penalty_ptv = pos(h_ptv)
h_penalty_oar = sum(neg(h_oar))
obj = d_penalty + h_penalty_ptv + h_penalty_oar

# Solve problem in CVXPY.
prob_nocon = Problem(Minimize(obj))
prob_nocon.solve(solver = "MOSEK")

print("CVXPY Results")
print("Objective:", prob_nocon.value)
print("Optimal Dose:", d.value)
print("Optimal Health:", h.value)

# Static beam problem in AdaRad.
# Prescription.
w_lo = np.array([0] + (K-1)*[1])
w_hi = np.array([1] + (K-1)*[0])
rx_health_weights = [w_lo, w_hi]
rx_health_goal = np.zeros((T,K))
rx_dose_weights = np.array([1, 1, 1, 1, 0.25])
rx_dose_goal = np.zeros((T,K))
patient_rx = {"is_target": is_target, "dose_goal": rx_dose_goal, "dose_weights": rx_dose_weights,
			  "health_goal": rx_health_goal, "health_weights": rx_health_weights}

# Solve problem in AdaRad.
prob_ada_nocon, b_ada, h_ada, d_ada, h_slack_ada = build_stat_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx, t_static = t_s, use_slack = False)
prob_ada_nocon.solve(solver = "MOSEK")

print("AdaRad Results")
print("Objective:", prob_ada_nocon.value)
print("Optimal Dose:", d_ada.value)
print("Optimal Health:", h_ada.value)

print("Compare CVXPY with AdaRad Results")
print("Objective SSE:", (prob_nocon.value - prob_ada_nocon.value)**2)
print("Optimal Beam SSE:", np.linalg.norm(b.value - b_ada.value))
print("Optimal Dose SSE:", np.linalg.norm(d.value - d_ada.value))
print("Optimal Health SSE:", np.linalg.norm(h.value - h_ada.value))

# Beam constraints.
beam_upper = np.full((T,n), 1.0)
patient_rx["beam_constrs"] = {"upper": beam_upper}

# Dose constraints.
dose_lower = np.zeros((T,K))
dose_upper = np.full((T,K), 20)
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

# Static beam problem with constraints in CVXPY.
# TODO: Change how bounds are calculated for static problem, e.g., h_ptv <= H_{T0}.
constrs = [b <= beam_upper[t_s,:], d <= dose_upper[t_s,:], d >= dose_lower[t_s,:], h_ptv <= health_upper[t_s,0], h_oar >= health_lower[t_s,1:]]
prob_con = Problem(Minimize(obj), constrs)
prob_con.solve(solver = "MOSEK")

print("CVXPY Results with Constraints")
print("Objective:", prob_con.value)
print("Optimal Dose:", d.value)
print("Optimal Health:", h.value)

# Static beam problem with constraints in AdaRad.
patient_rx["health_constrs"] = {"lower": health_lower, "upper": health_upper}
prob_ada_con, b_ada, h_ada, d_ada, h_slack_ada = build_stat_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx, t_static = t_s, use_slack = False)
prob_ada_con.solve(solver = "MOSEK")

print("AdaRad Results with Constraints")
print("Objective:", prob_ada_con.value)
print("Optimal Dose:", d_ada.value)
print("Optimal Health:", h_ada.value)

print("Compare CVXPY with AdaRad Results")
print("Objective SSE:", (prob_con.value - prob_ada_con.value)**2)
print("Optimal Beam SSE:", np.linalg.norm(b.value - b_ada.value))
print("Optimal Dose SSE:", np.linalg.norm(d.value - d_ada.value))
print("Optimal Health SSE:", np.linalg.norm(h.value - h_ada.value))

# TODO: Plot CVXPY and AdaRad results.