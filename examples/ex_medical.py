import matplotlib
matplotlib.use("TKAgg")

from fractionation.quad_funcs import dyn_quad_treat
from fractionation.quad_admm_funcs import dyn_quad_treat_admm
from fractionation.utilities.plot_utils import *
from fractionation.utilities.data_utils import line_integral_mat, health_prog_act

from example_utils import simple_structures, simple_colormap, simple_parms

def main(figpath = "", datapath = ""):
    T = 60     # Length of treatment.
    n_grid = 1000
    offset = 5  # Displacement between beams (pixels).
    n_angle = 20  # Number of angles.
    n_bundle = 50  # Number of beams per angle.
    n = n_angle * n_bundle  # Total number of beams.

    # Display structures on a polar grid.
    x_grid, y_grid, regions = simple_structures(n_grid, n_grid)
    struct_kw = simple_colormap(one_idx=True)
    plot_structures(x_grid, y_grid, regions, title="Anatomical Structures", one_idx=True, **struct_kw)

    # Problem data.
    K = np.unique(regions).size  # Number of structures.
    is_target = np.array([True] + (K - 1) * [False])   # Is the structure a target?
    alpha, beta, gamma = simple_parms(T, fast = True)
    A, angles, offs_vec = line_integral_mat(regions, angles=n_angle, n_bundle=n_bundle, offset=offset)
    A = A / n_grid
    A_list = T * [A]

    # Health prognosis.
    h_init = np.array([10] + (K - 1) * [0])
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    h_prog = health_prog_act(h_init, T, gamma=gamma)
    curves = [{"h": h_prog, "label": "Untreated", "kwargs": {"color": colors[1]}}]

    # Penalty functions.
    w_lo = np.array([0] + (K - 1) * [1])
    w_hi = np.array([1] + (K - 1) * [0])
    rx_health_weights = [w_lo, w_hi]
    rx_health_goal = np.zeros((T, K))
    rx_dose_weights = np.array((K - 1)*[1] + [0.25])
    rx_dose_goal = np.zeros((T, K))
    patient_rx = {"dose_goal": rx_dose_goal, "dose_weights": rx_dose_weights, "health_goal": rx_health_goal,
                  "health_weights": rx_health_weights, "is_target": is_target}

    # Beam constraints.
    beam_upper = np.full((T, n), 1.0)
    patient_rx["beam_constrs"] = {"upper": beam_upper}

    # Dose constraints.
    dose_lower = np.zeros((T, K))
    dose_upper = np.full((T, K), np.inf)
    patient_rx["dose_constrs"] = {"lower": dose_lower, "upper": dose_upper}

    # Health constraints.
    health_lower = np.full((T, K), -np.inf)
    health_upper = np.full((T, K), np.inf)
    patient_rx["health_constrs"] = {"lower": health_lower[:, ~is_target], "upper": health_upper[:, is_target]}

    # Dynamic treatment.
    d_init = np.zeros((T, K))
    res_dynamic = dyn_quad_treat(A_list, alpha, beta, gamma, h_init, patient_rx, d_init=d_init, use_slack=True,
                                 max_iter=15, solver="MOSEK", ccp_verbose=True)
    print("Dynamic Treatment")
    print("Status:", res_dynamic["status"])
    print("Objective:", res_dynamic["obj"])
    print("Solve Time:", res_dynamic["solve_time"])
    print("Iterations:", res_dynamic["num_iters"])

    # Plot dynamic beam, health, and treatment curves.
    plot_beams(res_dynamic["beams"], angles=angles, offsets=offs_vec, n_grid=n_grid, stepsize=1,
               cmap=transp_cmap(plt.cm.Reds, upper=0.5), title="Beam Intensities vs. Time", one_idx=True,
               structures=(x_grid, y_grid, regions), struct_kw=struct_kw)
    plot_health(res_dynamic["health"], curves=curves, stepsize=10, bounds=(health_lower, health_upper),
                title="Health Status vs. Time", label="Treated", color=colors[0], one_idx=True)
    plot_treatment(res_dynamic["doses"], stepsize=10, bounds=(dose_lower, dose_upper), title="Treatment Dose vs. Time",
                   one_idx=True)

if __name__ == '__main__':
    main(figpath = "/home/anqi/Dropbox/Research/Fractionation/Figures/",
		 datapath = "/home/anqi/Documents/software/fractionation/examples/output/")