import numpy as np
import matplotlib
matplotlib.use("TKAgg")

from adarad.quad_funcs import dyn_quad_treat
from adarad.quad_admm_funcs import dyn_quad_treat_admm
from adarad.utilities.plot_utils import *
from adarad.utilities.data_utils import line_integral_mat, health_prog_act

from example_utils import simple_structures, simple_colormap

def main(figpath = "", datapath = ""):
    T_list = [20, 40, 100, 150]  # Length of treatment.
    T_rec_list = [5, 10, 25, 40]   # Length of recovery stage.
    n_grid = 1000
    offset = 5  # Displacement between beams (pixels).
    n_angle = 20  # Number of angles.
    n_bundle = 50  # Number of beams per angle.
    n = n_angle * n_bundle  # Total number of beams.

    # Form anatomical structures on a Cartesian grid.
    x_grid, y_grid, regions = simple_structures(n_grid, n_grid)

    # Problem data.
    K = np.unique(regions).size  # Number of structures.
    A, angles, offs_vec = line_integral_mat(regions, angles=n_angle, n_bundle=n_bundle, offset=offset)
    A = A / n_grid

    alpha_vec = [0.01, 0.50, 0.25, 0.15, 0.005]
    beta_vec = [0.001, 0.05, 0.025, 0.015, 0.0005]
    gamma_vec = [0.05, 0, 0, 0, 0]
    h_init = np.array([1] + (K - 1) * [0])
    is_target = np.array([True] + (K - 1) * [False])

    w_lo = np.array([0] + (K - 1) * [1])
    w_hi = np.array([1] + (K - 1) * [0])
    rx_health_weights = [w_lo, w_hi]
    rx_dose_weights = np.array([1, 1, 1, 1, 0.25])

    T_list_len = len(T_list)
    solve_times = np.zeros((T_list_len, 2))
    for i in range(T_list_len):
        T = T_list[i]
        T_rec = T_rec_list[i]

        A_list = T * [A]
        alpha = np.array(T * [alpha_vec])
        beta  = np.array(T * [beta_vec])
        gamma = np.array(T * [gamma_vec])

        # Penalty functions.
        rx_health_goal = np.zeros((T, K))
        rx_dose_goal = np.zeros((T, K))
        patient_rx = {"is_target": is_target, "dose_goal": rx_dose_goal, "dose_weights": rx_dose_weights,
                      "health_goal": rx_health_goal, "health_weights": rx_health_weights}

        # Beam constraints.
        beam_upper = np.full((T, n), 1.0)
        patient_rx["beam_constrs"] = {"upper": beam_upper}

        # Dose constraints.
        dose_lower = np.zeros((T, K))
        dose_upper = np.full((T, K), 20)  # Upper bound on doses.
        patient_rx["dose_constrs"] = {"lower": dose_lower, "upper": dose_upper}

        # Health constraints.
        health_lower = np.full((T, K), -np.inf)
        health_upper = np.full((T, K), np.inf)
        # health_lower[:,1] = -1.0     # Lower bound on OARs.
        # health_lower[:,2] = -2.0
        # health_lower[:,3] = -2.0
        # health_lower[:,4] = -3.0
        health_upper[:(T-T_rec), 0] = 2.0   # Upper bound on PTV for t = 1,...,15.
        health_upper[(T-T_rec):, 0] = 0.05  # Upper bound on PTV for t = 16,...,20.
        patient_rx["health_constrs"] = {"lower": health_lower, "upper": health_upper}

        # Dynamic treatment.
        print("\nTreatment Length:", T)
        res_dynamic = dyn_quad_treat(A_list, alpha, beta, gamma, h_init, patient_rx, use_slack=True, slack_weight=1e4,
                                     max_iter=15, solver="MOSEK", ccp_verbose=True)
        print("Dynamic Treatment")
        print("Status:", res_dynamic["status"])
        print("Objective:", res_dynamic["obj"])
        print("Solve Time:", res_dynamic["solve_time"])
        print("Iterations:", res_dynamic["num_iters"])

        # Dynamic treatment with ADMM.
        res_dynamic_admm = dyn_quad_treat_admm(A_list, alpha, beta, gamma, h_init, patient_rx, use_slack=True, slack_weight=1e4,
                                          ccp_max_iter=15, solver="MOSEK", rho=5, admm_max_iter=50, admm_verbose=True)
        print("Dynamic Treatment with ADMM")
        print("Status:", res_dynamic_admm["status"])
        print("Objective:", res_dynamic_admm["obj"])
        print("Solve Time:", res_dynamic_admm["solve_time"])
        print("Iterations:", res_dynamic_admm["num_iters"])

        # Check if final objective value is similar.
        obj_diff = np.abs(res_dynamic["obj"] - res_dynamic_admm["obj"])
        print("Difference in Objective:", obj_diff)
        print("Normalized Difference in Objective:", obj_diff/res_dynamic["obj"])

        # Save total solve times.
        solve_times[i,0] = res_dynamic["solve_time"]
        solve_times[i,1] = res_dynamic_admm["solve_time"]

    # Plot and compare solve times.
    fig = plt.figure()
    fig.set_size_inches(10, 8)
    plt.plot(T_list, solve_times)
    plt.legend(labels = ["SCO", "ADMM"])
    # plt.title("Solve Time vs. Treatment Length")
    plt.xlabel("Treatment Length (T)")
    plt.ylabel("Solve Time (s)")
    plt.show()
    fig.savefig(figpath + "solve_time_sessions.png", bbox_inches="tight", dpi=300)

    # Plot ratio of standard:ADMM solve time.
    solve_ratios = solve_times[:, 0] / solve_times[:, 1]
    fig = plt.figure()
    fig.set_size_inches(10, 8)
    plt.plot(T_list, solve_ratios)
    # plt.title("Solve Ratio vs. Treatment Length")
    plt.xlabel("Treatment Length (T)")
    plt.ylabel("Solve Time Ratio (SCO:ADMM)")
    plt.show()
    fig.savefig(figpath + "solve_ratio_sessions.png", bbox_inches="tight", dpi=300)

    print("\nSolve Times")
    for i in range(T_list_len):
        print("T = {0}, \tSCO = {1}, ADMM = {2}, \tRatio = {3}".format(T_list[i], solve_times[i,0], solve_times[i,1], solve_ratios[i]))

if __name__ == '__main__':
    main(figpath = "/home/anqi/Dropbox/Research/Fractionation/Figures/", \
		 datapath = "/home/anqi/Documents/software/adarad/examples/output/")