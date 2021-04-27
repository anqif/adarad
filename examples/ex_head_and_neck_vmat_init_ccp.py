import numpy as np
import matplotlib
matplotlib.use("TKAgg")
from time import time

from cvxpy.settings import SOLUTION_PRESENT

from adarad.init_funcs import *
from adarad.quadratic.dyn_quad_prob import build_dyn_quad_prob
from adarad.utilities.plot_utils import *
from adarad.utilities.file_utils import yaml_to_dict
from adarad.utilities.data_utils import health_prog_act

SHOW_PLOTS = True

# input_path = "/home/anqi/Documents/software/adarad/examples/data/"
# output_path = "/home/anqi/Documents/software/adarad/examples/output/"
input_path = "/home/anqif/adarad/examples/data/"
output_path = "/home/anqif/adarad/examples/output/"
fig_path = output_path + "figures/"

output_prefix = output_path + "ex4_head_and_neck_vmat_"
init_prefix = output_prefix + "init_"
final_prefix = output_prefix + "ccp_"

fig_prefix = fig_path + "ex4_head_and_neck_vmat_"
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
    # Import data.
    patient_bio, patient_rx, visuals = yaml_to_dict(input_path + "ex_head_and_neck_VMAT_TROT_std.yml")

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

    # Treatment parameters.
    dose_weights = np.ones(K)
    dose_weights[10] = 0.01   # Body voxels.
    dose_weights = 0.01*dose_weights
    h_tayl_slack_weight = 100   # 1e4
    h_lo_slack_weight = 1/num_oar
    patient_rx_ada = {"is_target": is_target,
                      "dose_goal": np.zeros((T, K)),
                      "dose_weights": dose_weights,
                      "health_goal": np.zeros((T, K)),
                      "health_weights": [np.array([0] + (K-1)*[num_ptv/num_oar]), np.array([1] + (K-1)*[0])],
                      "beam_constrs": {"lower": beam_lower, "upper": beam_upper},
                      "dose_constrs": {"lower": dose_lower, "upper": dose_upper},
                      "health_constrs": {"lower": health_lower, "upper": health_upper}}

    # Stage 1: Static beam problem.
    prob_1, b_1, h_1, d_1, h_actual_1, h_slack_1 = \
        build_stat_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx_ada, t_static=t_s, slack_oar_weight=h_lo_slack_weight)

    print("Stage 1: Solving problem...")
    start = time()
    prob_1.solve(solver="MOSEK")
    end = time()

    if prob_1.status not in SOLUTION_PRESENT:
        raise RuntimeError("AdaRad Stage 1: Solver failed with status {0}".format(prob_1.status))

    b_static = b_1.value  # Save optimal static beams for stage 2.
    d_static = np.vstack([A_list[t] @ b_static for t in range(T)])
    d_stage_1 = d_1.value
    # h_stage_1 = h.value
    h_stage_1 = h_init - alpha[t_s]*d_stage_1 - beta[t_s]*d_stage_1**2 + gamma[t_s]
    prob_1_setup_time = 0 if prob_1.solver_stats.setup_time is None else prob_1.solver_stats.setup_time
    prob_1_solve_time = prob_1.solver_stats.solve_time
    prob_1_runtime = end - start

    print("Stage 1 Results")
    print("Objective:", prob_1.value)
    print("Optimal Dose:", d_stage_1)
    print("Optimal Beam (Max):", np.max(b_static))
    print("Optimal Health:", h_stage_1)
    print("Setup Time:", prob_1_setup_time)
    print("Solve Time:", prob_1_solve_time)

    # Plot optimal dose and health per structure.
    xlim_eps = 0.5
    plt.bar(range(K), d_stage_1, width=0.8)
    plt.step(*form_step_xy(np.arange(K), dose_lower[-1, :], buf=0.5), where="mid", lw=1, ls="--", color=colors[1])
    plt.step(*form_step_xy(np.arange(K), dose_upper[-1, :], buf=0.5), where="mid", lw=1, ls="--", color=colors[1])
    plt.title("Treatment Dose vs. Structure")
    plt.xlim(-xlim_eps, K - 1 + xlim_eps)
    if SHOW_PLOTS:
        plt.show()

    health_bounds_fin = np.zeros(K)
    health_bounds_fin[is_target] = health_upper[-1, is_target]
    health_bounds_fin[~is_target] = health_lower[-1, ~is_target]
    plt.bar(range(K), h_stage_1, width=0.8)
    plt.step(*form_step_xy(np.arange(K), health_bounds_fin, buf=0.5), where="mid", lw=1, ls="--", color=colors[1])
    plt.title("Health Status vs. Structure")
    plt.xlim(-xlim_eps, K - 1 + xlim_eps)
    if SHOW_PLOTS:
        plt.show()

    prob_2a, u_2a, b_2a, h_2a, d_2a, h_lin_dyn_slack_2a, h_lin_bnd_slack_2a = \
        build_scale_lin_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx_ada, b_static, use_dyn_slack=True,
                                  slack_dyn_weight=h_tayl_slack_weight, use_bnd_slack=True, slack_bnd_weight=h_lo_slack_weight)
    print("Stage 2: Solving initial problem...")

    start = time()
    prob_2a.solve(solver="MOSEK")
    end = time()

    if prob_2a.status not in SOLUTION_PRESENT:
        raise RuntimeError("AdaRad Stage 2a: Solver failed with status {0}".format(prob_2a.status))

    # Save results.
    u_stage_2_init = u_2a.value
    d_stage_2_init = d_2a.value  # Save optimal doses derived from constant factor for stage 2b.
    # h_stage_2_init = h.value
    h_stage_2_init = health_prog_act(h_init, T, alpha, beta, gamma, d_stage_2_init, is_target)
    s_stage_2_init = h_lin_dyn_slack_2a.value
    prob_2_init_setup_time = 0 if prob_2a.solver_stats.setup_time is None else prob_2a.solver_stats.setup_time
    prob_2_init_solve_time = prob_2a.solver_stats.solve_time
    prob_2_init_runtime = end - start

    print("Stage 2 Initialization")
    print("Objective:", prob_2a.value)
    print("Optimal Beam Weight:", u_stage_2_init)
    # print("Optimal Dose:", d_stage_2_init)
    # print("Optimal Health:", h_stage_2_init)
    # print("Optimal Health Slack:", s_stage_2_init)
    print("Setup Time:", prob_2_init_setup_time)
    print("Solve Time:", prob_2_init_solve_time)

    # Plot optimal dose and health over time.
    plot_treatment(d_stage_2_init, stepsize=10, bounds=(dose_lower, dose_upper), title="Treatment Dose vs. Time",
                   color=colors[0], one_idx=True, show=SHOW_PLOTS)
    plot_health(h_stage_2_init, curves=h_curves, stepsize=10, bounds=(health_lower, health_upper),
                title="Health Status vs. Time", label="Treated", color=colors[0], one_idx=True, show=SHOW_PLOTS)

    # Stage 2b: Dynamic scaling problem with time-varying factors.
    ccp_max_iter = 20
    ccp_eps = 1e-3
    prob_2b, u_2b, b_2b, h_2b, d_2b, d_parm_2b, h_dyn_slack_2b, h_bnd_slack_2b = \
        build_scale_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx_ada, b_static, use_dyn_slack=True,
                              slack_dyn_weight=h_tayl_slack_weight, use_bnd_slack=True, slack_bnd_weight=h_lo_slack_weight)
    print("Stage 2: Solving dynamic problem with CCP...")
    result_2b = ccp_solve(prob_2b, d_2b, d_parm_2b, d_stage_2_init, h_dyn_slack_2b, max_iter=ccp_max_iter,
                          ccp_eps=ccp_eps, solver="MOSEK", warm_start=True)
    if result_2b["status"] not in SOLUTION_PRESENT:
        raise RuntimeError("Stage 2b: CCP solve failed with status {0}".format(result_2b["status"]))

    # Save results.
    u_stage_2 = u_2b.value
    b_stage_2 = b_2b.value
    d_stage_2 = d_2b.value
    # h_stage_2 = h.value
    h_stage_2 = health_prog_act(h_init, T, alpha, beta, gamma, d_stage_2, is_target)
    s_stage_2 = h_dyn_slack_2b.value
    prob_2b_setup_time = result_2b["setup_time"]
    prob_2b_solve_time = result_2b["solve_time"]
    prob_2b_runtime = result_2b["total_time"]

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
    setup_time_init = prob_1_setup_time + prob_2_init_setup_time + prob_2b_setup_time
    solve_time_init = prob_1_solve_time + prob_2_init_solve_time + prob_2b_solve_time
    runtime_init = prob_1_runtime + prob_2_init_runtime + prob_2b_runtime
    print("Total Setup Time:", setup_time_init)
    print("Total Solve Time:", solve_time_init)
    print("Total (Setup + Solve) Time:", setup_time_init + solve_time_init)
    print("Total Runtime:", runtime_init)

    # Save to file.
    np.save(init_prefix + "beams.npy", b_stage_2)
    np.save(init_prefix + "doses.npy", d_stage_2)
    np.save(init_prefix + "health.npy", h_stage_2)
    np.save(init_prefix + "health_slack.npy", s_stage_2)

    # Plot optimal dose and health over time.
    plot_treatment(d_stage_2, stepsize=10, bounds=(dose_lower, dose_upper), title="Treatment Dose vs. Time",
                   color=colors[0], one_idx=True, filename=init_fig_prefix + "doses.png", show=SHOW_PLOTS)
    plot_health(h_stage_2, curves=h_curves, stepsize=10, bounds=(health_lower, health_upper), title="Health Status vs. Time",
                label="Treated", color=colors[0], one_idx=True, filename=init_fig_prefix + "health.png", show=SHOW_PLOTS)

    # raise RuntimeError("Finished Initialization")

    # Initial dose point of main stage (CCP).
    d_init_main = d_stage_2

    # Main Stage: Dynamic optimal control problem.
    prob_main, b_main, h_main, d_main, d_parm_main, h_dyn_slack_main = \
        build_dyn_quad_prob(A_list, alpha, beta, gamma, h_init, patient_rx_ada, use_slack=True, slack_weight=h_tayl_slack_weight)
    print("Main Stage: Solving dynamic problem with CCP...")
    result_main = ccp_solve(prob_main, d_main, d_parm_main, d_init_main, h_dyn_slack_main, ccp_verbose=True, max_iter=ccp_max_iter,
                            ccp_eps=ccp_eps, solver="MOSEK", warm_start=True)
    if result_main["status"] not in SOLUTION_PRESENT:
        raise RuntimeError("Main Stage: CCP solve failed with status {0}".format(result_main["status"]))

    # Save results.
    b_main_stage = b_main.value
    d_main_stage = d_main.value
    # h_main_stage = h_main.value
    h_main_stage = health_prog_act(h_init, T, alpha, beta, gamma, d_main_stage, is_target)
    s_main_stage = h_dyn_slack_main.value
    prob_main_setup_time = result_main["setup_time"]
    prob_main_solve_time = result_main["solve_time"]
    prob_main_runtime = result_main["total_time"]

    print("Main Stage Results")
    print("Objective:", prob_main.value)
    # print("Optimal Dose:", d_main_stage)
    # print("Optimal Health:", h_main_stage)
    # print("Optimal Health Slack:", s_main_stage)
    print("Setup Time:", prob_main_setup_time)
    print("Solve Time:", prob_main_solve_time)
    print("Runtime:", prob_main_runtime)

    print("\nSolver Stats: All Stages")
    setup_time_total = setup_time_init + prob_main_setup_time
    solve_time_total = solve_time_init + prob_main_solve_time
    runtime_total = runtime_init + prob_main_runtime
    print("Total Setup Time:", setup_time_total)
    print("Total Solve Time:", solve_time_total)
    print("Total (Setup + Solve) Time:", setup_time_total + solve_time_total)
    print("Total Runtime:", runtime_total)

    # Save to file.
    np.save(final_prefix + "beams.npy", b_main_stage)
    np.save(final_prefix + "doses.npy", d_main_stage)
    np.save(final_prefix + "health.npy", h_main_stage)
    np.save(final_prefix + "health_slack.npy", s_main_stage)

    # Plot optimal dose and health over time.
    plot_treatment(d_main_stage, stepsize=10, bounds=(dose_lower, dose_upper), title="Treatment Dose vs. Time", one_idx=True,
                   filename=final_fig_prefix + "doses.png", show=SHOW_PLOTS)
    plot_health(h_main_stage, curves=h_curves, stepsize=10, bounds=(health_lower, health_upper), title="Health Status vs. Time",
                label="Treated", color=colors[0], one_idx=True, filename=final_fig_prefix + "health.png", show=SHOW_PLOTS)

if __name__ == "__main__":
    main()
