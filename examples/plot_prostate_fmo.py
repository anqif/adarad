import matplotlib
import matplotlib.pyplot as plt

from adarad.utilities.plot_utils import *
from adarad.utilities.file_utils import yaml_to_dict
from adarad.utilities.data_utils import health_prog_act

data_path = "/home/anqi/Documents/software/adarad/examples/data/"
input_path = "/home/anqi/Documents/software/adarad/examples/output/kona64/"
output_path = "/home/anqi/Documents/papers/adapt_rad_therapy/figures/"

input_prefix = input_path + "ex3_prostate_fmo_full_"
init_prefix = input_prefix + "init_"
ccp_prefix = input_prefix + "ccp_"
admm_prefix = input_prefix + "admm_"
output_prefix = output_path + "ex3_"

def main():
    # Import data.
    patient_bio, patient_rx, visuals = yaml_to_dict(data_path + "ex_prostate_FMO_stanford_full.yml")
    d_init = np.load(init_prefix + "doses.npy")   # Post-initialization heuristic.
    h_init = np.load(init_prefix + "health.npy")
    d_admm = np.load(admm_prefix + "doses.npy")   # Final ADMM output.
    h_admm = np.load(admm_prefix + "health.npy")
    r_primal_admm = np.load(admm_prefix + "primal_residuals.npy")
    r_dual_admm = np.load(admm_prefix + "dual_residuals.npy")

    # Patient data.
    A_list = patient_bio["dose_matrices"]
    alpha = patient_bio["alpha"]
    beta = patient_bio["beta"]
    gamma = patient_bio["gamma"]
    h_0 = patient_bio["health_init"]

    T = len(A_list)
    K = h_0.shape[0]
    dose_lower = patient_rx["dose_constrs"]["lower"]
    dose_upper = patient_rx["dose_constrs"]["upper"]
    health_lower = patient_rx["health_constrs"]["lower"]
    health_upper = patient_rx["health_constrs"]["upper"]

    # Health prognosis.
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    h_prog = health_prog_act(h_0, T, gamma = gamma)

    # Collect comparison curves.
    d_curves = [{"d": d_init, "label": "Initial Plan", "kwargs": {"color": colors[2]}}]   # , "linestyle": "dashed"}}]
    h_curves = [{"h": h_init, "label": "Treated (Initial Plan)", "kwargs": {"color": colors[2]}},   # , "linestyle": "dashed"}}]
                {"h": h_prog, "label": "Untreated", "kwargs": {"color": colors[1]}}]

    # Plot dose and health curves.
    # plot_residuals(r_primal_admm, r_dual_admm, semilogy = True, title = "ADMM Residuals vs. Iteration")
    # plot_treatment(d_admm, curves = d_curves, stepsize = 20, bounds = (dose_lower, dose_upper), title = "Treatment Dose vs. Time",
    #                label = "Treated (Final Plan)", color = colors[0], one_idx = True)
    # plot_health(h_admm, curves = h_curves, stepsize = 20, bounds = (health_lower, health_upper), title="Health Status vs. Time",
    #             label = "Treated (Final Plan)", color = colors[0], one_idx = True)

    plot_residuals(r_primal_admm, r_dual_admm, semilogy=True, filename=output_prefix + "residuals.png")
    # plot_treatment(d_admm, curves = d_curves, stepsize = 20, bounds = (dose_lower, dose_upper), label = "Final Plan",
    #                color = colors[0], one_idx = True, filename = output_prefix + "doses.png", figsize = (16,12))
    plot_treatment(d_admm, curves = d_curves, stepsize = 20, bounds = (np.zeros((T,K)), np.full((T,K), np.inf)),
                   label = "Final Plan", color = colors[0], one_idx = True, filename = output_prefix + "doses.png",
                   figsize = (16,12))
    plot_health(h_admm, curves=h_curves, stepsize=20, bounds=(health_lower, health_upper), label="Treated (Final Plan)",
                color = colors[0], one_idx = True, filename = output_prefix + "health.png", figsize = (16,12))

if __name__ == "__main__":
    main()