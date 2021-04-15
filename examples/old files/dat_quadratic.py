import numpy as np
import matplotlib
matplotlib.use("TKAgg")
import pickle

from adarad.utilities.data_utils import line_integral_mat, health_prog_act
from adarad.utilities.plot_utils import plot_structures

from example_utils import simple_structures, simple_colormap

def main(datapath=""):
    T = 20  # Length of treatment.
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
    A, angles, offs_vec = line_integral_mat(regions, angles=n_angle, n_bundle=n_bundle, offset=offset)
    A = A / n_grid
    A_list = T * [A]

    # Save structures and dose matrix.
    np.save(datapath + "cardioid_5_structs_1000_beams-dose_matrix.npy", A)
    np.save(datapath + "cardioid_5_structs_1000_beams-angles.npy", angles)
    np.save(datapath + "cardioid_5_structs_1000_beams-offsets.npy", offs_vec)

    struct_dict = {"x_grid": x_grid, "y_grid": y_grid, "regions": regions}
    with open(datapath + "cardioid_5_structs_1000_beams-regions.p", "wb") as fp:
        pickle.dump(struct_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    alpha = np.array(T * [[0.01, 0.50, 0.25, 0.15, 0.005]])
    beta = np.array(T * [[0.001, 0.05, 0.025, 0.015, 0.0005]])
    gamma = np.array(T * [[0.05, 0, 0, 0, 0]])
    h_init = np.array([1] + (K - 1) * [0])

    # Health prognosis.
    prop_cycle = matplotlib.pyplot.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    h_prog = health_prog_act(h_init, T, gamma=gamma)
    curves = [{"h": h_prog, "label": "Untreated", "kwargs": {"color": colors[1]}}]

    # Penalty functions.
    w_lo = np.array([0] + (K - 1) * [1])
    w_hi = np.array([1] + (K - 1) * [0])
    rx_health_weights = [w_lo, w_hi]
    rx_health_goal = np.zeros((T, K))
    rx_dose_weights = np.array([1, 1, 1, 1, 0.25])
    rx_dose_goal = np.zeros((T, K))
    patient_rx = {"dose_goal": rx_dose_goal, "dose_weights": rx_dose_weights,
                  "health_goal": rx_health_goal, "health_weights": rx_health_weights}

    # Beam constraints.
    beam_upper = np.full((T, n), 1.0)
    # beam_upper = np.full((T,n), np.inf)
    patient_rx["beam_constrs"] = {"upper": beam_upper}

    # Dose constraints.
    dose_lower = np.zeros((T, K))
    dose_upper = np.full((T, K), 20)  # Upper bound on doses.
    # dose_upper = np.full((T,K), np.inf)
    patient_rx["dose_constrs"] = {"lower": dose_lower, "upper": dose_upper}

    # Health constraints.
    health_lower = np.full((T, K), -np.inf)
    health_upper = np.full((T, K), np.inf)
    health_lower[:, 1] = -1.0  # Lower bound on OARs.
    health_lower[:, 2] = -2.0
    health_lower[:, 3] = -2.0
    health_lower[:, 4] = -3.0
    health_upper[:15, 0] = 2.0  # Upper bound on PTV for t = 1,...,15.
    health_upper[15:, 0] = 0.05  # Upper bound on PTV for t = 16,...,20.

    is_target = np.array([True] + (K - 1) * [False])
    patient_rx["is_target"] = is_target
    patient_rx["health_constrs"] = {"lower": health_lower[:, ~is_target], "upper": health_upper[:, is_target]}

if __name__ == '__main__':
    main(datapath = "/home/anqi/Documents/software/adarad/examples/data/")
