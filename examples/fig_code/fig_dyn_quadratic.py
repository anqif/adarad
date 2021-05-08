import numpy as np
import matplotlib
matplotlib.use("TKAgg")

from adarad.medicine.prognosis import health_prog_act
from adarad.visualization.plot_funcs import plot_health

def main():
    T = 20   # Length of optimization.
    K = 2    # Number of structures.

    # Health dynamics matrices.
    alpha = np.array(T*[[1.75, 1.1]])
    beta = np.array(T*[[0.75, 0.5]])
    gamma = np.array(T*[[1.5, 1.0]])

    # Health prognosis.
    t_off = int(T/2)
    doses = np.zeros((T,K))
    doses[:t_off,:] = 0.75
    doses[t_off:,:] = 0.6665
    h_init = np.array([0.8] + (K-1)*[0])
    is_target = np.array([True] + (K-1)*[False])
    h_prog = health_prog_act(h_init, T, alpha, beta, gamma, doses, is_target)

    # Health constraints.
    health_lower = np.zeros((T,K))
    health_upper = np.zeros((T,K))
    health_lower[:,1] = -np.inf
    health_upper[:,0] = np.inf

    # Plot health trajectory.
    plot_health(h_prog, stepsize = 5, bounds = (health_lower, health_upper), title = "Health Status vs. Time",
                one_idx = True, ylim = (-2.0, 1.0))

if __name__ == '__main__':
    main()
