import numpy as np
import matplotlib
matplotlib.use("TKAgg")

from fractionation.utilities.data_utils import health_prog_quad
from fractionation.utilities.plot_utils import plot_health

def main(savepath = ""):
    T = 20   # Length of treatment.
    K = 2    # Number of structures.

    # Health dynamics matrices.
    alpha = np.array([1.0, 1.5])
    beta = np.array([0.5, 1.0])
    gamma = np.array([1.0, 2.0])

    # Health prognosis.
    t_off = int(T/2)
    doses = np.zeros((T,K))
    doses[:t_off,:] = 1.0
    doses[t_off:,:] = 0.15
    h_init = np.array([0.8] + (K-1)*[0])
    h_prog = health_prog_quad(h_init, T, alpha, beta, gamma, doses)

    # Health constraints.
    health_lower = np.zeros((T,K))
    health_upper = np.zeros((T,K))
    health_lower[:,1] = -np.inf
    health_upper[:,0] = np.inf

    # Plot health trajectory.
    plot_health(h_prog, stepsize = 5, bounds = (health_lower, health_upper), title = "Health Status vs. Time", one_idx = True)
    # plot_health(h_prog, stepsize = 5, bounds = (health_lower, health_upper), one_idx = True, ylim = (-0.5, 1.0), filename = savepath + "fig_health_dyn_quad.png")

if __name__ == '__main__':
	main(savepath = "/home/anqi/Dropbox/Research/Fractionation/Figures/")