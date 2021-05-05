import numpy as np
import matplotlib
matplotlib.use("TKAgg")

from adarad.utilities.data_utils import health_prognosis
from adarad.utilities.plot_utils import plot_health

def main(savepath = ""):
    T = 20   # Length of optimization.
    K = 2    # Number of structures.

    # Health dynamics matrices.
    F = np.diag([1.05, 0.65])
    G = -np.diag([0.065, 0.15])
    r = np.zeros(K)

    # Health prognosis.
    t_off = int(T/2)
    doses = np.zeros((T,K))
    doses[:t_off,:] = 1.0
    doses[t_off:,:] = 0.15
    h_init = np.array([0.8] + (K-1)*[0])
    h_prog = health_prognosis(h_init, T, F, G, r_list = r, doses = doses)

    # Health constraints.
    health_lower = np.zeros((T, K))
    health_upper = np.zeros((T, K))
    health_lower[:,1] = -np.inf
    health_upper[:,0] = np.inf

    # Plot health trajectory.
    plot_health(h_prog, stepsize = 5, bounds = (health_lower, health_upper), title = "Health Status vs. Time", one_idx = True, ylim = (-0.5, 1.0))
    # plot_health(h_prog, stepsize = 5, bounds = (health_lower, health_upper), one_idx = True, ylim = (-0.5, 1.0), filename = savepath + "fig_health_dyn_linear.png")

if __name__ == '__main__':
	main(savepath = "/home/anqi/Dropbox/Research/Fractionation/Figures/")