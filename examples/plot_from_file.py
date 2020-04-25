import matplotlib
matplotlib.use("TKAgg")

from fractionation.utilities.plot_utils import *
from fractionation.utilities.data_utils import health_prognosis

from example_utils import simple_structures, simple_colormap

def main(savepath = "", loadpath = "", fileprefix = ""):
	np.random.seed(1)

	T = 20          # Length of treatment.
	n_grid = 1000
	offset = 0.01   # Displacement between beams.
	n_angle = 20  # 10    # Number of angles.
	n_bundle = 50  # 10   # Number of beams per angle.
	n = n_angle*n_bundle   # Total number of beams.

	prop_cycle = plt.rcParams['axes.prop_cycle']
	colors = prop_cycle.by_key()['color']

	# Display structures on a polar grid.
	x_grid, y_grid, regions = simple_structures(n_grid, n_grid)
	struct_kw = simple_colormap(one_idx = True)
	plot_structures(x_grid, y_grid, regions, title = "Anatomical Structures", one_idx = True, **struct_kw)

	# Problem data.
	K = np.unique(regions).size   # Number of structures.
	angles = np.linspace(0, np.pi, n_angle+1)[:-1]
	offs_vec = offset*np.arange(-n_bundle//2, n_bundle//2)

	F = np.diag([1.05, 0.90, 0.75, 0.80, 0.95])
	G = -np.diag([0.01, 0.50, 0.25, 0.15, 0.0075])
	r = np.zeros(K)
	h_init = np.array([1] + (K-1)*[0])

	# Actual health status transition function.
	mu = 0
	sigma = 0.025
	h_noise = mu + sigma*np.random.randn(T,K)
	def health_map(h,t):
		h_jitter = h + h_noise[t]
		h_jitter[0] = np.maximum(h_jitter[0], 0)     # PTV: h_t >= 0.
		h_jitter[1:] = np.minimum(h_jitter[1:], 0)   # OAR: h_t <= 0.
		return h_jitter

	# Health prognosis.
	h_prog = health_prognosis(h_init, T, F, r_list = r, health_map = health_map)
	h_curves = [{"h": h_prog, "label": "Untreated", "kwargs": {"color": colors[1]}}]

	# Dose constraints.
	dose_lower = np.zeros((T,K))
	dose_upper = np.full((T,K), 25)   # Upper bound on doses.

	# Health constraints.
	health_lower = np.zeros((T,K))
	health_upper = np.zeros((T,K))
	health_lower[:,1] = -0.5
	health_lower[:,2] = -0.5
	health_lower[:,3] = -0.95
	health_lower[:,4] = -0.25
	health_upper[:15,0] = 1.5    # Upper bound on PTV for t = 1,...,15.
	health_upper[15:,0] = 0.05   # Upper bound on PTV for t = 16,...,20.

	# Load results of one-shot naive plan.
	naive_beams = np.load(loadpath + fileprefix + "beams.npy")
	naive_health = np.load(loadpath + fileprefix + "health.npy")
	naive_doses = np.load(loadpath + fileprefix + "doses.npy")

	# Load results of MPC plan.
	mpc_beams = np.load(loadpath + fileprefix + "mpc_beams.npy")
	mpc_health = np.load(loadpath + fileprefix + "mpc_health.npy")
	mpc_doses = np.load(loadpath + fileprefix + "mpc_doses.npy")

	# Set beam colors on logarithmic scale.
	b_min_naive = np.min(naive_beams[naive_beams > 0])
	b_max_naive = np.max(naive_beams)
	b_min_mpc = np.min(mpc_beams[mpc_beams > 0])
	b_max_mpc = np.max(mpc_beams)
	b_min = min(b_min_naive, b_min_mpc)
	b_max = max(b_max_naive, b_max_mpc)
	# b_col_norm = LogNorm(vmin = b_min, vmax = b_max)
	b_col_norm = Normalize(vmin = b_min, vmax = b_max)

	# Plot beam, health, and treatment curves.
	plot_beams(naive_beams, angles = angles, offsets = offs_vec, stepsize = 1, cmap = transp_cmap(plt.cm.Reds, upper = 0.5), \
				title = "Beam Intensities vs. Time", one_idx = True, structures = (x_grid, y_grid, regions), struct_kw = struct_kw, norm = b_col_norm)
	# plot_health(naive_health, curves = h_curves, stepsize = 10, bounds = (health_lower, health_upper), title = "Health Status vs. Time", one_idx = True)
	plot_treatment(naive_doses, stepsize = 10, bounds = (dose_lower, dose_upper), title = "Treatment Dose vs. Time", one_idx = True)

	# plot_beams(naive_beams, angles = angles, offsets = offs_vec, stepsize = 1, cmap = transp_cmap(plt.cm.Reds, upper = 0.5), \
	#			one_idx = True, structures = (x_grid, y_grid, regions), struct_kw = struct_kw, norm = b_col_norm, \
	#			filename = savepath + fileprefix + "dyn_beams.png")

	# Compare one-shot dynamic and MPC treatment results.
	d_curves = [{"d": naive_doses, "label": "Naive", "kwargs": {"color": colors[0]}}]
	h_naive = [{"h": naive_health, "label": "Treated (Naive)", "kwargs": {"color": colors[0]}}]
	h_curves = h_naive + h_curves

	plot_beams(mpc_beams, angles = angles, offsets = offs_vec, stepsize = 1, cmap = transp_cmap(plt.cm.Reds, upper = 0.5), \
				title = "Beam Intensities vs. Time", one_idx = True, structures = (x_grid, y_grid, regions), struct_kw = struct_kw, norm = b_col_norm)
	# plot_health(mpc_health, curves = h_curves, stepsize = 10, bounds = (health_lower, health_upper), \
	#			title = "Health Status vs. Time", label = "Treated (MPC)", color = colors[2], one_idx = True)
	plot_treatment(mpc_doses, curves = d_curves, stepsize = 10, bounds = (dose_lower, dose_upper), \
				title = "Treatment Dose vs. Time", label = "MPC", color = colors[2], one_idx = True)

	# plot_beams(mpc_beams, angles = angles, offsets = offs_vec, stepsize = 1, cmap = transp_cmap(plt.cm.Reds, upper = 0.5), \
	#		  	one_idx = True, structures = (x_grid, y_grid, regions), struct_kw = struct_kw, norm = b_col_norm, \
	#		  	filename = savepath + fileprefix + "mpc_admm_beams.png")

if __name__ == '__main__':
	main(savepath = "/home/anqi/Documents/software/fractionation/fractionation/output/", \
		 loadpath = "/home/anqi/Documents/software/fractionation/fractionation/output/", fileprefix = "ex_simple_noisy_")
