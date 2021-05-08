import matplotlib
matplotlib.use("TKAgg")

from adarad.visualization.plot_funcs import *
from adarad.utilities.beam_utils import line_integral_mat
from examples.utilities.simple_utils import simple_structures, simple_colormap

def main():
	n_grid = 1000
	offset = 5       # Displacement between beams (pixels).
	n_angle = 20     # Number of angles.
	n_bundle = 50    # Number of beams per angle.
	n = n_angle*n_bundle   # Total number of beams.

	# Structure data.
	x_grid, y_grid, regions = simple_structures(n_grid, n_grid)
	struct_kw = simple_colormap(one_idx = True)

	# Beam data.
	A, angles, offs_vec = line_integral_mat(regions, angles = n_angle, n_bundle = n_bundle, offset = offset)
	beam_kw = {"cmap": transp_cmap(plt.cm.Reds, upper = 0.5)}
	beam_sls = [slice(10,35), slice(350, 365), slice(600,615)]

	# Plot structures with different beams.
	beams = np.zeros(n)
	S = len(beam_sls)
	for i in range(S):
		beams[beam_sls[i]] = 1.0
		plot_struct_beams(x_grid, y_grid, regions, beams, angles, offs_vec, n_grid, beam_kw = beam_kw, one_idx = True,
						  **struct_kw)

if __name__ == '__main__':
	main()