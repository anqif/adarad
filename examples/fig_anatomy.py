import numpy as np
import matplotlib
matplotlib.use("TKAgg")

from fractionation.utilities.plot_utils import *
from fractionation.utilities.data_utils import line_integral_mat
from example_utils import simple_structures, simple_colormap

def main(figpath = "", datapath = ""):
	n_grid = 1000
	offset = 5       # Displacement between beams (pixels).
	n_angle = 20     # Number of angles.
	n_bundle = 50    # Number of beams per angle.
	n = n_angle*n_bundle   # Total number of beams.
	figprefix = "ex1_structures_beam_"

	# Structure data.
	x_grid, y_grid, regions = simple_structures(n_grid, n_grid)
	struct_kw = simple_colormap(one_idx = True)
	K = np.unique(regions).size   # Number of structures.

	# Beam data.
	A, angles, offs_vec = line_integral_mat(regions, angles = n_angle, n_bundle = n_bundle, offset = offset)
	beam_kw = {"cmap": transp_cmap(plt.cm.Reds, upper = 0.5)}
	beam_sls = [slice(10,35), slice(350, 365), slice(600,615)]
	# beam_sls = [slice(10,35), slice(510, 535), slice(350, 365), slice(600,615)]

	# Plot structures with different beams.
	beams = np.zeros(n)
	S = len(beam_sls)
	for i in range(S):
		beams[beam_sls[i]] = 1.0
		filename = figpath + figprefix + str(i+1)
		plot_struct_beams(x_grid, y_grid, regions, beams, angles, offs_vec, n_grid, beam_kw = beam_kw, one_idx = True, 
						  filename = filename, **struct_kw)

if __name__ == '__main__':
	main(figpath = "C:/Users/Anqi/Documents/Software/fractionation/examples/output/figures/",
		 datapath = "C:/Users/Anqi/Documents/Software/fractionation/examples/data/")