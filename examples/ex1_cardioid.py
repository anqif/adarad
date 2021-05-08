import adarad, numpy
import matplotlib.pyplot as plt
from pathlib import Path

from adarad import Case, CasePlotter, StructMap, BeamSet
from adarad.visualization.plot_funcs import transp_cmap

from examples.utilities.simple_utils import simple_colormap

def main(datapath = ""):
    # Construct the clinical case.
    case = Case(datapath + "ex_cardioid_synthetic.yml")
    case.physics.beams = BeamSet(angles = 20, bundles = 50, offset = 5)
    # case.physics.dose_matrix = numpy.load(datapath + "cardioid_5_structs_1000_beams-dose_matrix.npy")

    # Import anatomical structure data.
    struct_dict = numpy.load(datapath + "cardioid_5_structs_1000_beams-regions.p", allow_pickle = True)
    struct_map = StructMap(struct_dict["regions"], xy_grid = (struct_dict["x_grid"], struct_dict["y_grid"]))
    struct_kw = simple_colormap(one_idx = True)   # Color map arguments for structure visual.

    # Visualize anatomical structures.
    caseviz = CasePlotter(case, figsize = (10,8), one_idx = True, struct_map = struct_map, struct_kw = struct_kw)
    caseviz.plot_structures()

    # Solve using CCP algorithm.
    status, result = case.plan(use_slack = True, slack_weight = 1e4, max_iter = 15, solver = "MOSEK", auto_init = False,
                               ccp_verbose = True)
    print("Solve status: {}".format(status))
    print("Final objective: {}".format(result.output.objective))
    print("Solve time: {}".format(result.solver_stats.solve_time))
    print("Iterations: {}".format(result.solver_stats.num_iters))

    # Plot optimal beams, doses, and health statuses over time.
    caseviz.figsize = (16,8)
    caseviz.plot_beams(result, stepsize = 1, cmap = transp_cmap(plt.cm.Reds, upper = 0.5))
    caseviz.plot_treatment(result, stepsize = 10)
    caseviz.plot_health(result, stepsize = 10, plot_untreated = True)

if __name__ == '__main__':
    base_path = Path(__file__).parent
    file_path = (base_path / "data/ex1_cardioid").resolve()
    main(datapath = file_path.__str__() + "/")
