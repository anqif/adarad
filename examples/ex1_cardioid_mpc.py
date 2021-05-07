import adarad, numpy
import matplotlib.pyplot as plt

from adarad import Case, CasePlotter, StructMap, BeamSet
from adarad.visualization.plot_funcs import transp_cmap

from examples.utilities.simple_utils import simple_colormap

def main(datapath = ""):
    numpy.random.seed(1)

    # Construct the clinical case.
    case = Case(datapath + "ex_cardioid_synthetic.yml")
    case.physics.beams = BeamSet(angles = 20, bundles = 50, offset = 5)
    # case.physics.dose_matrix = numpy.load(datapath + "cardioid_5_structs_1000_beams-dose_matrix.npy")

    # Define actual (stochastic) health dynamics mapping.
    h_noise = 0.1*numpy.random.randn(case.prescription.T_treat, case.anatomy.n_structures)
    case.anatomy.structures[0].health_map = lambda h,d,t: numpy.maximum(h + h_noise[t,0], 0)   # PTV: h_t >= 0.
    for k in range(1,case.anatomy.n_structures):
        case.anatomy.structures[k].health_map = lambda h,d,t,k=k: numpy.minimum(h + h_noise[t,k], 0)   # OAR: h_t <= 0.

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
    print("CCP results")
    print("Solve status: {}".format(status))
    print("Final objective: {}".format(result.output.objective))
    print("Solve time: {}".format(result.solver_stats.solve_time))
    print("Iterations: {}".format(result.solver_stats.num_iters))

    # Save plan for later comparison.
    case.save_plan("Naive Plan")

    # Solve using CCP algorithm with MPC.
    status_mpc, result_mpc = case.plan(use_mpc = True, use_slack = True, slack_weight = 1e4, use_mpc_slack = True,
                                       mpc_slack_weights = 1e4, max_iter = 100, solver = "MOSEK", auto_init = False,
                                       mpc_verbose = True)
    print("\nCCP with MPC results")
    print("Solve status: {}".format(status_mpc))
    print("Final objective: {}".format(result_mpc.output.objective))
    print("Solve time: {}".format(result_mpc.solver_stats.solve_time))
    print("Iterations: {}".format(result_mpc.solver_stats.num_iters))

    # Plot optimal beams, doses, and health statuses over time.
    caseviz.figsize = (16,8)
    caseviz.plot_beams(result_mpc, stepsize = 1, cmap = transp_cmap(plt.cm.Reds, upper = 0.5))
    # caseviz.plot_treatment(result_mpc, stepsize = 10, plot_saved = True)
    # caseviz.plot_health(result_mpc, stepsize = 10, plot_untreated = True, plot_saved = True)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    caseviz.plot_treatment(result_mpc, stepsize = 10, plot_saved = True, label = "MPC Plan", color = colors[2],
                                       saved_plans_kw = {"color": colors[0]})
    caseviz.plot_health(result_mpc, stepsize = 10, plot_untreated = True, plot_saved = True, label = "MPC Plan", color = colors[2],
                                    untreated_kw = {"color": colors[1]}, saved_plans_kw = {"color": colors[0]})

if __name__ == '__main__':
    main(datapath = "/home/anqi/Documents/software/adarad/examples/data/ex1_cardioid/")
