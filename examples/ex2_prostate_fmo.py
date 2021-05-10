import adarad, numpy
from pathlib import Path

from adarad import Case, CasePlotter

def main(datapath = ""):
    # Construct the clinical case.
    case = Case(datapath + "ex_prostate_FMO_stanford.yml")
    case.physics.dose_matrix = numpy.load(datapath + "prostate_7_structs_34848_beams_mean-dose_matrix.npy")

    # Solve using ADMM algorithm.
    status, result = case.plan(use_admm = True, solver = "MOSEK", rho = 80.0, eps_abs = 1e-2, eps_rel = 1e-3,
                               use_slack = True, slack_weight = 1e4, ccp_max_iter = 10, ccp_eps = 1e-3,
                               auto_init = True, admm_verbose = True)
    print("Solve status: {}".format(status))
    print("Final objective: {}".format(result.output.objective))
    print("Solve time: {}".format(result.solver_stats.solve_time))
    print("Total time: {}".format(result.solver_stats.total_time))
    print("Iterations: {}".format(result.solver_stats.num_iters))

    # Plot primal and dual residuals.
    caseviz = CasePlotter(case, one_idx = True)
    caseviz.plot_residuals(semilogy = True)

    # Plot optimal doses and health statuses over time.
    caseviz.figsize = (16,8)
    caseviz.plot_treatment(result, stepsize = 10)
    caseviz.plot_health(result, stepsize = 10, plot_untreated = True)

if __name__ == '__main__':
    base_path = Path(__file__).parent
    file_path = (base_path / "data/ex2_prostate").resolve()
    main(datapath = file_path.__str__() + "/")
