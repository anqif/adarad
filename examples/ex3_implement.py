import adarad, numpy
from adarad import Case, CasePlotter
from adarad import BeamSet

from pathlib import Path

def main(datapath = ""):
    # Construct the clinical case.
    # case = Case(datapath + "patient_01-case-with_dose_mat.yml")
    case = Case()
    case.import_file(datapath + "patient_01-case-no_dose_mat.yml")
    case.physics.dose_matrix = numpy.load(datapath + "patient_01-dose_mat.npy")
    # case.physics.beams = BeamSet(angles = 20, bundles = 50, offset = 5)

    # Solve using ADMM algorithm.
    # status, result = case.plan(use_slack = True, slack_weight = 1e4, max_iter = 15, solver = "MOSEK", ccp_verbose = True)
    status, result = case.plan(use_admm = True, solver = "MOSEK", rho = 5, admm_max_iter = 500, admm_verbose = True,
                               use_slack = True, slack_weight = 1e4, ccp_max_iter = 15)
    print("Solve status: {}".format(status))
    print("Solve time: {}".format(result.solver_stats.solve_time))
    print("Iterations: {}".format(result.solver_stats.num_iters))

    # Plot optimal doses and health statuses over time.
    caseviz = CasePlotter(case)
    caseviz.plot_treatment(result, stepsize = 10)
    caseviz.plot_health(result, stepsize = 10)
    # caseviz.plot_health(result, stepsize = 10, plot_untreated = True)
    # caseviz.plot_slacks(result)
    # caseviz.plot_residuals(result, semilogy = True)

    # Save plan for later comparison.
    case.save_plan("Old Plan")

    # Constraint allows maximum of 10 Gy/session on the PTV.
    case.prescription["PTV"].dose_upper = 10

    # Re-plan the case with new dose constraint.
    # status2, result2 = case.plan(use_slack = True, slack_weight = 1e4, max_iter = 15, solver = "MOSEK", ccp_verbose = True)
    status2, result2 = case.plan(use_admm = True, solver = "MOSEK", rho = 5, admm_max_iter = 500, use_slack = True,
                                 slack_weight = 1e4, ccp_max_iter = 15)
    print("Solve status: {}".format(status2))

    # Compare old and new dose plans.
    caseviz.plot_treatment(result2, stepsize = 10, label = "New Plan", plot_saved = True)
    caseviz.plot_health(result2, stepsize = 10, label = "New Plan", plot_saved = True)

    # prop_cycle = plt.rcParams["axes.prop_cycle"]
    # colors = prop_cycle.by_key()["color"]
    # untreated_kw = {"color": colors[1]}
    # saved_plans_kw = {"Old Plan": {"color": colors[0]}}
    # caseviz.plot_treatment(result2, stepsize = 10, label = "New Plan", plot_saved = True, saved_plans_kw = saved_plans_kw,
    #                        color = colors[2])
    # caseviz.plot_health(result2, stepsize = 10, label = "New Plan", plot_saved = True, plot_untreated = True,
    #                     untreated_kw = untreated_kw, saved_plans_kw = saved_plans_kw, color = colors[2])

if __name__ == '__main__':
    base_path = Path(__file__).parent
    file_path = (base_path / "data/ex3_implement").resolve()
    main(datapath = file_path.__str__() + "/")
