# import adarad, numpy
# from adarad import Case, CasePlotter

import numpy
import matplotlib.pyplot as plt

from fractionation import Case, CasePlotter
from fractionation import BeamSet

def main(datapath = ""):
    # Construct the clinical case.
    # case = Case(datapath + "patient_01-case.yml")
    case = Case()
    case.import_file(datapath + "patient_01-case.yml")
    # case.physics.beams = BeamSet(angles=20, bundles=50, offset=5)
    case.physics.dose_matrix = numpy.load(datapath + "patient_01-dose_mat.npy")

    # Solve using ADMM algorithm.
    status, result = case.plan(use_slack=True, slack_weight=1e4, max_iter=15, solver="MOSEK", ccp_verbose=True)
    # status, result = case.plan(use_slack=True, slack_weight=1e4, ccp_max_iter=15, solver="MOSEK", rho=5,
    #                           admm_max_iter=500, use_admm=True, admm_verbose=True)
    print("Solve status: {}".format(status))
    print("Solve time: {}".format(result.solver_stats.solve_time))
    print("Iterations: {}".format(result.solver_stats.num_iters))

    # Plot the dose and health trajectories.
    caseviz = CasePlotter(case)
    caseviz.plot_treatment(result, stepsize=10)
    caseviz.plot_health(result, stepsize=10)
    # caseviz.plot_health(result, stepsize=10, plot_untreated=True)
    # caseviz.plot_slacks(result)
    # caseviz.plot_residuals(result)

    # Save plan for later comparison.
    case.save_plan("Old Plan")

    # Constraint allows maximum of 10 Gy on the PTV.
    case.prescription["PTV"].dose_upper = 10

    # Re-plan the case with new dose constraint.
    status2, result2 = case.plan(use_slack=True, slack_weight=1e4, max_iter=15, solver="MOSEK", ccp_verbose=True)
    # status2, result2 = case.plan(use_slack=True, slack_weight=1e4, ccp_max_iter=15, solver="ECOS", rho=5,
    #                             admm_max_iter=500, use_admm=True)
    print("Solve status: {}".format(status2))

    # Compare old and new treatment plans.
    caseviz.plot_treatment(result2, stepsize=10, label="New Plan", plot_saved=True)
    caseviz.plot_health(result2, stepsize=10, label="New Plan", plot_saved=True)

    # prop_cycle = plt.rcParams["axes.prop_cycle"]
    # colors = prop_cycle.by_key()["color"]
    # untreated_kw = {"color": colors[1]}
    # saved_plans_kw = {"Old Plan": {"color": colors[0]}}
    # caseviz.plot_treatment(result2, stepsize=10, label="New Plan", plot_saved=True, saved_plans_kw=saved_plans_kw,
    #                        color=colors[2])
    # caseviz.plot_health(result2, stepsize=10, label="New Plan", plot_saved=True, plot_untreated=True,
    #                     untreated_kw=untreated_kw, saved_plans_kw=saved_plans_kw, color=colors[2])

if __name__ == '__main__':
    main(datapath = "/home/anqi/Documents/software/fractionation/examples/data/")
