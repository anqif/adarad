# import adarad, numpy
# from adarad import Case, CasePlotter

import numpy
from fractionation.medicine.physics import BeamSet
from fractionation.medicine.case import Case
from fractionation.visualization.plotter import CasePlotter

def main(datapath = ""):
    # Construct the clinical case.
    case = Case(datapath + "case_patient_01.yaml")
    case.physics.beams = BeamSet(angles = 20, bundles = 50, offset = 5)
    case.physics.dose_matrix = numpy.load(datapath + "dmat_patient_01.npy")

    # Solve using ADMM algorithm.
    status, result = case.plan(slack_weight = 50, max_iter = 100, solver = "ECOS", use_admm = True)
    print("Solve status: {}".format(status))
    print("Solve time: {}".format(result.solver_stats.solve_time))
    print("Iterations: {}".format(result.solver_stats.num_iters))

    # Plot the dose and health trajectories.
    caseviz = CasePlotter(case)
    caseviz.plot_treatment(result, stepsize = 10)
    caseviz.plot_health(result, stepsize = 10)

    # Constraint allows maximum of 50 Gy on the PTV in first 10 sessions.
    case.prescription["PTV"].dose_upper[:10] = 50

    # Re-plan the case with new dose constraint.
    status2, result2 = case.plan(slack_weight = 50, max_iter = 100, solver = "ECOS", use_admm = True)
    print("Solve status: {}".format(status2))

    # Compare old and new treatment plans.
    caseviz.plot_treatment(result, stepsize = 10, label = "Old", show = False)
    caseviz.plot_treatment(result2, stepsize = 10, label = "New")
    caseviz.plot_health(result, stepsize = 10, label = "Old", show = False)
    caseviz.plot_health(result2, stepsize = 10, label = "New")

if __name__ == '__main__':
    main(datapath = "/home/anqi/Documents/software/fractionation/examples/output/")
