from fractionation.medicine.physics import Physics
from fractionation.medicine.patient import Anatomy
from fractionation.medicine.prescription import Prescription
from fractionation.history import RunRecord
from fractionation.utilities.file_utils import yaml_to_dict

from fractionation.quad_funcs import dyn_quad_treat
from fractionation.quad_admm_funcs import dyn_quad_treat_admm

class Case(object):
    def __init__(self, path=None, anatomy=None, physics=None, prescription=None):
        # TODO: Import YAML file and construct objects properly, e.g., dose_matrix = T*[A] if input A is a matrix.
        # NOTE: dose_matrix can either be a list (treatment_length = length of list) or an array with treatment_length defined separately.
        if path is not None:
            bio, rx, visuals = yaml_to_dict(path)
            self.__anatomy = Anatomy(bio)
            self.__physics = Physics()
            self.__prescription = Prescription(rx)
        else:
            self.__anatomy = anatomy
            self.__physics = physics
            self.__prescription = prescription

    @property
    def anatomy(self):
        return self.__anatomy

    @anatomy.setter
    def anatomy(self, data):
        self.__anatomy = Anatomy(data)

    @property
    def physics(self):
        return self.__physics

    @physics.setter
    def physics(self, data):
        if isinstance(data, dict):
            self.__physics = Physics(**data)
        else:
            self.__physics = Physics(data)

    @property
    def prescription(self):
        return self.__prescription

    @prescription.setter
    def prescription(self, data):
        self.__prescription = Prescription(data)
        if self.anatomy.is_empty():
            self.anatomy.structures = self.prescription.structures

    @property
    def structures(self):
        return self.anatomy.structures

    def plan(self, use_admm=False, *args, **kwargs):
        slack_weight = kwargs.pop("slack_weight", 0)
        run_rec = RunRecord(use_admm=use_admm, slack_weight=slack_weight)

        if use_admm:
            result = dyn_quad_treat(self.physics.dose_matrix, self.anatomy.alpha, self.anatomy.beta, self.anatomy.gamma,
                                    self.anatomy.h_init, self.prescription.asdict(), slack_weight=slack_weight, *args, **kwargs)
        else:
            result = dyn_quad_treat_admm(self.physics.dose_matrices, self.anatomy.alpha, self.anatomy.beta, self.anatomy.gamma,
                                         self.anatomy.h_init, self.prescription.asdict(), slack_weight=slack_weight, *args, **kwargs)

        self.__gather_optimal_vars(result, run_rec.output)
        self.__gather_solver_stats(result, run_rec.solver_stats)
        return run_rec.status, run_rec

    @staticmethod
    def __gather_optimal_vars(result, output):
        output.status = result["status"]
        output.objective = result["obj"]
        output.optimal_variables["beams"] = result["beams"]
        output.optimal_variables["doses"] = result["doses"]
        output.optimal_variables["health"] = result["health"]

    @staticmethod
    def __gather_solver_stats(result, solver_stats):
        # solver_stats.solver_name = result["solver_name"]
        solver_stats.solve_time = result["solve_time"]
        solver_stats.num_iters = result["num_iters"]
