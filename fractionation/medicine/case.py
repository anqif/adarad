import yaml
import os.path
import numpy as np
from warnings import warn

from fractionation.medicine.physics import Physics
from fractionation.medicine.patient import Anatomy, Structure
from fractionation.medicine.prescription import Prescription
from fractionation.history import RunRecord

from fractionation.quad_funcs import dyn_quad_treat
from fractionation.quad_admm_funcs import dyn_quad_treat_admm
from fractionation.utilities.read_utils import *

class Case(object):
    def __init__(self, path=None, anatomy=None, physics=None, prescription=None):
        # TODO: How do we handle treatment_length in A_t, alpha_t, beta_t, gamma_t, etc?
        # Import YAML file and construct objects properly, e.g., dose_matrix = T*[A] if input A is a matrix.
        # NOTE: dose_matrix can either be a list (treatment_length = length of list) or an array with treatment_length defined separately.
        if isinstance(path, str):
            structs, phys, rx = self.digest(path)
            self.__anatomy = Anatomy(structures=structs)
            self.__physics = Physics(**phys)
            self.__prescription = Prescription(rx)
        else:
            raise NotImplementedError("path must be a string")

    def digest(self, path):
        if ".yaml" not in path:
            raise ValueError("path must point to a yaml file")
        if not os.path.exists(path):
            raise RuntimeError("{0} does not exist".format(path))

        with open(path, "r") as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as err:
                print(err)

        if "treatment_length" not in data:
            raise ValueError("Case profile must include treatment_length")
        T = data["treatment_length"]

        if "structures" not in data:
            raise ValueError("Case profile must include structures")
        K = len(data["structures"])
        if K == 0:
            raise ValueError("Case profile must contain at least one structure")
        A_list = load_dose_matrix(data, T, K)
        n = A_list[0].shape[1]

        # Patient anatomy.
        struct_list = []
        for i in range(K):
            s_dict = data["structures"][i]
            struct_obj = Structure(s_dict["name"], is_target=s_dict["is_target"], health_init=s_dict["health_init"],
                                   alpha=s_dict["alpha"], beta=s_dict["beta"], gamma=s_dict["gamma"])
            struct_list.append(struct_obj)

        # Dose physics.
        phys_args = dict()
        phys_args["dose_matrix"] = A_list
        phys_args["beams"], phys_args["beam_lower"], phys_args["beam_upper"] = load_beams(data, T, n)

        # TODO: Treatment prescription
        rx_args = dict()

        return struct_list, phys_args, rx_args

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
        if self.physics.dose_matrix is None:
            raise RuntimeError("case.physics.dose_matrix is undefined")
        elif isinstance(self.physics.dose_matrix, list):
            if len(self.physics.dose_matrix) != self.prescription.T:
                raise ValueError("dose_matrix must have length T = {0}".format(self.prescription.T))
            dose_matrices = self.physics.dose_matrix
        else:
            dose_matrices = self.prescription.T*[self.physics.dose_matrix]

        model_parms = self.anatomy.model_parms_to_mat(T=self.prescription.T)
        slack_weight = kwargs.pop("slack_weight", 0)
        run_rec = RunRecord(use_admm=use_admm, slack_weight=slack_weight)

        if use_admm:
            result = dyn_quad_treat(dose_matrices, model_parms["alpha"], model_parms["beta"], model_parms["gamma"],
                                    self.anatomy.health_init, self.__gather_rx(), slack_weight=slack_weight, *args, **kwargs)
        else:
            result = dyn_quad_treat_admm(dose_matrices, model_parms["alpha"], model_parms["beta"], model_parms["gamma"],
                                         self.anatomy.health_init, self.__gather_rx(), slack_weight=slack_weight, *args, **kwargs)

        self.__gather_optimal_vars(result, run_rec.output)
        self.__gather_solver_stats(result, run_rec.solver_stats)
        return run_rec.status, run_rec

    def __gather_rx(self):
        rx = dict()
        rx["is_target"] = self.anatomy.is_target
        rx["dose_goal"] = self.prescription.dose_goal
        rx["dose_weights"] = self.prescription.dose_weights
        rx["dose_constrs"] = {"lower": self.prescription.dose_lower, "upper": self.prescription.dose_upper}

        rx["health_goal"] = self.prescription.health_goal
        rx["health_weights"] = self.prescription.health_weights
        rx["health_constrs"] = {"lower": self.prescription.health_lower, "upper": self.prescription.health_upper}
        rx["beam_constrs"] = {"lower": self.physics.beam_lower, "upper": self.physics.beam_upper}
        return rx

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
