import yaml
import os.path

from adarad.medicine.physics import Physics
from adarad.medicine.patient import Anatomy, Structure
from adarad.medicine.prescription import Prescription, StructureRx
from adarad.visualization.history import RunRecord

from adarad.optimization.seq_cvx.quad_funcs import dyn_quad_treat
from adarad.optimization.admm.quad_admm_funcs import dyn_quad_treat_admm
from adarad.utilities.data_utils import check_slack_parms
from adarad.utilities.file_utils import *

class Case(object):
    def __init__(self, path=None, anatomy=None, physics=None, prescription=None):
        if isinstance(path, str):
            anatomy_args, phys_args, rx_args = self.digest(path)
            self.__anatomy = Anatomy(**anatomy_args)
            self.__physics = Physics(**phys_args)
            self.__prescription = Prescription(**rx_args)
        else:
            if not (anatomy is None or isinstance(anatomy, Anatomy)):
                raise TypeError("anatomy must be of class Anatomy")
            if not (physics is None or isinstance(physics, Physics)):
                raise TypeError("physics must be of class Physics")
            if not (prescription is None or isinstance(prescription, Prescription)):
                raise TypeError("prescription must be of class Prescription")
            self.__anatomy = anatomy
            self.__physics = physics
            self.__prescription = prescription

        self.__current_plan = None
        self.__saved_plans = dict()

    def import_file(self, path):
        if not isinstance(path, str):
            raise TypeError("path must be a string")
        anatomy_args, phys_args, rx_args = self.digest(path)
        self.__anatomy = Anatomy(**anatomy_args)
        self.__physics = Physics(**phys_args)
        self.__prescription = Prescription(**rx_args)

    def digest(self, path):
        if not (".yml" in path or ".yaml" in path):
            raise ValueError("path must point to a YAML file")
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

        struct_list = []
        struct_rx_list = []
        for i in range(K):
            s_dict = data["structures"][i]
            s_dose = s_dict["dose"]
            s_health = s_dict["health"]

            # Anatomical structures.
            struct_obj = Structure(s_dict["name"], is_target=s_dict["is_target"], health_init=s_health.get("initial", 0),
                                   alpha=s_dict["alpha"], beta=s_dict["beta"], gamma=s_dict["gamma"])
            struct_list.append(struct_obj)

            # Prescription for structure.
            if s_dict["is_target"]:
                h_w_under = s_health.get("under", 0)
                h_w_over = s_health.get("over", 1)
                h_lower = -np.inf
                h_upper = s_health.get("upper_bound", np.inf)
            else:
                h_w_under = s_health.get("under", 1)
                h_w_over = s_health.get("over", 0)
                h_lower = s_health.get("lower_bound", -np.inf)
                h_upper = np.inf
            struct_rx = StructureRx(s_dict["name"], is_target=s_dict["is_target"],
                                    dose_goal=s_dose.get("goal", 0), dose_weight=s_dose.get("weight", 1),
                                    dose_lower=s_dose.get("lower_bound", 0), dose_upper=s_dose.get("upper_bound", np.inf),
                                    health_goal=s_health.get("goal", 0), health_weights={"under": h_w_under, "over": h_w_over},
                                    health_lower=h_lower, health_upper=h_upper)
            struct_rx_list.append(struct_rx)

        # Patient anatomy.
        anatomy_args = {"structures": struct_list}

        # Dose physics.
        phys_args = dict()
        phys_args["dose_matrix"] = A_list
        phys_args["beams"], phys_args["beam_lower"], phys_args["beam_upper"] = load_beams(data, T, n)

        # Treatment prescription.
        rx_args = dict()
        rx_args["T_treat"] = data["treatment_length"]
        rx_args["structure_rxs"] = struct_rx_list
        return anatomy_args, phys_args, rx_args

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
        if not isinstance(data, Prescription):
            raise TypeError("prescription must be of type Prescription")
        self.__prescription = data
        if self.anatomy.is_empty():
            self.anatomy.structures = [Structure(s.name, s.is_target) for s in self.prescription.structure_rxs]

    @property
    def structures(self):
        return self.anatomy.structures

    def health_prognosis(self, T=1):
        return self.anatomy.health_prognosis(T)

    @property
    def current_plan(self):
        return self.__current_plan

    @property
    def saved_plans(self):
        return self.__saved_plans

    def save_plan(self, name):
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        self.__saved_plans[name] = self.__current_plan

    def get_plan(self, name):
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if name not in self.saved_plans:
            raise RuntimeError("{0} is not the name of a saved plan".format(name))
        return self.saved_plans[name]

    def delete_plan(self, name):
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if name not in self.saved_plans:
            raise RuntimeError("{0} is not the name of a saved plan".format(name))
        del self.__saved_plans[name]

    def plan(self, use_admm=False, use_slack=None, slack_weight=None, *args, **kwargs):
        if self.physics.dose_matrix is None:
            raise RuntimeError("case.physics.dose_matrix is undefined")
        elif isinstance(self.physics.dose_matrix, list):
            if len(self.physics.dose_matrix) != self.prescription.T_treat:
                raise ValueError("dose_matrix must have length T_treat = {0}".format(self.prescription.T_treat))
            dose_matrices = self.physics.dose_matrix
        else:
            dose_matrices = self.prescription.T_treat*[self.physics.dose_matrix]

        model_parms = self.anatomy.model_parms_to_mat(T=self.prescription.T_treat)
        use_slack, slack_weight = check_slack_parms(use_slack, slack_weight, default_slack=1)
        run_rec = RunRecord(use_admm=use_admm, use_slack=use_slack, slack_weight=slack_weight)

        if use_admm:
            result = dyn_quad_treat_admm(dose_matrices, model_parms["alpha"], model_parms["beta"], model_parms["gamma"],
                                         self.anatomy.health_init, self.__gather_rx(), use_slack=use_slack,
                                         slack_weight=slack_weight, *args, **kwargs)

            # Save ADMM residuals.
            run_rec.output.residuals["primal"] = result["primal"]
            run_rec.output.residuals["dual"] = result["dual"]
        else:
            result = dyn_quad_treat(dose_matrices, model_parms["alpha"], model_parms["beta"], model_parms["gamma"],
                                    self.anatomy.health_init, self.__gather_rx(), use_slack=use_slack,
                                    slack_weight=slack_weight, *args, **kwargs)

        self.__gather_optimal_vars(result, run_rec.output)
        self.__gather_solver_stats(result, run_rec.solver_stats)
        self.__current_plan = run_rec
        return run_rec.status, run_rec

    def __gather_rx(self):
        rx_for_solve = dict()
        rx_obj_mats = self.prescription.rx_to_mats()

        rx_for_solve["is_target"] = self.anatomy.is_target
        rx_for_solve["dose_goal"] = rx_obj_mats["dose_goal"]
        rx_for_solve["dose_weights"] = rx_obj_mats["dose_weights"]
        rx_for_solve["dose_constrs"] = {"lower": rx_obj_mats["dose_lower"], "upper": rx_obj_mats["dose_upper"]}

        rx_for_solve["health_goal"] = rx_obj_mats["health_goal"]
        rx_for_solve["health_weights"] = [rx_obj_mats["health_weights_under"], rx_obj_mats["health_weights_over"]]
        rx_for_solve["health_constrs"] = {"lower": rx_obj_mats["health_lower"], "upper": rx_obj_mats["health_upper"]}
        rx_for_solve["beam_constrs"] = {"lower": self.physics.beam_lower, "upper": self.physics.beam_upper}
        return rx_for_solve

    @staticmethod
    def __gather_optimal_vars(result, output):
        output.status = result["status"]
        output.objective = result["obj"]
        output.optimal_variables["beams"] = result["beams"]
        output.optimal_variables["doses"] = result["doses"]
        output.optimal_variables["health"] = result["health"]
        output.optimal_variables["slacks"] = result["health_slacks"]

    @staticmethod
    def __gather_solver_stats(result, solver_stats):
        # solver_stats.solver_name = result["solver_name"]
        solver_stats.total_time = result["total_time"]
        solver_stats.solve_time = result["solve_time"]
        solver_stats.num_iters = result["num_iters"]
