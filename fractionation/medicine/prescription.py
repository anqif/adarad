import numpy as np

def check_scal_vec(data, name="data"):
    if not (np.isscalar(data) or isinstance(data, np.ndarray) and data.ndim == 1):
        raise TypeError("{0} must be a scalar or a vector".format(name))
    return True

def check_nonneg(data, name="data"):
    if (np.isscalar(data) and data < 0) or (isinstance(data, np.ndarray) and np.any(data < 0)):
        raise ValueError("{0} must be non-negative".format(name))
    return True

class StructureRx(object):
    def __init__(self, name, is_target, dose_goal=0, dose_weight=1, dose_lower=0, dose_upper=np.inf,
                            health_goal=0, health_weights=None, health_lower=-np.inf, health_upper=np.inf):
        self.name = name
        self.is_target = is_target

        if check_nonneg(dose_goal, "dose_goal") and check_scal_vec(dose_goal, "dose_goal"):
            self.__dose_goal = dose_goal
        if check_nonneg(dose_weight, "dose_weight"):
            self.__dose_weight = dose_weight
        if check_scal_vec(dose_lower, "dose_lower"):
            self.__dose_lower = dose_lower
        if check_scal_vec(dose_upper, "dose_upper"):
            self.__dose_upper = dose_upper

        if check_scal_vec(health_goal, "health_goal"):
            self.__health_goal = health_goal
        if health_weights is None:
            if is_target:
                self.__health_weights = {"under": 0, "over": 1}
            else:
                self.__health_weights = {"under": 1, "over": 0}
        else:
            if not isinstance(health_weights, dict):
                raise TypeError("health_weights must be a dict")
            elif not ("under" in health_weights and "over" in health_weights):
                raise ValueError("health_weights must contain keys 'under' and 'over'")
            elif not (np.isscalar(health_weights["under"]) and np.isscalar(health_weights["over"])):
                raise ValueError("health_weights must contain scalar values")
            self.__health_weights = health_weights
        if check_scal_vec(health_lower, "health_lower"):
            self.__health_lower = health_lower
        if check_scal_vec(health_upper, "health_upper"):
            self.__health_upper = health_upper

    @property
    def dose_goal(self):
        return self.__dose_goal

    @dose_goal.setter
    def dose_goal(self, data):
        if not (np.isscalar(data) or isinstance(data, np.ndarray) and data.ndim == 1):
            raise TypeError("dose_goal must be a scalar or a vector")
        if (np.isscalar(data) and data < 0) or (isinstance(data, np.ndarray) and np.any(data < 0)):
            raise ValueError("dose_goal must be non-negative")
        self.__dose_goal = data

    @property
    def dose_weight(self):
        return self.__dose_weight

    @dose_weight.setter
    def dose_weight(self, data):
        if not np.isscalar(data) or data < 0:
            raise ValueError("dose_weight must be a non-negative scalar")
        self.__dose_weight = data

    @property
    def dose_lower(self):
        return self.__dose_lower

    @dose_lower.setter
    def dose_lower(self, data):
        if not (np.isscalar(data) or isinstance(data, np.ndarray) and data.ndim == 1):
            raise TypeError("dose_lower must be a scalar or a vector")
        self.__dose_lower = data

    @property
    def dose_upper(self):
        return self.__dose_upper

    @dose_upper.setter
    def dose_upper(self, data):
        if not (np.isscalar(data) or isinstance(data, np.ndarray) and data.ndim == 1):
            raise TypeError("dose_upper must be a scalar or a vector")
        self.__dose_upper = data

    @property
    def health_goal(self):
        return self.__health_goal

    @health_goal.setter
    def health_goal(self, data):
        if not (np.isscalar(data) or isinstance(data, np.ndarray) and data.ndim == 1):
            raise TypeError("health_goal must be a scalar or a vector")
        self.__health_goal = data

    # @property
    # def health_weights(self):
    #    return self.__health_weights

    @property
    def health_weight_under(self):
        return self.__health_weights["under"]

    @health_weight_under.setter
    def health_weight_under(self, data):
        if not np.isscalar(data) or data < 0:
            raise ValueError("health_weight_under must be a non-negative scalar")
        self.__health_weights["under"] = data

    @property
    def health_weight_over(self):
        return self.__health_weights["over"]

    @health_weight_over.setter
    def health_weight_over(self, data):
        if not np.isscalar(data) or data < 0:
            raise ValueError("health_weight_over must be a non-negative scalar")
        self.__health_weights["over"] = data

    @property
    def health_lower(self):
        return self.__health_lower

    @health_lower.setter
    def health_lower(self, data):
        if not (np.isscalar(data) or isinstance(data, np.ndarray) and data.ndim == 1):
            raise TypeError("dose_goal must be a scalar or a vector")
        self.__health_lower = data

    @property
    def health_upper(self):
        return self.__health_upper

    @health_upper.setter
    def health_upper(self, data):
        if not (np.isscalar(data) or isinstance(data, np.ndarray) and data.ndim == 1):
            raise TypeError("dose_goal must be a scalar or a vector")
        self.__health_upper = data

class Prescription(object):
    def __init__(self, T, structure_rxs=None):
        if T <= 0:
            raise ValueError("treatment length T must be a positive integer")
        self.T = int(T)
        self.__structure_rxs = []
        if structure_rxs is not None:
            self.__structure_rxs = structure_rxs

    def __contains__(self, comparator):
        for s in self:
            if comparator == s.name:
                return True
        return False

    def __getitem__(self, key):
        for s in self:
            if key == s.name:
                return s
        raise KeyError("key {0} does not correspond to a structure name".format(key))

    def __iter__(self):
        return self.__structure_rxs.values().__iter__()

    @property
    def structure_rxs(self):
        return self.__structure_rxs

    @property
    def n_structure_rxs(self):
        return len(self.structure_rxs)

    def is_empty(self):
        return self.n_structure_rxs == 0

    @property
    def is_target(self):
        return np.array([s.is_target for s in self.structure_rxs])

    @property
    def dose_weights(self):
        return np.array([s.dose_weight for s in self.structure_rxs])

    @property
    def health_weights(self):
        K = self.n_structure_rxs
        w_under = np.zeros(K)
        w_over = np.zeros(K)
        for i in range(K):
            s = self.structure_rxs[i]
            w_under[i] = s.health_weight_under
            w_over[i] = s.health_weight_over
        return [w_under, w_over]

    def rx_to_mats(self):
        T = self.T
        K = self.n_structure_rxs

        mat_dict = dict()
        mat_dict["dose_goal"] = np.zeros((T,K))
        mat_dict["dose_weights"] = np.ones(K)
        mat_dict["dose_lower"] = np.zeros((T,K))
        mat_dict["dose_upper"] = np.full((T,K), np.inf)

        mat_dict["health_goal"] = np.zeros((T,K))
        mat_dict["health_weights_under"] = np.ones(K)
        mat_dict["health_weights_over"] = np.ones(K)
        mat_dict["health_lower"] = np.full((T,K), -np.inf)
        mat_dict["health_upper"] = np.full((T,K), np.inf)

        for i in range(K):
            s = self.structure_rxs[i]
            mat_dict["dose_goal"][:,i] = s.dose_goal
            mat_dict["dose_weights"][i] = s.dose_weight
            mat_dict["dose_lower"][:,i] = s.dose_lower
            mat_dict["dose_upper"][:,i] = s.dowe_upper

            mat_dict["health_goal"][:,i] = s.health_goal
            mat_dict["health_weights_under"][i] = s.health_weight_under
            mat_dict["health_weights_over"][i] = s.health_weight_over
            mat_dict["health_lower"][:,i] = s.health_lower
            mat_dict["health_upper"][:,i] = s.health_upper
        return mat_dict
