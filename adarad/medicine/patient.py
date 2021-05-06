import numpy as np
from adarad.medicine.prognosis import health_prog_act


class Structure(object):
    def __init__(self, name, is_target=False, health_init=0, alpha=0, beta=0, gamma=0):
        self.name = str(name)
        self.is_target = bool(is_target)
        self.health_init = health_init
        self.model_parms = {"alpha": alpha, "beta": beta, "gamma": gamma}

    @property
    def alpha(self):
        return self.model_parms["alpha"]

    @property
    def beta(self):
        return self.model_parms["beta"]

    @property
    def gamma(self):
        return self.model_parms["gamma"]

    def __str__(self):
        return self.name

class Anatomy(object):
    def __init__(self, structures=None):
        self.__structures = []
        if structures is not None:
            self.__structures = structures

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
        return self.__structures.values().__iter__()

    @property
    def structures(self):
        return self.__structures

    @structures.setter
    def structures(self, data):
        # Check iterable.
        try:
            _ = (s for s in data)
        except TypeError:
            raise TypeError("structures must be iterable")

        if isinstance(data, dict):
            data = data.values()

        for s in data:
            if not isinstance(s, Structure):
                raise ValueError("structures must contain only elements of type Structure")
        self.__structures = data

    @property
    def n_structures(self):
        return len(self.structures)

    def is_empty(self):
        return self.n_structures == 0

    @property
    def is_target(self):
        return np.array([s.is_target for s in self.structures])

    @property
    def health_init(self):
        return np.array([s.health_init for s in self.structures])

    def health_prognosis(self, T=1):
        K = self.n_structures
        gamma_mat = np.zeros((T,K))
        for i in range(K):
            s = self.structures[i]
            s_gamma = s.model_parms["gamma"]
            if not (np.isscalar(s_gamma) or (isinstance(s_gamma, np.ndarray) and s_gamma.ndim == 1 and s_gamma.shape[0] == T)):
                raise ValueError("gamma parameter of structure {0} must be a scalar or vector of length {1}".format(s,T))
            gamma_mat[:,i] = s_gamma
        return health_prog_act(self.health_init, T, gamma=gamma_mat)

    def model_parms_to_mat(self, T=1):
        K = self.n_structures
        parm_dict = {"alpha": np.zeros((T,K)), "beta": np.zeros((T,K)), "gamma": np.zeros((T,K))}

        i = 0
        for s in self.structures:
            for key in parm_dict.keys():
                s_parm = s.model_parms[key]
                if not (np.isscalar(s_parm) or (isinstance(s_parm, np.ndarray) and s_parm.ndim == 1 and s_parm.shape[0] == T)):
                    raise ValueError("model parameter {0} of structure {1} must be a scalar or vector of length {2}".format(key,s,T))
                parm_dict[key][:,i] = s_parm
            i = i + 1
        return parm_dict
