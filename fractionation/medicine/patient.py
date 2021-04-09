import numpy as np

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

    def model_parms_to_mat(self, T=1):
        K = self.n_structures
        parm_dict = {"alpha": np.zeros((T,K)), "beta": np.zeros((T,K)), "gamma": np.zeros((T,K))}

        i = 0
        for s in self.structures:
            for key in parm_dict.keys():
                s_parm = s.model_parms[key]
                if not np.isscalar(s_parm) or len(s_parm) != T:
                    raise ValueError("model parameter {0} of structure {1} must be a scalar or vector of length {2}".format(key,s,T))
                parm_dict[key][:,i] = s_parm
            i = i + 1
        return parm_dict
