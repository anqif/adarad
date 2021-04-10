import numpy as np

class ProblemParms(object):
    def __init__(self, goal=0, lower_bound=-np.inf, upper_bound=np.inf):
        self.__goal = goal
        if lower_bound > upper_bound:
            raise ValueError("lower_bound must be less than or equal to upper_bound")
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @property
    def goal(self):
        return self.__goal

    @goal.setter
    def goal(self, data):
        if not (np.isscalar(data) or isinstance(data, np.ndarray) and data.ndim == 1):
            raise ValueError("goal must be a scalar or a vector")
        self.__goal = data

class SquareParms(ProblemParms):
    def __init__(self, goal=0, weight=1, lower_bound=-np.inf, upper_bound=np.inf):
        if not np.isscalar(weight) or weight < 0:
            raise ValueError("weight must be a non-negative scalar")
        self.__weight = weight
        super().__init__(goal, lower_bound, upper_bound)

    @property
    def weight(self):
        return self.__weight

    @weight.setter
    def weight(self, data):
        if not np.isscalar(data) or data < 0:
            raise ValueError("weight must be a non-negative scalar")
        self.__weight = data

class HingeParms(ProblemParms):
    def __init__(self, goal=0, weight_under=1, weight_over=1, lower_bound=-np.inf, upper_bound=np.inf):
        if not np.isscalar(weight_under) or weight_under < 0:
            raise ValueError("weight_under must be a non-negative scalar")
        if not np.isscalar(weight_over) or weight_over < 0:
            raise ValueError("weight_over must be a non-negative scalar")
        self.__weight_under = weight_under
        self.__weight_over = weight_over
        super().__init__(goal, lower_bound, upper_bound)

    @property
    def weight_under(self):
        return self.__weight_under

    @weight_under.setter
    def weight_under(self, data):
        if not np.isscalar(data) or data < 0:
            raise ValueError("weight_under must be a non-negative scalar")
        self.__weight_under = data

    @property
    def weight_over(self):
        return self.__weight_over

    @weight_over.setter
    def weight_over(self, data):
        if not np.isscalar(data) or data < 0:
            raise ValueError("weight_over must be a non-negative scalar")
        self.__weight_over = data

    @property
    def weights(self):
        return self.__weight_under, self.__weight_over

    @weights.setter
    def weights(self, data):
        if len(data) != 2:
            raise ValueError("weights must be a tuple, list, or array of length 2")
        self.__weight_under = data[0]
        self.__weight_over = data[1]
