class Structure(object):
    def __init__(self, name, is_target = False, h_init = 0, alpha = 0, beta = 0, gamma = 0):
        self.name = str(name)
        self.is_target = bool(is_target)

class Anatomy(object):
    def __init__(self, structures = None):
        self.__structures = {}
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