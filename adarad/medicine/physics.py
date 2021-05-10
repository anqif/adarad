import numpy as np

class BeamSet(object):
    def __init__(self, angles=10, bundles=1, offset=10):
        if isinstance(angles, int):
            self.angles = np.linspace(0, np.pi, angles+1)[:-1]
        elif isinstance(angles, np.ndarray):
            self.angles = angles
        else:
            raise ValueError("angles must be a positive integer or numpy.ndarray")
        if bundles <= 0:
            raise ValueError("bundles must be a positive integer")
        if offset < 0:
            raise ValueError("offset must be a non-negative number")

        b_half = bundles // 2
        d_vec = np.arange(-b_half, b_half + 1)
        if bundles % 2 == 0:
            d_vec = d_vec[:-1]
        self.offsets = offset * d_vec

    @property
    def n_beams(self):
        return len(self.angles)*len(self.offsets)

class Physics(object):
    def __init__(self, dose_matrix=None, beams=None, beam_lower=0, beam_upper=np.inf):
        if dose_matrix:
            K, n = self.check_dose_matrix(dose_matrix)
        self.__dose_matrix = dose_matrix

        if isinstance(beams, BeamSet):
            if beams.n_beams != n:
                raise ValueError("number of beams must equal number of columns in dose_matrix")
        self.__beams = beams
        self.beam_lower = beam_lower
        self.beam_upper = beam_upper

    @staticmethod
    def check_dose_matrix(dose_matrix):
        if isinstance(dose_matrix, list):
            if len(dose_matrix) == 0:
                raise ValueError("dose_matrix cannot be an empty list")
            if dose_matrix[0].ndim != 2:
                raise ValueError("dose_matrix must be a list of 2-D arrays")
            K, n = dose_matrix[0].shape
            for dm in dose_matrix:
                if not np.all(dm.shape == (K, n)):
                    raise ValueError("dose_matrix must be a list of matrices with the same dimensions")
        elif isinstance(dose_matrix, np.ndarray):
            if dose_matrix.ndim != 2:
                raise ValueError("dose_matrix must be a 2-D array")
            K, n = dose_matrix.shape
        else:
            raise ValueError("dose_matrix must be a list or numpy.ndarray")
        return K, n

    @property
    def dose_matrix(self):
        return self.__dose_matrix

    @dose_matrix.setter
    def dose_matrix(self, data):
        K, n = self.check_dose_matrix(data)
        if self.beams is not None and self.beams.n_beams != n:
            raise ValueError("data must be a matrix with {0} columns".format(n))
        if not isinstance(data, list):
            self.__dose_matrix = [data]
        self.__dose_matrix = data

    @property
    def beams(self):
        return self.__beams

    @beams.setter
    def beams(self, data):
        if isinstance(data, BeamSet):
            beams_new = data
        elif isinstance(data, dict):
            beams_new = BeamSet(**data)
        else:
            beams_new = BeamSet(data)

        if self.__dose_matrix is not None:
            if isinstance(self.__dose_matrix, list):
                K, n = self.__dose_matrix[0].shape
            else:
                K, n = self.__dose_matrix.shape
            if beams_new.n_beams != n:
                raise ValueError("data must define a set of {0} beams".format(n))
        self.__beams = beams_new
