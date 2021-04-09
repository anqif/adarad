import numpy as np

class Prescription(object):
    def __init__(self, treatment_length, n_structures, rx_dict=None):
        if treatment_length <= 0:
            raise ValueError("treatment_length must be a positive integer")
        if n_structures <= 0:
            raise ValueError("n_structures must be a positive integer")
        self.T = int(treatment_length)
        K = int(n_structures)

        # Default values.
        # TODO: Set values based on ingested case.
        self.dose_goal = np.zeros((self.T, K))
        self.dose_weights = np.ones((K,))
        self.dose_lower = 0
        self.dose_upper = np.inf

        self.health_goal = np.zeros((self.T, K))
        self.health_weights = [np.ones((K,)), np.ones((K,))]
        self.health_lower = -np.inf
        self.health_upper = np.inf