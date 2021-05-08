# Base class for unit tests.
from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt

from adarad.medicine.prognosis import health_prog_act
from adarad.utilities.beam_utils import line_integral_mat
from examples.utilities.simple_utils import simple_structures

class BaseTest(TestCase):
    # AssertAlmostEqual for lists.
    def assertItemsAlmostEqual(self, a, b, places: int = 5) -> None:
        if np.isscalar(a):
            a = [a]
        else:
            a = self.mat_to_list(a)
        if np.isscalar(b):
            b = [b]
        else:
            b = self.mat_to_list(b)
        for i in range(len(a)):
            self.assertAlmostEqual(a[i], b[i], places)

    # Overridden method to assume lower accuracy.
    def assertAlmostEqual(self, a, b, places: int = 5, delta=None) -> None:
        super(BaseTest, self).assertAlmostEqual(a, b, places=places, delta=delta)

    def mat_to_list(self, mat):
        """Convert a numpy matrix to a list.
        """
        if isinstance(mat, (np.matrix, np.ndarray)):
            return np.asarray(mat).flatten('F').tolist()
        else:
            return mat

    def setUpSimpleProblem(self):
        # Problem data.
        T = 20  # Length of treatment.
        n_grid = 1000
        offset = 5  # Displacement between beams (pixels).
        n_angle = 20  # Number of angles.
        n_bundle = 50  # Number of beams per angle.
        n = n_angle * n_bundle  # Total number of beams.
        self.t_s = 0  # Static session.

        # Anatomical structures.
        x_grid, y_grid, regions = simple_structures(n_grid, n_grid)
        K = np.unique(regions).size  # Number of structures.

        A, angles, offs_vec = line_integral_mat(regions, angles=n_angle, n_bundle=n_bundle, offset=offset)
        A = A / n_grid
        self.A_list = T * [A]

        self.alpha = np.array(T * [[0.01, 0.50, 0.25, 0.15, 0.005]])
        self.beta = np.array(T * [[0.001, 0.05, 0.025, 0.015, 0.0005]])
        self.gamma = np.array(T * [[0.05, 0, 0, 0, 0]])
        self.h_init = np.array([1] + (K - 1) * [0])

        is_target = np.array([True] + (K - 1) * [False])
        num_ptv = np.sum(is_target)
        num_oar = K - num_ptv

        # Health prognosis.
        prop_cycle = plt.rcParams['axes.prop_cycle']
        self.colors = prop_cycle.by_key()['color']
        h_prog = health_prog_act(self.h_init, T, gamma=self.gamma)
        self.h_curves = [{"h": h_prog, "label": "Untreated", "kwargs": {"color": self.colors[1]}}]

        # Beam constraints.
        beam_upper = np.full((T, n), 1.0)

        # Dose constraints.
        dose_lower = np.zeros((T, K))
        dose_upper = np.full((T, K), 20)

        # Health constraints.
        health_lower = np.full((T, K), -np.inf)
        health_upper = np.full((T, K), np.inf)
        health_lower[:, 1] = -1.0  # Lower bound on OARs.
        health_lower[:, 2] = -2.0
        health_lower[:, 3] = -2.0
        health_lower[:, 4] = -3.0
        health_upper[:15, 0] = 2.0  # Upper bound on PTV for t = 1,...,15.
        health_upper[15:, 0] = 0.05  # Upper bound on PTV for t = 16,...,20.

        self.patient_rx = {"is_target": is_target,
                           "dose_goal": np.zeros((T, K)),
                           "dose_weights": np.array((K - 1) * [1] + [0.25]),
                           "health_goal": np.zeros((T, K)),
                           "health_weights": [np.array([0] + (K - 1) * [0.25]), np.array([1] + (K - 1) * [0])],
                           "beam_constrs": {"upper": beam_upper},
                           "dose_constrs": {"lower": dose_lower, "upper": dose_upper},
                           "health_constrs": {"lower": health_lower, "upper": health_upper}}
