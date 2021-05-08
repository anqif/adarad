# Base class for unit tests.
from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt

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
