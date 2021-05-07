# Base class for unit tests.
from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt

class BaseTest(TestCase):
    # AssertAlmostEqual for lists.
    def assertItemsAlmostEqual(self, a, b, places=4):
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
    def assertAlmostEqual(self, a, b, places=4):
        super(BaseTest, self).assertAlmostEqual(a.real, b.real, places=places)
        super(BaseTest, self).assertAlmostEqual(a.imag, b.imag, places=places)
