import numpy as np
from adarad.visualization.plot_funcs import *
from adarad.tests.base_test import BaseTest

class TestPlotFuncs(BaseTest):
    """Unit tests for AdaRad plotting functions."""

    def setUp(self):
        np.random.seed(1)

        num_iters = [50, 100, 200, 250, 40, 60, 80, 100]
        self.vec_single_1 = np.random.randn(50)
        self.vec_single_2 = np.random.randn(50)
        self.vec_list_1 = [np.random.randn(n) for n in num_iters]
        self.vec_list_2 = [np.random.randn(n) for n in num_iters]

    def test_plot_residuals(self):
        plot_residuals(self.vec_single_1, self.vec_single_2)
        plot_residuals(self.vec_list_1, self.vec_list_2)

    def test_plot_slacks(self):
        plot_slacks(self.vec_single_1)
        plot_slacks(self.vec_list_1)
