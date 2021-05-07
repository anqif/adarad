import numpy as np
from adarad.visualization.plot_funcs import *
from adarad.tests.base_test import BaseTest
from examples.utilities.simple_utils import *

class TestPlotFuncs(BaseTest):
    """Unit tests for AdaRad plotting functions."""

    def setUp(self):
        np.random.seed(1)
        num_iters = [50, 100, 200, 250, 40, 60, 80, 100]
        self.vec_single_1 = np.random.randn(50)
        self.vec_single_2 = np.random.randn(50)
        self.vec_list_1 = [np.random.randn(n) for n in num_iters]
        self.vec_list_2 = [np.random.randn(n) for n in num_iters]

    def test_plot_structures(self):
        n = 1000
        x_grid, y_grid, regions = simple_structures(n, n)
        struct_kw = simple_colormap(one_idx=True)
        plot_structures(x_grid, y_grid, regions, title="Anatomical Structures", one_idx=True, **struct_kw)

    def test_plot_residuals(self):
        plot_residuals(self.vec_single_1, self.vec_single_2)
        plot_residuals(self.vec_list_1, self.vec_list_2)

    def test_plot_slacks(self):
        plot_slacks(self.vec_single_1)
        plot_slacks(self.vec_list_1)
