from unittest import TestCase

import numpy as np

from datagen import Dimension


class Test(TestCase):
    def setUp(self):
        """
        This code executes every time a subtest of this class is run.
        """
        variables = np.array([(0, 10), (0, 15), (0, 20), (0, 25)])
        n_cases = 300
        divs = None
        lower, upper = 0, 70
        label = "Test"
        tolerance = 0.1
        dim = Dimension(variables, n_cases, divs, (lower, upper), label)
        dim.tolerance = tolerance
        self.dim = dim

    def test_get_cases_extreme(self):
        cases = self.dim.get_cases_extreme(10)
        for idx in range(len(cases)):
            self.assertAlmostEqual(sum(cases[idx]), 10., places=5)
            for var in range(len(cases[idx])):
                self.assertTrue(self.dim.variables[var, 0] <=
                                cases[idx][var] <=
                                self.dim.variables[var, 1])

    def test_get_cases_normal(self):
        cases = self.dim.get_cases_normal(10)
        for idx in range(len(cases)):
            for var in range(len(cases[idx])):
                self.assertTrue(self.dim.variables[var, 0] <=
                                cases[idx][var] <=
                                self.dim.variables[var, 1])