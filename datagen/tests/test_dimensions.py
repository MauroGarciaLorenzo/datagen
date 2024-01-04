from unittest import TestCase

import numpy as np

from datagen import Dimension


class Test(TestCase):
    def setUp(self):
        """
        This code executes every time a subtest of this class is run.
        """
        variables = np.array([(0, 10), (0, 15), (0, 20), (0, 25)])
        n_cases = 10
        divs = None
        lower, upper = 0, 70
        is_true_dimension = True
        tolerance = 0.1
        dim1 = Dimension(variables, n_cases, divs, (lower, upper), is_true_dimension)
        dim1.tolerance = tolerance

        variables = np.array([(0, 0.1), (0, 0.1), (0, 0.1), (10, 25)])
        n_cases = 10
        divs = None
        lower, upper = 10, 25.3
        is_true_dimension = False
        tolerance = 0.1
        dim2 = Dimension(variables, n_cases, divs, (lower, upper), is_true_dimension)
        dim2.tolerance = tolerance
        self.samples = np.random.uniform(0, 70, 300)
        self.generator = np.random.default_rng(1)
        self.dims = {"dim1":dim1, "dim":dim2}

    def test_get_cases_extreme(self):
        for label, dim in self.dims.items():
            for sample in self.samples:
                variables_sum = dim.variables.sum(axis=0)
                if not variables_sum[0] <= sample <= variables_sum[1]:
                    self.assertRaises(
                        ValueError, dim.get_cases_extreme, label, sample, self.generator)
                else:
                    cases = dim.get_cases_extreme(label, sample, self.generator)
                    for idx in range(len(cases)):
                        self.assertAlmostEqual(sum(cases[idx]), sample,
                                               places=2)
                        for var in range(len(cases[idx])):
                            self.assertTrue(dim.variables[var, 0] <=
                                            cases[idx][var] <=
                                            dim.variables[var, 1])
                            self.assertIsNotNone(cases[idx][var])

    def test_get_cases_normal(self):

        for label, dim in self.dims.items():
            for sample in self.samples:
                variables_sum = dim.variables.sum(axis=0)
                if not variables_sum[0] <= sample <= variables_sum[1]:
                    self.assertRaises(
                        ValueError, dim.get_cases_extreme, label, sample, self.generator)
                else:
                    cases = dim.get_cases_extreme(label, sample, self.generator)
                    for idx in range(len(cases)):
                        for var in range(len(cases[idx])):
                            self.assertTrue(dim.variables[var, 0] <=
                                            cases[idx][var] <=
                                            dim.variables[var, 1])
                            self.assertIsNotNone(cases[idx][var])

