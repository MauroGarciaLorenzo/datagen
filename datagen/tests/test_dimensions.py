import os
import sys
from unittest import TestCase
from datagen.src.case_generation import gen_samples

import numpy as np
import math

from datagen import Dimension


class Test(TestCase):
    os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

    def setUp(self):
        """
        This code executes every time a subtest of this class is run.
        """
        variables = np.array([(0, 10), (0, 15), (0, 20), (0, 25)])
        n_cases = 10
        n_samples = 300
        divs = None
        lower, upper = 0, 70
        self.dimension_tolerance = 0.3
        label = "Test"
        tolerance = 0.1
        dim1 = Dimension(variable_borders=variables, n_cases=n_cases, divs=divs,
                         borders=(lower, upper),
                         label=label)
        dim1.tolerance = tolerance
        self.generator = np.random.default_rng(1)

        variables = np.array([(0, 0.1), (0, 0.1), (0, 0.1), (10, 25)])
        n_cases = 10
        divs = None
        lower, upper = 10, 25.3
        label = "Test1"
        tolerance = 0.1
        dim2 = Dimension(variable_borders=variables, n_cases=n_cases, divs=divs,
                         borders=(lower, upper),
                         label=label)
        dim2.tolerance = tolerance
        self.dims = [dim1, dim2]
        self.samples = gen_samples(n_samples, self.dims, self.generator)

    def test_get_cases_extreme(self):
        print("RUNNING TEST PROCESS GET CASES EXTREME")
        for dim in self.dims:
            for _, s in self.samples.iterrows():
                sample = s[dim.label]
                variables_sum = dim.variable_borders.sum(axis=0)
                if not variables_sum[0] <= sample <= variables_sum[1]:
                    self.assertRaises(
                        ValueError, dim.get_cases_extreme, sample, self.generator)
                else:
                    cases = dim.get_cases_extreme(sample, self.generator)
                    for idx in range(len(cases)):
                        self.assertAlmostEqual(sum(cases[idx]), sample,
                                               places=2)
                        for var in range(len(cases[idx])):
                            self.assertTrue(dim.variable_borders[var, 0] <=
                                            cases[idx][var] <=
                                            dim.variable_borders[var, 1])
                            self.assertIsNotNone(cases[idx][var])

    def test_get_cases_normal(self):
        print("RUNNING TEST PROCESS GET CASES NORMAL")
        for dim in self.dims:
            for _, s in self.samples.iterrows():
                sample = s[dim.label]
                variables_sum = dim.variable_borders.sum(axis=0)
                if not variables_sum[0] <= sample <= variables_sum[1]:
                    self.assertRaises(
                        ValueError, dim.get_cases_normal, sample,
                        self.generator)
                else:
                    cases = dim.get_cases_normal(sample, self.generator)
                    for idx in range(len(cases)):
                        print(sum(cases[idx]), sample)
                        # Assert that the sum of the resulting case is within
                        # dimension_tolerance of the range of the dimension
                        self.assertLessEqual(sum(cases[idx]) - sample,
                                             self.dimension_tolerance *
                                             (dim.borders[1] - dim.borders[0]))
                        for var in range(len(cases[idx])):
                            self.assertTrue(dim.variable_borders[var, 0] <=
                                            cases[idx][var] <=
                                            dim.variable_borders[var, 1])
                            self.assertIsNotNone(cases[idx][var])

