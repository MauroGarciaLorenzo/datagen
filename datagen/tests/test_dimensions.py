from unittest import TestCase

import numpy as np

from datagen import Dimension


class Test(TestCase):
    def setUp(self):
        """
        This code executes every time a subtest of this class is run.
        """
        variables = np.array([(0, 10), (0, 15), (0, 20), (0, 25)])
        n_cases = 3
        divs = None
        lower, upper = 0, 70
        label = "Test"
        tolerance = 0.1
        dim = Dimension(variables, n_cases, divs, (lower, upper), label)
        dim.tolerance = tolerance
        self.dim = dim

    def test_get_cases_extreme(self):
        cases = self.dim.get_cases_extreme(10)
        for case in cases:
            self.assertAlmostEquals(sum(case), 10, self.dim.tolerance)
