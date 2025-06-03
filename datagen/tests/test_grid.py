import os
import sys
from unittest import TestCase
import numpy as np
from itertools import product

from datagen.src.dimensions import Dimension
from datagen.src.grid import gen_grid


class TestGrid(TestCase):
    os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

    def setUp(self):
        self.variable_borders = np.array([(0.0, 1.0)])
        self.dim_x = Dimension(
            variable_borders=self.variable_borders,
            n_cases=1,
            divs=2,
            borders=(0.0, 1.0),
            label="x",
            tolerance=0.01,
            independent_dimension=True
        )
        self.dim_y = Dimension(
            variable_borders=self.variable_borders,
            n_cases=1,
            divs=2,
            borders=(0.0, 1.0),
            label="y",
            tolerance=0.01,
            independent_dimension=True
        )
        self.dim_z = Dimension(
            variable_borders=self.variable_borders,
            n_cases=1,
            divs=2,
            borders=(0.0, 1.0),
            label="z",
            tolerance=0.01,
            independent_dimension=True
        )

    def test_gen_grid(self):
        print("RUNNING TEST GEN GRID")
        grid = gen_grid([self.dim_x, self.dim_y, self.dim_z])
        self.assertEqual(len(grid), 8)

        # Expected borders: all combinations of (0.0, 0.5) and (0.5, 1.0)
        intervals = [(0.0, 0.5), (0.5, 1.0)]
        expected_combinations = set(product(intervals, repeat=3))

        actual_combinations = set()
        for cell in grid:
            borders = tuple(dim.borders for dim in sorted(cell.dimensions, key=lambda d: d.label))
            actual_combinations.add(borders)

        self.assertEqual(expected_combinations, actual_combinations)
