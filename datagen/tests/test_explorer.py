import os
import sys
from unittest import TestCase
import pandas as pd
import numpy as np

from datagen.src.case_generation import gen_cases
from datagen.src.dimensions import Dimension
from datagen.src.grid import gen_grid
from datagen.src.objective_function import dummy
from datagen.tests.test_sampling import create_generator
from datagen.src.explorer import explore_grid
from datagen.src.file_io import join_and_cleanup_csvs


class TestExplorer(TestCase):
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')))
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    def setUp(self):
        # Create a dummy dimension
        self.variable_borders = np.array([(0.0, 1.0), (0.0, 1.0)])
        self.dimension = Dimension(
            variable_borders=self.variable_borders,
            n_cases=1,
            divs=2,
            borders=(0.0, 1.0),
            label="x",
            tolerance=0.01
        )

        self.grid = gen_grid([self.dimension])

        # Generate dims_df and cases_df according to dimension
        self.samples_df = pd.DataFrame({"x": [0.25, 0.75]})
        self.generator = create_generator()

        stabilities = [1, 0]
        self.cases_df, self.dims_df = gen_cases(samples_df=self.samples_df,
                                                dimensions=[self.dimension],
                                                generator=self.generator)
        self.cases_df["Stability"] = stabilities

        # Other required parameters
        self.ax = None
        self.depth = 1
        self.n_samples = 2
        self.use_sensitivity = False
        self.max_depth = 2
        self.divs_per_cell = 2
        self.feasible_rate = 0.1
        self.func_params = {}
        self.func = dummy
        self.dataframes = {}
        self.parent_entropy = 0
        self.parent_name = "0"
        self.dst_dir = "results"
        self.chunk_length = 10
        self.entropy_threshold = 0.05
        self.delta_entropy_threshold = 0
        self.df_names=set()

    def test_explore_grid(self):
        print("RUNNING TEST EXPLORE GRID")
        result = explore_grid(
            ax=self.ax,
            cases_df=pd.DataFrame(),
            grid=self.grid,
            depth=self.depth,
            dims_df=pd.DataFrame(),
            func=self.func,
            n_samples=self.n_samples,
            use_sensitivity=self.use_sensitivity,
            max_depth=self.max_depth,
            divs_per_cell=self.divs_per_cell,
            generator=self.generator,
            feasible_rate=self.feasible_rate,
            func_params=self.func_params,
            parent_entropy=self.parent_entropy,
            parent_name=self.parent_name,
            dst_dir=self.dst_dir,
            chunk_length=self.chunk_length,
            entropy_threshold=self.entropy_threshold,
            delta_entropy_threshold=self.delta_entropy_threshold,
            df_names=self.df_names
        )
        join_and_cleanup_csvs(self.dst_dir)

        cases_df = pd.read_csv(os.path.join(self.dst_dir, "cases_df.csv"))
        dims_df = pd.read_csv(os.path.join(self.dst_dir, "dims_df.csv"))

        # Assert that the number of cases equals the initial cases plus the
        # cases that will be generated in the remaining levels:
        # (n_cases * n_samples * n_divs)
        self.assertEqual(len(cases_df), (self.max_depth-self.depth)
                         * self.n_samples * self.dimension.n_cases * self.dimension.divs)

        # Assert at least one case per subinterval in [0, 1] split in 4
        values = dims_df["x"].values
        intervals = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
        for low, high in intervals:
            self.assertTrue(
                np.any((values >= low) & (values < high)),
                f"No values found in interval [{low}, {high})"
            )
