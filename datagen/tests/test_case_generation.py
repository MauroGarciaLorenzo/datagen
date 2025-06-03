import os
import sys
from unittest import TestCase

import pandas as pd

from datagen.src.case_generation import gen_samples, gen_cases
from datagen.src.dimension_processing import process_p_load_dimension, \
    process_control_dimension, process_other_dimensions
from datagen.tests.test_sampling import create_dims, create_generator
from datagen.src.case_generation import process_p_cig_dimension
from datagen.src.dimensions import Dimension
from datagen.src.sampling import generate_columns


class Test(TestCase):
    os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

    def setUp(self):
        self.dims = create_dims()
        self.n_samples = 100
        self.generator = create_generator()
        self.df_samples = gen_samples(self.n_samples, self.dims, self.generator)

    def test_gen_samples(self):
        print("RUNNING TEST GEN SAMPLES")
        df_samples = gen_samples(self.n_samples, self.dims, self.generator)

        for dim in self.dims:
            self.assertTrue(all(df_samples[dim.label] >= dim.borders[0]))
            self.assertTrue(all(df_samples[dim.label] <= dim.borders[1]))

    def test_gen_cases(self):
        print("RUNNING TEST GEN CASES")
        cases_df, dims_df = gen_cases(self.df_samples, self.dims,
                                      self.generator)

        for dim in dims_df.columns[:-1]:
            dim_cols = [col for col in cases_df.columns if
                        col.startswith(dim + "_")]
            summed = cases_df[dim_cols].sum(axis=1).reset_index(drop=True)
            expected = dims_df[dim].reset_index(drop=True)

            for i, (exp, calc) in enumerate(zip(expected, summed)):
                with self.subTest(dimension=dim, index=i):
                    self.assertAlmostEqual(exp, calc, places=2)

    def test_process_p_cig_dimension(self):
        print("RUNNING TEST PROCESS P_CIG DIMENSION")
        # Create a p_cig dimension
        p_cig = Dimension(
            variable_borders=[(0, 50), (0, 50)],
            n_cases=1,
            divs=1,
            borders=(20, 100),
            label="p_cig"
        )

        # Extend samples_df with p_cig and perc_g_for
        df_samples = self.df_samples.copy()
        df_samples["p_cig"] = self.generator.uniform(20, 100, len(df_samples))
        df_samples["perc_g_for"] = self.generator.uniform(0.0, 1.0, len(df_samples))

        cases_df, dims_df = process_p_cig_dimension(df_samples, p_cig, self.generator)

        df_dims = pd.DataFrame(dims_df, columns=["p_g_for", "p_g_fol"])

        # Check p_cig == p_g_for + p_g_fol
        p_cig_expected = df_samples["p_cig"].reset_index(drop=True)
        p_cig_computed = (df_dims["p_g_for"] + df_dims["p_g_fol"]).reset_index(drop=True)

        for i, (expected, actual) in enumerate(zip(p_cig_expected, p_cig_computed)):
            with self.subTest(index=i):
                self.assertAlmostEqual(expected, actual, places=2)

        # Assert: total rows = n_samples * n_cases
        self.assertEqual(len(cases_df), len(df_samples) * p_cig.n_cases)


    def test_process_p_load_dimension(self):
        print("RUNNING TEST PROCESS P_LOAD DIMENSION")
        dim = Dimension(
            values=[0.5, 1.0],
            n_cases=1,
            label="p_load"
        )

        df_samples = self.df_samples.copy()
        df_samples["p_sg"] = self.generator.uniform(20, 100, len(df_samples))
        df_samples["p_cig"] = self.generator.uniform(20, 100, len(df_samples))

        cases_df, dims_df = process_p_load_dimension(df_samples, dim)

        for i in range(len(dims_df)):
            expected = 0.9 * (
                        df_samples["p_sg"].iloc[i] + df_samples["p_cig"].iloc[
                    i])
            self.assertAlmostEqual(dims_df.iloc[i, 0], expected, places=2)

            # Assert: case ≈ expected * dim.values
            for j, val in enumerate(dim.values):
                self.assertAlmostEqual(cases_df.iloc[i, j], expected * val,
                                       places=2)

        # Assert: total rows = n_samples * n_cases
        self.assertEqual(len(cases_df), len(df_samples) * dim.n_cases)

    def test_process_control_dimension(self):
        print("RUNNING TEST PROCESS CONTROL DIMENSION")
        dim = Dimension(
            label="tau_g_for",
            n_cases=2
        )

        df_samples = self.df_samples.copy()
        df_samples["tau_g_for"] = self.generator.uniform(5, 15,
                                                           len(df_samples))

        cases_df, dims_df = process_control_dimension(df_samples, dim)

        # Assert: all values are equal in dims and cases
        self.assertTrue(
            (cases_df["tau_g_for"] == dims_df["tau_g_for"]).all())

        # Assert: total rows = n_samples * n_cases
        self.assertEqual(len(cases_df), len(df_samples) * dim.n_cases)

    def test_process_other_dimensions(self):
        print("RUNNING TEST PROCESS OTHER DIMENSION")
        dim = Dimension(
            variable_borders=[(0, 10), (0, 20)],
            n_cases=2,
            divs=1,
            label="Dim1"
        )

        df_samples = self.df_samples.copy()
        df_samples["Dim1"] = self.generator.uniform(0, 30,
                                                         len(df_samples))

        cases_df, dims_df = process_other_dimensions(df_samples, dim,
                                                     self.generator)

        # Assert: no NaNs
        self.assertFalse(cases_df.isna().any().any())

        # Assert: same number of rows in dims and cases
        self.assertEqual(len(cases_df), len(dims_df))