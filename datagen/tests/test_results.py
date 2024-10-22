"""
Verify the goodness of produced csv files. Delete ./datagen/tests/results
to obtain new cases!
"""
import os
import re
import shutil

import pandas as pd
import sys
import unittest


class TestResults(unittest.TestCase):
    results_dir = "results"
    
    def setUp(self):
        """
        Run a single case
        """
        # Init - Save results to ./datagen/test/ not to mix with normal results
        yaml_path = '../../setup/test_setup.yaml'
        working_dir = '.'

        # Get module
        sys.path.append('../../')
        from run_datagen_ACOPF import main

        # Run case if directory is empty
        if not os.listdir(self.results_dir):
            print(f'\n{"".join(["="] * 20)}\n'
                  f"=== Running case ==="
                  f'\n{"".join(["="] * 20)}\n')
            main(setup_path=yaml_path, working_dir=working_dir)
            print(f"\n=== Case ran correctly ===\n")

        for directory in os.listdir(self.results_dir):
            case_dir = os.path.join(self.results_dir, directory)
            self.cases_df = pd.read_csv(os.path.join(case_dir, "cases_df.csv"),
                                        index_col=0)
            self.dims_df = pd.read_csv(os.path.join(case_dir, "dims_df.csv"),
                                       index_col=0)
            break  # TODO: do a real loop and test several cases

    def test_no_duplicate_rows(self):
        """ Check if there are any duplicated rows in cases_df. """
        cases_duplicates = self.cases_df[self.cases_df.duplicated()]
        dims_duplicates = self.dims_df[self.dims_df.duplicated()]

        # Assert no duplicates exist
        self.assertTrue(cases_duplicates.empty,
                        f"Duplicate rows found in cases_df:\n{cases_duplicates}")
        self.assertTrue(dims_duplicates.empty,
                        f"Duplicate rows found in dims_df:\n{dims_duplicates}")

    def test_column_sums_match(self):
        # Use a regex to find columns ending with '_VarXX'
        var_column_pattern = re.compile(r'^(.*)_Var\d+$')

        # Dictionary to hold column groups in cases_df that need to be summed
        columns_to_sum = {}

        for col in self.cases_df.columns:
            match = var_column_pattern.match(col)
            if match:
                # Extract the prefix (e.g., 'col' from 'col_Var1')
                prefix = match.group(1)
                # Group columns by their prefix to sum them later
                if prefix not in columns_to_sum:
                    columns_to_sum[prefix] = []
                columns_to_sum[prefix].append(col)
            else:
                # Direct comparison for columns without '_VarXX'
                if col in self.dims_df.columns:
                    with self.subTest(col=col):
                        self.assertTrue(
                            self.cases_df[col].equals(self.dims_df[col]),
                            f"Column {col} in cases_df does not match"
                            f" the same column in dims_df")
                else:
                    print(f"Warning: column {col} found in cases_df.csv "
                          f"but not in dims_df.csv")

        # Now handle the columns that need to be summed
        for prefix, related_columns in columns_to_sum.items():
            if prefix in self.dims_df.columns:
                # Sum the related columns in cases_df
                cases_sum = self.cases_df[related_columns].sum(axis=1)
                dims_col = self.dims_df[prefix]

                # Assert that the sums match a column in dims_df
                with self.subTest(prefix=prefix):
                    pd.testing.assert_series_equal(cases_sum, dims_col,
                                                   check_names=False)
            else:
                self.fail(f"Column {prefix} not found in dims_df!")


if __name__ == '__main__':
    unittest.main()
