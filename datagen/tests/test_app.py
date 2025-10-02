"""
Unit tests to verify that executions of datagen run properly. Also test the
resulting csv files for data integrity.

To run the tests and send the output to a file, run from the terminal:
>> python -m unittest test_app | tee test_app.txt
"""
import os
import pandas as pd
import re
import sys
import unittest
import yaml


class Test(unittest.TestCase):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    current_case = None
    working_dir = os.getcwd()

    def test_sensitivity(self):
        """
        Run 'run_datagen_ACOPF.py' to test the sensitivity analysis functionality.
        The function 'test_sensitivity_obj_func' returns 1 only depending on
        the value of dimension tau_Dim_0, so this should be the only important one.
        """
        print("RUNNING TEST SENSITIVITY")
        n_samples = 10
        n_cases = 10
        max_depth = 1
        seed = 17
        yaml_path = '../../setup/test_setup.yaml'

        # Load YAML config
        with open(yaml_path) as stream:
            base_yaml = yaml.safe_load(stream)

        from run_datagen_sensitivity import main

        passed = True
        failed = []
        errors_failed = []

        # Update YAML parameters
        base_yaml['n_samples'] = n_samples
        base_yaml['n_cases'] = n_cases
        base_yaml['max_depth'] = max_depth
        base_yaml['seed'] = seed
        self.current_case = (n_samples, n_cases, max_depth, seed)

        # Save updated YAML
        with open(yaml_path, 'w') as stream:
            yaml.dump(base_yaml, stream)

        try:
            print(f'\n{"=" * 60}\n'
                  f"=== Running with n_samples={n_samples}, "
                  f"n_cases={n_cases}, max_depth={max_depth} ==="
                  f'\n{"=" * 60}\n', flush=True)

            # Get logger output
            path_results = main(setup_path=yaml_path, working_dir=self.working_dir)
            log_file = os.path.join(path_results, "log.txt")

            with open(log_file, 'r') as f:
                log_output = f.read()

            # Parse log output for dimension divisions
            dim_divisions = {}
            for line in log_output.split('\n'):
                if 'Dimension: tau_Dim_' in line:
                    parts = line.split(', divisions: ')
                    dim_name = parts[0].split('Dimension: ')[1].strip()
                    divisions = int(parts[1].strip())
                    dim_divisions[dim_name] = divisions

            # Validate output
            assert dim_divisions.get('tau_Dim_0', None) == 4, \
                f"tau_Dim_0 should have 4 divisions, got {dim_divisions.get('tau_Dim_0', None)}"
            for dim in [f'tau_Dim_{i}' for i in range(1, 6)]:
                assert dim_divisions.get(dim, None) == 1, \
                    f"{dim} should have 1 division, got {dim_divisions.get(dim, None)}"

            print(f"\n=== Everything went fine ===\n", flush=True)
            print("Dimension divisions validation passed!")

        except Exception as e:
            import traceback
            passed = False
            print(f"\n=== Error executing test sensitivity")
            print(f"Exception caught: {e}", flush=True)
            tb = traceback.format_exc()
            print(tb)
            failed.append((n_samples, n_cases, max_depth))
            errors_failed.append(
                tb)

        if not passed:
            for fail, error in zip(failed, errors_failed):
                print(f"Failed with configurations: {fail}")
                print(f"Errors: {error}")
        self.assertTrue(passed,
                        f"Some configurations failed: {failed}. Errors:\n" + "\n".join(
                            errors_failed))


    def test_ACOPF(self):
        """
        Run 'run_datagen_ACOPF.py' with different configurations
        of parameters:
            n_samples
            n_cases
            max_depth
            seed
        and check for errors during execution.
        """
        print("RUNNING TEST ACOPF")
        # Cases initialization
        hyperparam_combinations = [  # n_samples, n_cases, max_depth
            [1, 1, 1]
        ]
        seeds = [17]
        # Repeat case for each seed
        hyperparam_combinations = [
            hyparams + [seed] for hyparams in hyperparam_combinations
            for seed in seeds
        ]

        expected_files = {
            "df_computing_times.csv",
            "df_damp.csv",
            "df_freq.csv",
            "df_imag.csv",
            "df_op.csv",
            "df_real.csv",
            "cases_df.csv",
            "dims_df.csv",
            "execution_logs.txt",
            "execution_time.csv"
        }

        # Directory initialization
        yaml_path = '../../setup/test_setup.yaml'

        # Load YAML file
        with open(yaml_path) as stream:
            base_yaml = yaml.safe_load(stream)

        # Get module
        sys.path.append('../../')
        from run_datagen_ACOPF import main

        # Loop over hyperparameters
        passed = True
        failed = []
        errors_failed = []
        for n_samples, n_cases, max_depth, seed in hyperparam_combinations:
            # Update YAML
            base_yaml['n_samples'] = n_samples
            base_yaml['n_cases'] = n_cases
            base_yaml['max_depth'] = max_depth
            base_yaml['seed'] = seed
            # Set current case for messaging purposes
            self.current_case = (n_samples, n_cases, max_depth, seed)

            # Save YAML
            with open(yaml_path, 'w') as stream:
                yaml.dump(base_yaml, stream)

            # Run
            try:
                print(f'\n{"".join(["="] * 60)}\n'
                      f"=== Running with n_samples={n_samples}, "
                      f"n_cases={n_cases}, max_depth={max_depth} ==="
                      f'\n{"".join(["="] * 60)}\n', flush=True)
                path_results = main(setup_path=yaml_path, working_dir=self.working_dir, warmup=False)
                print(f"\n=== Everything went fine ===\n", flush=True)
            except Exception as e:
                passed = False
                print(f"\n=== Error with n_samples={n_samples}, "
                      f"n_cases={n_cases}, max_depth={max_depth} ===\n",
                      flush=True)
                print(f"Exception caught: {e}", flush=True)
                failed.append((n_samples, n_cases, max_depth))
                errors_failed.append(e)
                continue

            cases_df = pd.read_csv(os.path.join(path_results, "cases_df.csv"))
            dims_df = pd.read_csv(os.path.join(path_results, "dims_df.csv"))

            # Launch tests
            self.check_output_files(path_results, expected_files)
            self.no_duplicate_rows(cases_df, dims_df)
            self.column_sums_match(cases_df, dims_df)

        # Assert executions that failed before running the subtests
        if not passed:
            for fail, error in zip(failed, errors_failed):
                print(f"Failed with configurations: {fail}")
                print(f"Errors: {error}")
        self.assertTrue(passed, f"Some configurations failed: {failed}")


    def test_depth(self):
        """
        Run 'run_dummy.py' with different depths and check for errors. In this
        case, test_setup yaml is modified to use different depth cfgs.
        """
        print("RUNNING TEST DEPTH")

        # Parameters
        depths = [3, 5]
        n_samples = 3
        n_cases = 3
        seed = 16

        # Paths
        yaml_path = '../../setup/test_setup.yaml'

        # Load YAML
        with open(yaml_path) as stream:
            base_yaml = yaml.safe_load(stream)

        # Import target function
        sys.path.append('../../')
        from run_dummy import main

        passed = True
        failed = []
        errors_failed = []

        for depth in depths:
            # Update YAML
            base_yaml['n_samples'] = n_samples
            base_yaml['n_cases'] = n_cases
            base_yaml['max_depth'] = depth
            base_yaml['seed'] = seed

            # Save YAML
            with open(yaml_path, 'w') as stream:
                yaml.dump(base_yaml, stream)

            try:
                print(f"\n=== Running with max_depth={depth} ===\n",
                      flush=True)
                path_results = main(setup_path=yaml_path)
                print(f"\n=== Success with max_depth={depth} ===\n",
                      flush=True)
            except Exception as e:
                passed = False
                print(
                    f"\n=== Failed with max_depth={depth} ===\nException: {e}",
                    flush=True)
                failed.append(depth)
                errors_failed.append(e)
                continue


            cases_df = pd.read_csv(os.path.join(path_results, "cases_df.csv"))
            dims_df = pd.read_csv(os.path.join(path_results, "dims_df.csv"))

            # Run checks
            self.no_duplicate_rows(cases_df, dims_df)
            self.column_sums_match(cases_df, dims_df)

        # Final check
        if not passed:
            for d, e in zip(failed, errors_failed):
                print(f"Failed depth: {d}\nError: {e}")
        self.assertTrue(passed, f"Failures for depths: {failed}")


    def test_variability(self):
        """
        Run run_variability.py with different seeds and check for errors.
        The seeds are used to initialize a generator that creates the variable
        borders. This ensures that the execution is tested with different
        variable ranges.
        """
        print("RUNNING TEST VARIABILITY")

        # Parameters
        depth = 2
        n_samples = 2
        n_cases = 2
        seeds = [16, 17, 18]

        # Paths
        yaml_path = '../../setup/test_setup.yaml'

        # Load YAML
        with open(yaml_path) as stream:
            base_yaml = yaml.safe_load(stream)

        # Import target function
        sys.path.append('../../')
        from run_variability import main

        passed = True
        failed = []
        errors_failed = []

        for seed in seeds:
            # Update YAML
            base_yaml['n_samples'] = n_samples
            base_yaml['n_cases'] = n_cases
            base_yaml['max_depth'] = depth
            base_yaml['seed'] = seed

            # Save YAML
            with open(yaml_path, 'w') as stream:
                yaml.dump(base_yaml, stream)

            try:
                print(f"\n=== Running with max_depth={depth} ===\n",
                      flush=True)
                path_results = main(setup_path=yaml_path)
                print(f"\n=== Success with max_depth={depth} ===\n",
                      flush=True)
            except Exception as e:
                passed = False
                print(
                    f"\n=== Failed with max_depth={depth} ===\nException: {e}",
                    flush=True)
                failed.append(depth)
                errors_failed.append(e)
                continue

            cases_df = pd.read_csv(os.path.join(path_results, "cases_df.csv"))
            dims_df = pd.read_csv(os.path.join(path_results, "dims_df.csv"))

            # Run checks
            self.no_duplicate_rows(cases_df, dims_df)
            self.column_sums_match(cases_df, dims_df)

        # Final check
        if not passed:
            for d, e in zip(failed, errors_failed):
                print(f"Failed depth: {d}\nError: {e}")
        self.assertTrue(passed, f"Failures for depths: {failed}")


    def check_output_files(self, case_dir, expected_files):
        files_in_dir = set(os.listdir(case_dir))
        missing_files = expected_files - files_in_dir

        # Assert that every file exists
        self.assertFalse(missing_files, f"Missing files: {missing_files}")

        for filename in expected_files:
            file_path = os.path.join(case_dir, filename)

            # Assert file exists and is not empty
            self.assertTrue(os.path.isfile(file_path),
                            f"{filename} not found.")
            self.assertGreater(os.path.getsize(file_path), 0,
                               f"{filename} is empty.")

            # Asserts for the csvs
            if "df" in filename and filename.endswith(".csv"):
                df = pd.read_csv(file_path)

                if filename in ["dims_df.csv", "cases_df.csv"]:
                    # dims_df and cases_df has no empty or NaN value
                    self.assertFalse(
                        df.isnull().values.any() or (df == '').values.any(),
                        f"{filename} contains empty or NaN values.")
                else:
                    # case_id and Stability have value != NaN
                    self.assertIn("case_id", df.columns,
                                  f"{filename} missing 'case_id' column.")
                    self.assertIn("Stability", df.columns,
                                  f"{filename} missing 'Stability' column.")
                    self.assertFalse(
                        df[["case_id", "Stability"]].isnull().values.any() or
                        (df[["case_id", "Stability"]] == '').values.any(),
                        f"{filename} contains empty or NaN values in 'case_id' or 'Stability'.")

    def no_duplicate_rows(self, cases_df, dims_df):
        """ Check if there are any duplicated rows in cases_df. """
        cases_duplicates = cases_df[cases_df.duplicated()]
        dims_duplicates = dims_df[dims_df.duplicated()]

        # Assert no duplicates exist
        self.assertTrue(
            cases_duplicates.empty,
            f"Duplicate rows found in cases_df:\n{cases_duplicates}")
        self.assertTrue(
            dims_duplicates.empty,
            f"Duplicate rows found in dims_df:\n{dims_duplicates}")

    def column_sums_match(self, cases_df, dims_df):
        """
        Check that sums of cases_df columns match the overall values in
        dims_df.
        """
        # Use a regex to find columns ending with '_VarXX'
        var_column_pattern = re.compile(r'^(.*)_Var\d+$')

        # Dictionary to hold column groups in cases_df that need to be summed
        columns_to_sum = {}

        for col in cases_df.columns:
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
                if col in dims_df.columns:
                    with self.subTest(col=col):
                        self.assertTrue(
                            cases_df[col].equals(dims_df[col]),
                            f"Column {col} in cases_df does not match"
                            f" the same column in dims_df")

        # Now handle the columns that need to be summed
        for prefix, related_columns in columns_to_sum.items():
            if prefix in dims_df.columns:
                # Sum the related columns in cases_df
                cases_sum = cases_df[related_columns].sum(axis=1)
                dims_col = dims_df[prefix]

                # Assert that the sums match a column in dims_df
                with self.subTest(prefix=prefix,
                                  msg=f"Current case: {self.current_case}"):
                    pd.testing.assert_series_equal(
                        cases_sum, dims_col, check_names=False, rtol=1e-3)
            else:
                self.fail(f"Column {prefix} not found in dims_df!")


if __name__ == '__main__':
    unittest.main()
