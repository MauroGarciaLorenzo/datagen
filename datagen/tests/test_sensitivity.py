"""
Unit tests to verify that the execution of the datagen using the sensitivity
analysis works properly
"""

import os
import shutil
import sys
import unittest
import yaml


class Test(unittest.TestCase):
    results_dir = "results"
    old_results_dir = "results_old"
    current_case = None

    def test_sensitivity(self):
        """
        Run 'run_datagen_ACOPF.py' to test the sensitivity analysis functionality.
        To do so, the test uses the test_sensitivity_obj_func function, which
        returns 1 only depending on the value of the dimension tau_Dim_0, so
        this is the only "important" dimension.
        """

        n_samples = 10
        n_cases = 10
        max_depth = 1
        seed = 17

        # Directory initialization
        yaml_path = '../../setup/test_setup.yaml'
        working_dir = '.'
        if os.path.isdir(self.results_dir):
            shutil.rmtree(self.results_dir)
        os.makedirs(self.results_dir)
        if not os.path.isdir(self.old_results_dir):
            os.makedirs(self.old_results_dir)

        # Load YAML file
        with open(yaml_path) as stream:
            base_yaml = yaml.safe_load(stream)

        from run_datagen_sensitivity import main

        # Loop over hyperparameters
        passed = True
        failed = []
        errors_failed = []
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

            # Redirect stdout to capture the output
            from io import StringIO

            # Get the output of the main function
            captured_output = StringIO()
            sys.stdout = captured_output
            main(setup_path=yaml_path, working_dir=working_dir)
            output = captured_output.getvalue()

            # Parse the output to check dimension divisions
            dim_divisions = {}
            for line in output.split('\n'):
                if 'Dimension: tau_Dim_' in line:
                    parts = line.split(', divisions: ')
                    dim_name = parts[0].split('Dimension: ')[1].strip()
                    divisions = int(parts[1].strip())
                    dim_divisions[dim_name] = divisions

            # Assert the divisions
            assert dim_divisions.get('tau_Dim_0',
                                     None) == 4, f"tau_Dim_0 should have 4 divisions, got {dim_divisions.get('tau_Dim_0', None)}"
            for dim in [f'tau_Dim_{i}' for i in range(1, 6)]:
                assert dim_divisions.get(dim,
                                         None) == 1, f"{dim} should have 1 division, got {dim_divisions.get(dim, None)}"

            print(f"\n=== Everything went fine ===\n", flush=True)
            print("Dimension divisions validation passed!")
        except Exception as e:
            passed = False
            print(f"\n=== Error executing test sensitivity")
            print(f"Exception caught: {e}", flush=True)
            failed.append((n_samples, n_cases, max_depth))
            errors_failed.append(e)

        # Assert executions that failed before running the subtests
        if not passed:
            for fail, error in zip(failed, errors_failed):
                print(f"Failed with configurations: {fail}")
                print(f"Errors: {error}")
        self.assertTrue(passed, f"Some configurations failed: {failed}. Errors: {errors_failed}")


if __name__ == '__main__':
    unittest.main()
