"""
To run the tests and send the output to a file, run from the terminal:
>> python -m unittest test_app | tee test_app.txt
"""
from itertools import product

import numpy as np
import sys
import yaml

from unittest import TestCase


class Test(TestCase):
    def test_app(self):
        """
        Run 'run_datagen_ACOPF.py' with different configurations of
        parameters:
            n_samples
            n_cases
            max_depth
        and check for errors during execution.
        """
        # Init
        yaml_path = '../../setup/test_setup.yaml'
        working_dir = '../../'
        min_n = 1
        max_n = 3
        n_samples, n_cases, max_depth = \
            [np.arange(min_n, max_n + 1).tolist()] * 3
        hyperparam_combinations = product(n_samples, n_cases, max_depth)

        # Load YAML file
        with open(yaml_path) as stream:
            base_yaml = yaml.safe_load(stream)

        # Get module
        sys.path.append('../../')
        from run_datagen_ACOPF import main

        # Loop over hyperparameters
        passed = True
        for n_samples, n_cases, max_depth in hyperparam_combinations:
            # Update YAML
            base_yaml['n_samples'] = n_samples
            base_yaml['n_cases'] = n_cases
            base_yaml['max_depth'] = max_depth

            # Save YAML
            with open(yaml_path, 'w') as stream:
                yaml.dump(base_yaml, stream)

            # Run
            failed = []
            errors_failed = []
            try:
                print(f'\n{"".join(["="] * 60)}\n'
                      f"=== Running with n_samples={n_samples}, "
                      f"n_cases={n_cases}, max_depth={max_depth} ==="
                      f'\n{"".join(["="] * 60)}\n', flush=True)
                main(setup_path=yaml_path, working_dir=working_dir)
                print(f"\n=== Evertything went fine ===\n", flush=True)
            except Exception as e:
                passed = False
                print(f"\n=== Error with n_samples={n_samples}, "
                      f"n_cases={n_cases}, max_depth={max_depth} ===\n",
                      flush=True)
                print(f"Exception caught: {e}", flush=True)
                failed.append((n_samples, n_cases, max_depth))
                errors_failed.append(e)

        # Assert
        if not passed:
            for fail, error in zip(failed, errors_failed):
                print(f"Failed with configurations: {fail}")
                print(f"Errors: {error}")
        self.assertTrue(passed, f"Some configurations failed: {failed}")
