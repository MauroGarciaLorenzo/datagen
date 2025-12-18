import os
import unittest

import pandas as pd

class Test(unittest.TestCase):
    def test_join_and_cleanup_csvs(self):
        """
        Assert that join_and_cleanup_csvs can handle "undefined" csvs. The
        first explored csv will be df_damp_0.1.csv, which is undefined.
        """
        print("RUNNING TEST JOIN_AND_CLEANUP_CSVS")
        import shutil

        datagen_root = os.path.join(os.path.dirname(__file__), "..", "..")
        inputs_path = os.path.join(datagen_root, "datagen", "tests", "inputs",
                                   "test_join_and_cleanup_csvs")
        outputs_path = os.path.join(datagen_root, "datagen", "tests", "outputs",
                                    "test_join_and_cleanup_csvs")
        provisional_path = os.path.join(inputs_path, "..",
                                        "test_join_and_cleanup_csvs_provisional")

        if os.path.exists(provisional_path):
            shutil.rmtree(provisional_path)

        os.makedirs(provisional_path)

        for filename in os.listdir(inputs_path):
            if filename.endswith(".csv"):
                source_file = os.path.join(inputs_path, filename)
                shutil.copy(source_file, provisional_path)

        # Import target function
        from datagen.src.file_io import join_and_cleanup_csvs

        passed = True
        errors_failed = []

        try:
            print(f"\n=== Running test_join_and_cleanup_csvs ===\n",
                  flush=True)
            join_and_cleanup_csvs(provisional_path)
            print(f"\n=== test_join_and_cleanup_csvs success ===\n",
                  flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            passed = False
            print(
                f"\n=== test_join_and_cleanup_csvs failed ===\nException: {e}",
                flush=True)
            errors_failed.append(e)

        if passed:
            df_damp = pd.read_csv(os.path.join(provisional_path, "df_damp.csv"))

            # Run checks
            expected_df_damp_path = os.path.join(outputs_path, "df_damp.csv")
            expected_df_damp = pd.read_csv(expected_df_damp_path)

            pd.testing.assert_frame_equal(df_damp, expected_df_damp)

        # Final check
        else:
            for e in errors_failed:
                print(f"Error: {e}")

        shutil.rmtree(provisional_path)

        self.assertTrue(passed, f"Failed test_checkpointing")