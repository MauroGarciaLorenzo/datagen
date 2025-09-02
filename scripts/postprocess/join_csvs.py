import glob
import os
import sys

import pandas as pd


def main(results_dir, output_file="cases_df_join.csv"):
    """
    Join all cases_df_*.csv files into a single CSV.
    Keeps header only once.
    """
    # Find all matching files
    csv_files = sorted(glob.glob(os.path.join(results_dir, "cases_df_*.csv")))

    if not csv_files:
        raise FileNotFoundError("No cases_df_*.csv files found in directory")

    # Load and concatenate
    dfs = [pd.read_csv(f) for f in csv_files]
    final_df = pd.concat(dfs, ignore_index=True)

    # Save the result
    output_path = os.path.join(results_dir, output_file)
    final_df.to_csv(output_path, index=False)

    return final_df

if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        main(sys.argv[1])
