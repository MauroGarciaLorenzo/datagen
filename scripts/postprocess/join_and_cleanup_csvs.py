import os
import re
import glob
import sys
import logging
import pandas as pd

def join_and_cleanup_csvs(dst_dir):
    """
    Joins all {var_name}_{cell_name}.csv files in dst_dir into one {var_name}.csv.
    Detects var_name correctly even if it contains underscores.
    Deletes the partial CSV files after joining.
    Adds a continuous line index to the final CSV.
    """
    all_csvs = glob.glob(os.path.join(dst_dir, "*.csv"))

    var_files = {}
    for f in all_csvs:
        fname = os.path.basename(f)
        if not fname.endswith(".csv"):
            continue

        # Match pattern: {var_name}_{cell_name}.csv, where cell_name = numbers + dots
        m = re.match(r"(.+)_([0-9.]+)\.csv$", fname)
        if m:
            var_name = m.group(1)
            var_files.setdefault(var_name, []).append(f)

    # Process each var_name group
    for var_name, files in var_files.items():
        print(f"Joining {len(files)} CSVs for {var_name}...")

        dfs = [pd.read_csv(f) for f in sorted(files)]
        joined_df = pd.concat(dfs, ignore_index=True)

        out_path = os.path.join(dst_dir, f"{var_name}.csv")
        joined_df.to_csv(out_path, index=True)
        print(f"Saved: {out_path}")

        # Delete partial CSVs
        logger = logging.getLogger(__name__)
        level = logger.getEffectiveLevel()
        level_name = logging.getLevelName(level)
        if level_name != "DEBUG":
            for f in files:
                os.remove(f)
                print(f"Deleted: {f}")
        else:
            print("Logging level is DEBUG; keeping partial files")


if __name__ == "__main__":
    args = sys.argv
    if len(args) < 2:
        print("User must provide the destination dir")
    join_and_cleanup_csvs(dst_dir=sys.argv[1])