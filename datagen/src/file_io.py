import os
import random
import re
import sys
from datetime import datetime
import pandas as pd
from datagen.src.logger import logger
import csv
import glob
try:
    from datagen.src.constants import NAN_COLUMN_NAME
except ImportError:
    NAN_COLUMN_NAME = "undefined"


def write_dataframes_to_excel(df_dict, path, filename):
    excel_file_path = os.path.join(path, filename)
    # Create a Pandas Excel writer using xlsxwriter as the engine
    with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
        # Iterate over each key-value pair in the dictionary
        for sheet_name, df in df_dict.items():
            # Write each DataFrame to a separate sheet with the sheet name as the key
            if isinstance(df, dict):
                df = pd.DataFrame(df)
            if isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                logger.warning("Not writing %s. Not a DataFrame or Series",
                               sheet_name)


def save_dataframes(output_dataframes_array, path_results, seed):
    index = 0
    for dataframe in output_dataframes_array:
        if dataframe is None:
            continue
        for key, value in dataframe.items():
            cu = os.environ.get("COMPUTING_UNITS")
            filename = f"cu{cu}_case_{str(index)}_{key}_seed{str(seed)}"
            if not os.path.exists(path_results):
                os.makedirs(path_results)
            if isinstance(value, dict):
                write_dataframes_to_excel(
                    value, path_results, f"{filename}.xlsx")
            else:
                pd.DataFrame.to_csv(
                    value, os.path.join(path_results, f"{filename}.csv"))
        index += 1


def save_df(dataframe, dst_dir, cell_name, var_name):
    """Append cases_df to dst_dir."""
    os.makedirs(dst_dir, exist_ok=True)
    csv_path = os.path.join(dst_dir, f"{var_name}_{cell_name}.csv")
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Write header only if file does not exist
        if not file_exists:
            writer.writerow(dataframe.columns)

        # Write rows
        for row in dataframe.itertuples(index=False, name=None):
            writer.writerow(row)


def save_results(execution_logs, dst_dir, execution_time):
    """
    Save set of results including the main cases/dims dataframes plus
    execution logs and all elements inside the output_dataframes dictionary,
    mostly dataframes. These dataframe rows must follow the same sorting
    according to the case ids.
    """
    if dst_dir is None:
        dst_dir = "results"

    execution_time_df = pd.DataFrame(
        {"execution_time": [execution_time]})

    execution_time_df.to_csv((os.path.join(dst_dir, "execution_time.csv")), index=False)


def save_execution_logs(children_info, dst_dir):
    log_path = os.path.join(dst_dir, "execution_logs.txt")

    with open(log_path, "a") as log_file:
        for log_entry in children_info:
            log_file.write("Dimensions:\n")
            for dim in log_entry[0]:
                log_file.write(f"{dim}\n")
            log_file.write(f"Entropy: {log_entry[1]}\n")
            log_file.write(f"Delta Entropy: {log_entry[2]}\n")
            log_file.write(f"Depth: {log_entry[3]}\n")
            log_file.write("\n")


def init_dst_dir(calling_module, seed, n_cases, n_samples, max_depth,
                 working_dir, ax, dimensions):
    from .utils import clean_dir
    # Get slurm job id
    slurm_job_id = os.getenv("SLURM_JOB_ID", default=None)
    slurm_str = ""
    if slurm_job_id:
        slurm_str = f"_slurm{slurm_job_id}"

    # Get slurm n_nodes
    slurm_num_nodes = os.environ.get('SLURM_JOB_NUM_NODES', default=None)
    slurm_nodes_str = ""
    if slurm_num_nodes:
        slurm_nodes_str = f"_nodes{slurm_num_nodes}"

    # Get computing units assigned to the objective function
    cu = os.environ.get("COMPUTING_UNITS", default=None)
    cu_str = ""
    if cu:
        cu_str = f"_cu{cu}"

    # Get random number
    rnd_num = random.randint(1000, 9999)

    # Get timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dir_name = f"{calling_module}{slurm_str}{cu_str}{slurm_nodes_str}_LF09_seed{seed}_nc{n_cases}" \
               f"_ns{n_samples}_d{max_depth}_{timestamp}_{rnd_num}"
    path_results = os.path.join(
        working_dir, "results", dir_name)

    clean_dir(os.path.join(path_results))
    if ax is not None and len(dimensions) == 2:
        clean_dir(os.path.join(path_results, "figures"))

    return path_results


def log_cell_info(cell_name, depth, parent_entropy, delta_entropy, feasible_ratio, status, dst_dir):
    csv_path = os.path.join(dst_dir, "cell_info.csv")
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Cell Name", "Depth", "Entropy", "Delta Entropy", "Feasible Ratio", "Status"])
        writer.writerow([cell_name, depth, parent_entropy, delta_entropy, feasible_ratio, status])


def clean_incomplete_cells(dst_dir):
    """
    Removes all CSVs for cells that do not have every required df_name.
    Example filenames: cases_df_0.1.2.csv, samples_df_0.1.2.csv, ...
    """
    pattern = re.compile(r"(.+)_([0-9.]+)\.csv$")
    csv_files = glob.glob(os.path.join(dst_dir, "*.csv"))

    cell_to_dfs = {}   # {cell_name: set of df_names}
    all_df_names = set()

    for f in csv_files:
        fname = os.path.basename(f)
        m = pattern.match(fname)
        if not m:
            continue
        df_name, cell_name = m.groups()
        all_df_names.add(df_name)
        cell_to_dfs.setdefault(cell_name, set()).add(df_name)

    # Find incomplete cells
    incomplete_cells = [cell for cell, dfs in cell_to_dfs.items()
                        if dfs != all_df_names]

    # Delete their CSVs
    for cell_name in incomplete_cells:
        for df_name in all_df_names:
            file_path = os.path.join(dst_dir, f"{df_name}_{cell_name}.csv")
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"[CLEAN] Deleted incomplete file: {file_path}", flush=True)



def join_and_cleanup_csvs(dst_dir):
    """
    Joins all {var_name}_{cell_name}.csv files in dst_dir into one {var_name}.csv.
    Detects var_name correctly even if it contains underscores.
    Deletes the partial CSV files after joining.
    Adds a continuous line index to the final CSV.
    Prefers a first CSV whose header does not contain 'undefined' or NAN_COLUMN_NAME.
    Falls back to alphabetical order if none qualify.
    """
    all_csvs = glob.glob(os.path.join(dst_dir, "*.csv"))

    var_files = {}
    for f in all_csvs:
        fname = os.path.basename(f)
        if not fname.endswith(".csv"):
            continue

        m = re.match(r"(.+)_([0-9.]+)\.csv$", fname)
        if m:
            var_name = m.group(1)
            var_files.setdefault(var_name, []).append(f)

    for var_name, files in var_files.items():
        print(f"Joining {len(files)} CSVs for {var_name}...")

        sorted_files = sorted(files)
        valid_first_file = None

        # Prefer first file whose header has no undefined or NAN_COLUMN_NAME
        for f in sorted_files:
            try:
                header = pd.read_csv(f, nrows=0).columns.tolist()
                if not any(h in ("undefined", NAN_COLUMN_NAME) for h in header):
                    valid_first_file = f
                    break
            except Exception:
                continue

        if valid_first_file:
            sorted_files.remove(valid_first_file)
            sorted_files.insert(0, valid_first_file)

        out_path = os.path.join(dst_dir, f"{var_name}.csv")

        with open(out_path, "w", newline="", encoding="utf-8") as out:
            writer = None
            reference_header = None

            for idx, f in enumerate(sorted_files):
                with open(f, "r", newline="", encoding="utf-8") as infile:
                    reader = csv.DictReader(infile)

                    if idx == 0:
                        reference_header = reader.fieldnames
                        writer = csv.DictWriter(out, fieldnames=reference_header)
                        writer.writeheader()

                    for row in reader:
                        aligned_row = {col: row.get(col, "") for col in reference_header}
                        writer.writerow(aligned_row)

        print(f"Saved: {out_path}")

        if logger.get_logging_level() != "DEBUG":
            for f in files:
                #os.remove(f)
                print(f"Deleted: {f}")
        else:
            print("Logging level is DEBUG; keeping partial files")


if __name__ == "__main__":
    args = sys.argv

    if len(args) >= 2 and args[1] == "--merge-results":
        if len(args) == 3:
            # User provided results_dir
            dst_dir = args[2]
        else:
            # No results_dir given, use last directory in ../../results
            results_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "results")
            )
            subdirs = [
                os.path.join(results_root, d)
                for d in os.listdir(results_root)
                if os.path.isdir(os.path.join(results_root, d))
            ]
            if not subdirs:
                print("No results directories found.")
                sys.exit(1)
            dst_dir = max(subdirs, key=os.path.getmtime)  # most recent dir
        print("Using destination_dir: ", dst_dir)
        join_and_cleanup_csvs(dst_dir=dst_dir)

    else:
        print("Usage: python file_io.py --merge-results [results_dir]")
