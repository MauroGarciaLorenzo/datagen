import os
import random
from datetime import datetime
from .utils import clean_dir
import pandas as pd
import logging
import csv
logger = logging.getLogger(__name__)

from datagen.src.data_ops import sort_df_rows_by_another, sort_df_last_columns


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


def save_results(cases_df, dims_df, execution_logs, output_dataframes,
                 dst_dir, execution_time):
    """
    Save set of results including the main cases/dims dataframes plus
    execution logs and all elements inside the output_dataframes dictionary,
    mostly dataframes. These dataframe rows must follow the same sorting
    according to the case ids.
    """
    if dst_dir is None:
        dst_dir = "results"

    cases_df.to_csv(os.path.join(dst_dir, "cases_df.csv"))
    dims_df.to_csv(os.path.join(dst_dir, "dims_df.csv"))
    execution_time_df = pd.DataFrame(
        {"execution_time": [execution_time]})

    execution_time_df.to_csv((os.path.join(dst_dir, "execution_time.csv")), index=False)

    for key, value in output_dataframes.items():
        if isinstance(value, pd.DataFrame):
            # All dataframes should have the same sorting
            sorted_df = sort_df_rows_by_another(cases_df, value, "case_id")
            # Sort columns at the end
            sorted_df = sort_df_last_columns(sorted_df)
            # Save dataframe
            sorted_df.to_csv(os.path.join(dst_dir, f"case_{key}.csv"))
        else:
            for k, v in value.items():
                if isinstance(v, pd.DataFrame):
                    value.to_csv(os.path.join(dst_dir, f"case_{k}.csv"))
                else:
                    logger.warning(f"Invalid nested format for output '{k}'")

    if execution_logs!=None:
        with open(os.path.join(dst_dir, "execution_logs.txt"), "w") as log_file:
            for log_entry in execution_logs:
                log_file.write("Dimensions:\n")
                for dim in log_entry[0]:
                    log_file.write(f"{dim}\n")
                log_file.write(f"Entropy: {log_entry[1]}\n")
                log_file.write(f"Delta Entropy: {log_entry[2]}\n")
                log_file.write(f"Depth: {log_entry[3]}\n")
                log_file.write("\n")

def init_dst_dir(calling_module, seed, n_cases, n_samples, max_depth,
                 working_dir, ax, dimensions):
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


def log_cell_info(cell_name, depth, delta_entropy, feasible_ratio, status, dst_dir):
    csv_path = os.path.join(dst_dir, "cell_info.csv")
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Cell Name", "Depth", "Delta Entropy", "Feasible Ratio", "Status"])
        writer.writerow([cell_name, depth, delta_entropy, feasible_ratio, status])