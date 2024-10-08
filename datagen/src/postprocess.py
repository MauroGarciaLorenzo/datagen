"""
Utility functions to postprocess results.
"""
import os
import re

from sys import argv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scienceplots

plt.style.use('science')


def cu_perf_standalone(src_dir, case_name, dst_dir):
    """
    Read the results of the execution of the objective function for different
    computing units and plot.
    """
    total_times = {}
    for filename in os.listdir(src_dir):
        if not filename.startswith(f"cu") or case_name not in filename:
            continue
        cu = int(filename.split("_")[0][2:])
        df = pd.read_csv(os.path.join(src_dir, filename), index_col=0)
        total_times[cu] = df.sum().sum()
    # Plot
    total_times = dict(sorted(total_times.items()))
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(total_times.keys(), total_times.values(), marker="o")
    plt.xscale("log")
    plt.xlabel("Computing Units")
    plt.ylabel("Total time (s)")
    plt.title("Total execution time of the feasibility objective function")
    plt.savefig(os.path.join(dst_dir, f"{case_name}_cu_performance.pdf"))
    plt.savefig(os.path.join(dst_dir, f"{case_name}_cu_performance.png"),
                dpi=300)
    plt.show()


def cu_perf_datagen(analysis_name, results_dir=None, compss_dir=None,
                    figures_dir=None):
    """
    Read the results of the execution of the whole datagen application for
    different computing units and plot the overall time comparison.
    """
    # Paths to directories
    if results_dir is None:
        results_dir = f'../../results/{analysis_name}'
    if compss_dir is None:
        compss_dir = f'../../logs/{analysis_name}'
    if figures_dir is None:
        figures_dir = f'../../figures/{analysis_name}'

    # Prepare lists for job_id, total_time, and n_cpus
    job_ids = []
    total_times = []
    n_cases = []
    n_cpus_list = []
    times_per_case = []

    # Loop through subdirectories in the results directory
    for subdir in os.listdir(results_dir):
        subdir_path = os.path.join(results_dir, subdir)

        if os.path.isdir(subdir_path):
            # Extract job_id from the directory name
            match = re.search(r'slurm(\d+)_', subdir)
            if match:
                job_id = match.group(1)
                job_ids.append(job_id)

                # Read the case_df_computing_times.csv file
                csv_file = os.path.join(subdir_path,
                                        'case_df_computing_times.csv')
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file)

                    # Sum all elements ignoring the first column (index)
                    total_time = df.iloc[:, 1:].sum().sum()
                    n_case = len(df)
                    time_per_case = total_time / n_case

                    total_times.append(total_time)
                    n_cases.append(n_case)
                    times_per_case.append(time_per_case)

                    # Look for the job_id folder in .COMPSs and search n_cpus
                    job_dir = os.path.join(compss_dir, job_id)
                    for root, dirs, files in os.walk(job_dir):
                        for file in files:
                            if file == 'job1_NEW.out':
                                out_file = os.path.join(root, file)
                                with open(out_file, 'r') as f:
                                    for line in f:
                                        # Find the line with "COMPUTING_UNITS:"
                                        if "COMPUTING_UNITS:" in line:
                                            n_cpus_match = re.search(
                                                r'COMPUTING_UNITS:\s+(\d+)',
                                                line)
                                            if n_cpus_match:
                                                n_cpus = int(
                                                    n_cpus_match.group(1))
                                                n_cpus_list.append(n_cpus)
                                                break
                        else:
                            continue
                        break

    # Sort lists based on n_cpus (1, 2, 4, 8, ...)
    sorted_lists = sorted(zip(
        n_cpus_list, total_times, n_cases, times_per_case, job_ids))
    n_cpus_list_sorted, total_times_sorted, n_cases_sorted, \
        times_per_case_sorted, job_ids_sorted = zip(
        *sorted_lists)

    # Convert back to lists (if needed)
    n_cpus_list = list(n_cpus_list_sorted)
    total_times = list(total_times_sorted)
    n_cases = list(n_cases_sorted)
    times_per_case = list(times_per_case_sorted)
    job_ids = list(job_ids_sorted)

    # Other metrics
    parallel_efficiency = times_per_case[0] / (
            np.array(times_per_case) * np.array(n_cpus_list)) * 100

    # Ensure results directory for the figure exists
    os.makedirs(figures_dir, exist_ok=True)

    # Create plot of n_cpus vs total_time/n_cases
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(n_cpus_list, times_per_case, 'bo-')
    ax1.set_xlabel('Number of CPUs')
    ax1.set_ylabel('Time per Case (s)', color='b')
    ax1.set_title(
        'Time per Case vs Number of CPUs assigned to Obj. Func.')
    ax2 = ax1.twinx()
    ax2.plot(n_cpus_list, n_cases, 'gx--')
    ax2.set_ylabel('Number of cases', color='g')
    # Save the figure
    output_figure = os.path.join(figures_dir,
                                 'ncpus_vs_time_per_case')
    fig.savefig(f"{output_figure}.png", dpi=300)
    fig.savefig(f"{output_figure}.pdf")

    # Create plot for parallel efficiency
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(n_cpus_list, parallel_efficiency, 'bo-')
    ax1.set_xlabel('Number of CPUs')
    ax1.set_ylabel('Parallel Efficiency (\%)')
    ax1.set_title('Parallel Efficiency')
    output_figure = os.path.join(figures_dir,
                                 'ncpus_vs_parallel_efficiency')
    fig.savefig(f"{output_figure}.png", dpi=300)
    fig.savefig(f"{output_figure}.pdf")

    # Save dataframe with results
    df = pd.DataFrame({
        'job ID': job_ids,
        'n_cpus': n_cpus_list,
        'n_cases': n_cases,
        'times per case': times_per_case,
        'parallel efficiency': parallel_efficiency
    })
    output_file = os.path.join(figures_dir, 'raw_data.csv')
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    name = 'cu_perf_datagen_1node'
    cu_perf_datagen(name)

    # # Parse arguments for src_dir and case_name
    # if len(argv) == 4:
    #     src_dir = argv[1]
    #     case_name = argv[2]
    #     dst_dir = argv[3]
    # else:
    #     print("Using default values for src_dir and case_name")
    #     src_dir = "../../performance_tests/results"
    #     case_name = "case_0_computing_times_seed17"
    #     dst_dir = "../../figures"
    #     print(f"src_dir: {src_dir}")
    #     print(f"case_name: {case_name}")
    # cu_perf_standalone(src_dir, case_name, dst_dir)
