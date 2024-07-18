"""
Utility functions to postprocess results.
"""
import os
from sys import argv

import pandas as pd
import matplotlib.pyplot as plt

import scienceplots

plt.style.use('science')


def cu_performance(src_dir, case_name, dst_dir):
    """
    Read the results of the execution for different computing units
    and plot.
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


if __name__ == "__main__":
    # Parse arguments for src_dir and case_name
    if len(argv) == 4:
        src_dir = argv[1]
        case_name = argv[2]
        dst_dir = argv[3]
    else:
        print("Using default values for src_dir and case_name")
        src_dir = "../../performance_tests/results"
        case_name = "case_0_computing_times_seed17"
        dst_dir = "../../figures"
        print(f"src_dir: {src_dir}")
        print(f"case_name: {case_name}")
    cu_performance(src_dir, case_name, dst_dir)
