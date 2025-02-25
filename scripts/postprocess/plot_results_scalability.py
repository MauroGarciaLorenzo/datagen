import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(csv_path):
    columns = ["JobID", "CaseStudy", "Time", "Nodes", "CPUs", "Status",
               "Extra"]
    df = pd.read_csv(csv_path, sep="|", names=columns)
    df = df[df["Status"] == "COMPLETED"]
    df["Time"] = pd.to_timedelta(
        df["Time"]).dt.total_seconds()
    return df


def compute_metrics(df):
    grouped = df.groupby("Nodes")["Time"].agg(["mean", "std"])
    grouped = grouped.sort_index()
    base_time = grouped.iloc[0]["mean"]  # Tiempo en 1 nodo
    grouped["Speedup"] = base_time / grouped["mean"]
    grouped["Efficiency"] = grouped["Speedup"] / grouped.index
    return grouped


def plot_scaling(grouped, output_dir):
    nodes = grouped.index
    speedup = grouped["Speedup"]
    efficiency = grouped["Efficiency"]
    times = grouped["mean"]
    errors = grouped["std"]

    os.makedirs(output_dir, exist_ok=True)

    # Strong Scaling Plot
    plt.figure()
    plt.errorbar(nodes, speedup, yerr=errors / times, fmt='o-',
                 label='Measured Speedup')
    plt.plot(nodes, nodes, '--', label='Ideal Speedup')
    plt.xlabel("Number of Nodes")
    plt.ylabel("Speedup")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "strong_scaling.png"), dpi=300)

    # Execution Time Plot
    plt.figure()
    plt.errorbar(nodes, times, yerr=errors, fmt='o-', label='Execution Time')
    plt.xlabel("Number of Nodes")
    plt.ylabel("Execution Time (s)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "execution_time.png"), dpi=300)

    # Efficiency Plot
    plt.figure()
    plt.errorbar(nodes, efficiency, yerr=errors / (nodes * times), fmt='o-',
                 label='Measured Efficiency')
    plt.axhline(y=1, linestyle='--', color='r', label='Ideal Efficiency')
    plt.xlabel("Number of Nodes")
    plt.ylabel("Efficiency")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "efficiency.png"), dpi=300)

    plt.show()


def main(csv_path):
    df = load_data(csv_path)
    grouped = compute_metrics(df)
    print("--------_---", os.getcwd())
    output_dir = "../../figures/experiment_1_scalability"
    plot_scaling(grouped, output_dir)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_csv>")
        sys.exit(1)
    main(sys.argv[1])
