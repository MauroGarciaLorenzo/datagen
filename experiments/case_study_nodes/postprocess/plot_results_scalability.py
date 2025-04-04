import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use("science")


def load_data(csv_path):
    columns = ["JobID", "CaseStudy", "Time", "Nodes", "CPUs", "Status",
               "Extra", "NNodes"]
    df = pd.read_csv(csv_path, sep="|", names=columns)
    df = df[df["Status"] == "COMPLETED"]

    # Convert "Time" column to numeric and filter invalid values
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df = df[df["Time"] >= 0]

    # Ensure "Nodes" is numeric
    df["Nodes"] = pd.to_numeric(df["Nodes"], errors="coerce")
    df = df.dropna(subset=["Nodes", "Time"])
    df["Nodes"] = df["Nodes"].astype(int)
    return df


def compute_metrics(df):
    # Group by Nodes and compute mean, std, and count for execution time
    grouped = df.groupby("Nodes")["Time"].agg(["mean", "std", "count"])
    grouped = grouped.sort_index()

    if grouped.empty:
        return grouped

    # Calculate 95% Confidence Intervals for execution time
    grouped["CI_Time"] = 1.96 * (grouped["std"] / np.sqrt(grouped["count"]))

    # Calculate speedup and efficiency for each individual run
    base_time = df[df["Nodes"] == 1]["Time"].mean()  # Base time (1 node)
    df["Speedup"] = base_time / df["Time"]
    df["Efficiency"] = df["Speedup"] / df["Nodes"]

    # Group by Nodes and compute mean, std, and count for speedup and efficiency
    speedup_grouped = df.groupby("Nodes")["Speedup"].agg(["mean", "std", "count"])
    efficiency_grouped = df.groupby("Nodes")["Efficiency"].agg(["mean", "std", "count"])

    # Calculate 95% Confidence Intervals for speedup and efficiency
    speedup_grouped["CI_Speedup"] = 1.96 * (speedup_grouped["std"] / np.sqrt(speedup_grouped["count"]))
    efficiency_grouped["CI_Efficiency"] = 1.96 * (efficiency_grouped["std"] / np.sqrt(efficiency_grouped["count"]))

    # Merge the results into a single DataFrame
    grouped["Speedup"] = speedup_grouped["mean"]
    grouped["Efficiency"] = efficiency_grouped["mean"]
    grouped["CI_Speedup"] = speedup_grouped["CI_Speedup"]
    grouped["CI_Efficiency"] = efficiency_grouped["CI_Efficiency"]

    return grouped


def plot_scaling(grouped, output_dir):
    nodes = grouped.index
    speedup = grouped["Speedup"]
    efficiency = grouped["Efficiency"]
    times = grouped["mean"]
    errors_time = grouped["CI_Time"]
    errors_speedup = grouped["CI_Speedup"]
    errors_efficiency = grouped["CI_Efficiency"]

    os.makedirs(output_dir, exist_ok=True)

    # Strong Scaling Plot
    plt.figure(figsize=(6.4, 4.8))
    plt.errorbar(nodes, speedup, yerr=errors_speedup, fmt='o-', markersize=2, label='Measured Speedup', capsize=5)
    plt.plot(nodes, nodes, '--', label='Ideal Speedup')
    plt.xlabel("Number of Nodes")
    plt.ylabel("Speedup")
    plt.legend(
        frameon=True,
        framealpha=0.7,
        facecolor='white'
    )
    plt.grid()
    plt.savefig(os.path.join(output_dir, "strong_scaling.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "strong_scaling.pdf"), dpi=300)

    # Execution Time Plot (Converted to Hours)
    plt.figure(figsize=(6.4, 4.8))
    times_hours = times / 3600  # Convert seconds to hours
    errors_hours = errors_time / 3600
    plt.errorbar(nodes, times_hours, yerr=errors_hours, fmt='o-', markersize=2, label='Execution Time', capsize=5)
    plt.xlabel("Number of Nodes")
    plt.ylabel("Execution Time (hours)")
    plt.legend(
        frameon=True,
        framealpha=0.7,
        facecolor='white'
    )
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "execution_time.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "execution_time.pdf"), dpi=300)

    # Efficiency Plot
    plt.figure(figsize=(6.4, 4.8))
    plt.errorbar(nodes, efficiency, yerr=errors_efficiency, fmt='o-', markersize=2, label='Measured Efficiency', capsize=5)
    plt.axhline(y=1, linestyle='--', color='r', label='Ideal Efficiency')
    plt.xlabel("Number of Nodes")
    plt.ylabel("Efficiency")
    plt.legend(
        frameon=True,
        framealpha=0.7,
        facecolor='white'
    )
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "efficiency.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "efficiency.pdf"), dpi=300)

    # Save versions with logarithmic x-axis
    for filename in ["strong_scaling", "execution_time", "efficiency"]:
        plt.figure(figsize=(6.4, 4.8))
        if filename == "execution_time":
            plt.errorbar(nodes, times_hours, yerr=errors_hours, fmt='o-', markersize=2, label='Execution Time', capsize=5)
            plt.ylabel("Execution Time (hours)")
        elif filename == "strong_scaling":
            plt.errorbar(nodes, speedup, yerr=errors_speedup, fmt='o-', markersize=2, label='Measured Speedup', capsize=5)
            plt.plot(nodes, nodes, '--', label='Ideal Speedup')
            plt.ylabel("Speedup")
        else:
            plt.errorbar(nodes, efficiency, yerr=errors_efficiency, fmt='o-', markersize=2, label='Measured Efficiency', capsize=5)
            plt.axhline(y=1, linestyle='--', color='r', label='Ideal Efficiency')
            plt.ylabel("Efficiency")

        plt.xlabel("Number of Nodes")
        plt.xscale("log")  # Logarithmic scale
        plt.legend(
            frameon=True,
            framealpha=0.7,
            facecolor='white'
        )
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{filename}_log.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, f"{filename}_log.pdf"), dpi=300)


def main(csv_path):
    df = load_data(csv_path)
    grouped = compute_metrics(df)
    print(grouped)
    if grouped.empty:
        print("No valid data to plot.")
        return
    dirname = os.path.dirname(__file__)
    output_dir = f"{dirname}/../results"
    plot_scaling(grouped, output_dir)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_csv>")
        sys.exit(1)
    main(sys.argv[1])