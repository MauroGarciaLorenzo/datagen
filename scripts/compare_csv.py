import pandas as pd
import matplotlib.pyplot as plt
import sys


def plot_csv_columns(csv_paths):
    # Load the CSV files
    dataframes = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            sys.exit(1)

    # Ensure all CSVs have the same columns
    base_columns = set(dataframes[0].columns)
    for i, df in enumerate(dataframes[1:], start=2):
        if set(df.columns) != base_columns:
            print(f"Error: CSV {i} has different columns.")
            sys.exit(1)

    # Prepare columns to plot
    columns = list(base_columns)
    cols_to_remove = ["case_id"] + [col for col in columns if col.startswith("tau")]
    columns = [col for col in columns if col not in cols_to_remove]

    num_cols = len(columns)
    if num_cols % 2 != 0:
        print("Warning: Number of columns must be even to form pairs.")

    colors = ['blue', 'red', 'green', 'orange', 'purple']
    labels = [f'CSV {i+1}' for i in range(len(dataframes))]

    # Plot
    for i in range(1, num_cols, 2):
        col_x = columns[i]
        col_y = columns[i + 1]

        plt.figure(figsize=(8, 6))
        for df, color, label in zip(dataframes, colors, labels):
            print(label, df[col_x], df[col_y])
            plt.scatter(df[col_x], df[col_y], label=label, alpha=0.6, color=color)

        plt.xlabel(col_x)
        plt.ylabel(col_y)
        plt.title(f"Scatter Plot: {col_x} vs {col_y}")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <csv1> ... <csvn>")
        sys.exit(1)

    csv_paths = sys.argv[1:]
    plot_csv_columns(csv_paths)
