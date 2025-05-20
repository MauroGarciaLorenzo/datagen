import pandas as pd
import matplotlib.pyplot as plt
import sys


def plot_csv_columns(csv_path1, csv_path2):
    # Load the CSV files
    try:
        df1 = pd.read_csv(csv_path1)
        df2 = pd.read_csv(csv_path2)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        sys.exit(1)

    # Ensure both CSVs have the same columns, regardless of order
    columns1 = set(df1.columns)
    columns2 = set(df2.columns)

    if columns1 != columns2:
        print(
            "Error: CSV files must have the same columns. The following differences were found:")
        diff1 = columns1 - columns2  # Columns in df1 but not in df2
        diff2 = columns2 - columns1  # Columns in df2 but not in df1

        if diff1:
            print(
                f"Columns in the first CSV but not in the second: {', '.join(diff1)}")
        if diff2:
            print(
                f"Columns in the second CSV but not in the first: {', '.join(diff2)}")
        sys.exit(1)

    # Ensure columns are in the same order for plotting
    columns1 = list(df1.columns)
    columns2 = list(df2.columns)

    if columns1 != columns2:
        print(
            "Warning: Column order differs between the CSVs. Columns will be matched by name.")

    # Column selection
    cols_to_remove = ["case_id"]

    for col in columns1:
        if col.startswith("tau"):
            cols_to_remove.append(col)

    for col in cols_to_remove:
        columns1.remove(col)
        columns2.remove(col)

    # Pair columns for plotting
    num_cols = len(columns1)

    if num_cols % 2 != 0:
        print("Warning: Number of columns must be even to form pairs.")

    # Create plots for each pair of columns
    for i in range(1, num_cols, 2):
        col_x = columns1[i]
        col_y = columns2[i + 1]

        plt.figure(figsize=(8, 6))
        print("DF1:")
        print(df1[col_x])
        print(df1[col_y])
        print("---------------------------------------------")
        print("DF2:")
        print(df2[col_x])
        print(df2[col_y])
        print("---------------------------------------------")

        plt.scatter(df1[col_x], df1[col_y], color='blue', label='CSV 1',
                    alpha=0.7)
        plt.scatter(df2[col_x], df2[col_y], color='red', label='CSV 2',
                    alpha=0.7)
        plt.xlabel(col_x)
        plt.ylabel(col_y)
        plt.title(f"Scatter Plot: {col_x} vs {col_y}")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <csv_path1> <csv_path2>")
        sys.exit(1)

    csv_path1 = sys.argv[1]
    csv_path2 = sys.argv[2]

    plot_csv_columns(csv_path1, csv_path2)
