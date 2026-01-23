import os
import sys
sys.path.append("..")

from datagen.src.start_app import start
from setup import setUp_basic
from pycompss.api.task import task


@task()
def main():
    (dimensions, n_samples, rel_tolerance, dummy, max_depth, use_sensitivity,
     ax, divs_per_cell, plot_boxplot) = setUp_basic()
    execution_logs, result_dir = start(dimensions, n_samples,
                                       rel_tolerance, dummy, max_depth,
                                       use_sensitivity=use_sensitivity,
                                       ax=ax, sensitivity_divs=2)


    for file in os.listdir(result_dir):
        file_path = os.path.join(result_dir, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting files within {file_path}: {e}")

    with open(os.path.join(result_dir, "execution_logs.txt"), "w") as log_file:
        for log_entry in execution_logs:
            log_file.write("Dimensions:\n")
            for dim in log_entry[0]:
                log_file.write(f"{dim}\n")
            log_file.write(f"Entropy: {log_entry[1]}\n")
            log_file.write(f"Delta Entropy: {log_entry[2]}\n")
            log_file.write(f"Depth: {log_entry[3]}\n")
            log_file.write("\n")


if __name__ == "__main__":
    main()
