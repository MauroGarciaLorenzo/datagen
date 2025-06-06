import os
import sys
import logging

from datagen import print_dict_as_yaml

logger = logging.getLogger(__name__)

from matplotlib import pyplot as plt
from datagen.src.parsing import parse_setup_file
from datagen.src.dimensions import Dimension
from datagen.src.start_app import start
from datagen.src.objective_function import test_sensitivity_obj_func

try:
    from pycompss.api.task import task
    from pycompss.api.api import compss_wait_on
except ImportError:
    from datagen.dummies.task import task
    from datagen.dummies.api import compss_wait_on

import warnings
warnings.filterwarnings("ignore")

@task(on_failure='FAIL')
def main(working_dir=None, setup_path="setup/default_setup.yaml"):
    use_sensitivity = True
    divs_per_cell = 4
    feasible_rate = 0.5
    logging_level = logging.DEBUG
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    setup = parse_setup_file(setup_path)

    if working_dir is None:
        working_dir = os.path.join(os.path.dirname(__file__), "..", "..")

    n_samples = setup["n_samples"]
    n_cases = setup["n_cases"]
    rel_tolerance = setup["rel_tolerance"]
    max_depth = setup["max_depth"]
    seed = setup["seed"]

    # Print case configuration
    print(f"\n{''.join(['='] * 30)}\n"
          f"Running application with the following parameters:"
          f"\n{''.join(['='] * 30)}")
    print_dict_as_yaml(setup)

    dimensions = [
        Dimension(label="tau_Dim_0", n_cases=n_cases, divs=1, borders=(-1, 1)),
        Dimension(label="tau_Dim_1", n_cases=n_cases, divs=1, borders=(-1, 1)),
        Dimension(label="tau_Dim_2", n_cases=n_cases, divs=1, borders=(-1, 1)),
        Dimension(label="tau_Dim_3", n_cases=n_cases, divs=1, borders=(-1, 1)),
        Dimension(label="tau_Dim_4", n_cases=n_cases, divs=1, borders=(-1, 1)),
        Dimension(label="tau_Dim_5", n_cases=n_cases, divs=1, borders=(-1, 1))
    ]

    dir_name = f"datagen_sensitivity_seed{seed}_nc{n_cases}" \
               f"_ns{n_samples}_d{max_depth}"
    path_results = os.path.join(
        working_dir, "results", dir_name)
    if not os.path.isdir(path_results):
        os.makedirs(path_results)

    stability_array = []
    output_dataframes_array = []
    cases_df, dims_df, execution_logs, output_dataframes = \
        start(dimensions, n_samples, rel_tolerance,
              func=test_sensitivity_obj_func,
              max_depth=max_depth, dst_dir=path_results,
              use_sensitivity=use_sensitivity,
              divs_per_cell=divs_per_cell, plot_boxplot=False, seed=seed,
              feasible_rate=feasible_rate, logging_level=logging_level)

    stability_array = compss_wait_on(stability_array)
    output_dataframes_array = compss_wait_on(output_dataframes_array)
    return os.path.abspath(path_results)


if __name__ == "__main__":
    args = sys.argv
    if len(args) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        main()
