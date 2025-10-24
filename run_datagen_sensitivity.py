import os
import sys
import logging

logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use("Agg")
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
    feasible_rate = setup["feasible_rate"]
    entropy_threshold = setup["entropy_threshold"]
    delta_entropy_threshold = setup["delta_entropy_threshold"]
    chunk_length = setup["chunk_length"]
    dst_dir = setup.get("dst_dir") or None

    dimensions = [
        Dimension(label="tau_Dim_0", n_cases=n_cases, divs=1, borders=(-1, 1)),
        Dimension(label="tau_Dim_1", n_cases=n_cases, divs=1, borders=(-1, 1)),
        Dimension(label="tau_Dim_2", n_cases=n_cases, divs=1, borders=(-1, 1)),
        Dimension(label="tau_Dim_3", n_cases=n_cases, divs=1, borders=(-1, 1)),
        Dimension(label="tau_Dim_4", n_cases=n_cases, divs=1, borders=(-1, 1)),
        Dimension(label="tau_Dim_5", n_cases=n_cases, divs=1, borders=(-1, 1))
    ]

    stability_array = []
    output_dataframes_array = []


    execution_logs, dst_dir = \
        start(dimensions, n_samples, rel_tolerance,
              func=test_sensitivity_obj_func,
              max_depth=max_depth, dst_dir=dst_dir,
              use_sensitivity=use_sensitivity,
              sensitivity_divs=divs_per_cell, plot_boxplot=False, seed=seed,
              logging_level=logging_level, feasible_rate=feasible_rate,
              entropy_threshold=entropy_threshold, chunk_length=chunk_length,
              delta_entropy_threshold=delta_entropy_threshold,
              yaml_path=setup_path
              )

    stability_array = compss_wait_on(stability_array)
    output_dataframes_array = compss_wait_on(output_dataframes_array)
    return dst_dir


if __name__ == "__main__":
    args = sys.argv
    if len(args) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        main()
