import os
import random
import logging
import sys
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

from datagen.src.objective_function import dummy
from datagen.src.dimensions import Dimension
from datagen.src.parsing import parse_setup_file
from datagen.src.start_app import start
import yaml

try:
    from pycompss.api.task import task
    from pycompss.api.api import compss_wait_on
except ImportError:
    from datagen.dummies.task import task
    from datagen.dummies.api import compss_wait_on


@task(on_failure='FAIL')
def main(setup_path="setup/default_setup.yaml"):
    """
    In this method we work with dimensions (main axes), which represent a
    list of variable_borders. For example, the value of each variable of a concrete
    dimension could represent the power supplied by a generator, while the
    value linked to that dimension should be the total sum of energy produced.

    For each dimension it must be declared:
        -variable_borders: list of variable_borders represented by tuples containing its
                lower and upper borders.
        -n_cases: number of cases taken for each sample (each sample represents
                the total sum of a dimension). A case is a combination of
                variable_borders where all summed together equals the sample.
        -divs: number of divisions in that dimension. It will be the growth
                order of the number of cells
        -lower: lower bound of the dimension (minimum value of a sample)
        -upper: upper bound of the dimension (maximum value of a sample)
        -label: dimension identifier
        -independent_dimension: indicates whether the dimension is true (sampleable) or not.

    Apart from that, it can also be specified the number of samples and
    the relative tolerance (indicates the portion of the size of the original
    dimension). For example, if we have a dimension of size 10 and relative
    tolerance is 0.5, the smallest cell in this dimension will have size 5.
    Lastly, user should provide the objective function. As optional parameters,
    the user can define:
        -use_sensitivity: a boolean indicating whether sensitivity analysis is
        used or not.
        -ax: plot axes in case it is desired to show stability points and cells
        divisions (dimensions length must be 2). Plots saved in
        "datagen/results/figures".
        -plot_boxplot: a boolean indicating whether boxplots for each variable
        must be obtained or not. Plots saved in "datagen/results/figures".
    """
    setup = parse_setup_file(setup_path)
    n_samples = setup["n_samples"]
    n_cases = setup["n_cases"]
    rel_tolerance = setup["rel_tolerance"]
    max_depth = setup["max_depth"]
    seed = setup["seed"]
    feasible_rate = setup["feasible_rate"]
    entropy_threshold = setup["entropy_threshold"]
    delta_entropy_threshold = setup["delta_entropy_threshold"]
    chunk_length = setup["chunk_length"]
    computing_units = setup["environment"]["COMPUTING_UNITS"]

    # Set up seeded generator
    generator = np.random.default_rng(seed)

    # Helper function to generate valid (min, max) pairs from the seed received
    def generate_variable_borders(n_vars, low=0, high=10):
        borders = []
        for _ in range(n_vars):
            a = generator.uniform(low, high)
            b = generator.uniform(low, high)
            min_val, max_val = sorted([a, b])
            min_val = max(min_val, 0)
            max_val = max(max_val, 0)
            borders.append((min_val, max_val))
        return borders

    # Generate dimensions
    variables_d0 = generate_variable_borders(3, low=0, high=5)
    variables_d1 = generate_variable_borders(4, low=0, high=10)

    # Compute borders from generated variable ranges
    borders_d0 = (
        sum([v[0] for v in variables_d0]),
        sum([v[1] for v in variables_d0])
    )
    borders_d1 = (
        sum([v[0] for v in variables_d1]),
        sum([v[1] for v in variables_d1])
    )

    # Directory for output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rnd_num = random.randint(1000, 9999)
    dir_name = f"dummy_seed{seed}_nc{n_cases}_ns{n_samples}_d{max_depth}_{timestamp}_{rnd_num}"
    path_results = os.path.join("results", dir_name)
    
    # Dimensions
    dimensions = [
        Dimension(variable_borders=variables_d0, n_cases=n_cases, divs=2,
                  borders=borders_d0, label="Dim_0"),
        Dimension(variable_borders=variables_d1, n_cases=n_cases, divs=1,
                  borders=borders_d1, label="Dim_1")
    ]

    # Run experiment
    fig, ax = plt.subplots()
    use_sensitivity = True
    logging_level = logging.DEBUG

    execution_logs = \
        start(dimensions, n_samples, rel_tolerance, func=dummy,
              max_depth=max_depth, use_sensitivity=use_sensitivity, ax=ax,
              divs_per_cell=2, plot_boxplot=False, seed=seed,
              dst_dir=path_results, logging_level=logging_level,
              feasible_rate=feasible_rate, chunk_length=chunk_length,
              entropy_threshold=entropy_threshold,
              delta_entropy_threshold=delta_entropy_threshold,
              computing_units=computing_units
              )
    return path_results

if __name__ == '__main__':
    args = sys.argv
    if len(args) < 2:
        yaml_path = "setup/default_setup.yaml"
    else:
        yaml_path = args[1]
    main(yaml_path)
