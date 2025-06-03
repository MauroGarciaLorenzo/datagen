import os
import random
import logging
import sys
from datetime import datetime

from matplotlib import pyplot as plt

from datagen.src.objective_function import dummy
from datagen.src.dimensions import Dimension
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

    variables_d0 = [(0, 2), (2, 2.5), (3, 3.5)]
    variables_d1 = [(10,20), (0, 1.5), (3, 4), (4, 5)]
    args_dict = parse_yaml_args(setup_path)
    n_samples = args_dict["n_samples"]
    n_cases = args_dict["n_cases"]
    rel_tolerance = args_dict["rel_tolerance"]
    max_depth = args_dict["max_depth"]
    seed = args_dict["seed"]
    print(n_samples, n_cases, rel_tolerance, max_depth)
    use_sensitivity = True
    logging_level = logging.DEBUG
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rnd_num = random.randint(1000, 9999)
    fig, ax = plt.subplots()
    dimensions = [
        Dimension(variable_borders=variables_d0, n_cases=n_cases, divs=2,
                  borders=(5, 8), label="Dim_0"),
        Dimension(variable_borders=variables_d1, n_cases=n_cases, divs=1,
                  borders=(17, 30.5), label="Dim_1")
    ]

    dir_name = f"dummy_seed{seed}_nc{n_cases}" \
               f"_ns{n_samples}_d{max_depth}_{timestamp}_{rnd_num}"
    path_results = os.path.join("results", dir_name)
    cases_df, dims_df, execution_logs, output_dataframes = \
        start(dimensions, n_samples, rel_tolerance, func=dummy,
              max_depth=max_depth, use_sensitivity=use_sensitivity, ax=ax,
              divs_per_cell=2, plot_boxplot=True, seed=seed,
              dst_dir=path_results, logging_level=logging_level)


def parse_yaml_args(setup_path):
    """Parses arguments from a YAML file and returns them as a dictionary."""
    if not os.path.isabs(setup_path):
        os.path.join(os.path.dirname(__file__), setup_path)
    try:
        with open(setup_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to parse YAML file: {e}")

    required_keys = ['n_samples', 'n_cases', 'rel_tolerance', 'max_depth',
                     'seed']
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required keys in YAML: {missing_keys}")

    return {
        'n_samples': int(config['n_samples']),
        'n_cases': int(config['n_cases']),
        'rel_tolerance': float(config['rel_tolerance']),
        'max_depth': int(config['max_depth']),
        'seed': int(config['seed']),
    }


if __name__ == '__main__':
    args = sys.argv
    if len(args) < 2:
        yaml_path = "setup/default_setup.yaml"
    else:
        yaml_path = args[1]
    main(yaml_path)
