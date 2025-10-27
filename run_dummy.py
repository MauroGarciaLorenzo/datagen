import logging
import sys

from datagen.src.objective_function import dummy
from datagen.src.dimensions import Dimension
from datagen.src.parsing import parse_setup_file
from datagen.src.start_app import start

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
    variables_d0 = [(0, 2), (0, 1.5), (0, 1.5)]
    variables_d1 = [(0, 1), (0, 1.5), (0, 1.5), (0, 2)]
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
    dst_dir = setup.get("dst_dir", None)
    use_sensitivity = setup.get("use_sensitivity", None)
    sensitivity_divs = setup.get("sensitivity_divs")

    logging_level = logging.INFO
    #fig, ax = plt.subplots()
    dimensions = [
        Dimension(variable_borders=variables_d0, n_cases=n_cases, divs=2,
                  borders=(0, 5), label="Dim_0"),
        Dimension(variable_borders=variables_d1, n_cases=n_cases, divs=1,
                  borders=(1, 6), label="Dim_1")
    ]


    execution_logs, dst_dir = \
        start(dimensions, n_samples, rel_tolerance, func=dummy,
              max_depth=max_depth, use_sensitivity=use_sensitivity, ax=None,
              sensitivity_divs=sensitivity_divs, plot_boxplot=False, seed=seed,
              dst_dir=dst_dir, logging_level=logging_level,
              feasible_rate=feasible_rate, chunk_length=chunk_length,
              entropy_threshold=entropy_threshold,
              delta_entropy_threshold=delta_entropy_threshold,
              yaml_path=setup_path
              )
    return dst_dir

if __name__ == '__main__':
    args = sys.argv
    if len(args) < 2:
        yaml_path = "setup/default_setup.yaml"
    else:
        yaml_path = args[1]
    main(yaml_path)
