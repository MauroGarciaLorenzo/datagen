
import numpy as np

from datagen.src.objective_function import dummy
from datagen.src.classes import Dimension
from datagen.src.main import main

try:
    from pycompss.api.task import task
    from pycompss.api.api import compss_wait_on
except ImportError:
    from datagen.dummies import task
    from datagen.dummies import compss_wait_on


@task(returns=1)
def run():
    """
    In this method we work with dimensions (main axes), which represent a
    list of variables. For example, the value of each variable of a concrete
    dimension could represent the power supplied by a generator, while the
    value linked to that dimension should be the total sum of energy produced.

    For each dimension it must be declared:
        -variables: list of variables represented by tuples containing its
                lower and upper borders.
        -n_cases: number of cases taken for each sample (each sample represents
                the total sum of a dimension). A case is a combination of
                variables where all summed together equals the sample.
        -divs: number of divisions in that dimension. It will be the growth
                order of the number of cells
        -lower: lower bound of the dimension (minimum value of a sample)
        -upper: upper bound of the dimension (maximum value of a sample)
        -label: dimension identifier

    Apart from that, it can also be specified the number of samples and the
    tolerance (maximum difference upper-lower bound of a dimension within a
    cell to be subdivided).
    """
    variables_d1 = np.array([(0, 2), (0, 1.5), (0, 1.5)])
    variables_d2 = np.array([(0, 1), (0, 1.5), (0, 1.5), (1, 2)])
    variables_d3 = np.array([(1, 3.5), (1, 3.5)])
    n_samples = 2
    n_cases = 2
    tolerance = 0.5
    # max_depth = 5
    # ax = plt.figure().add_subplot(projection='3d')
    ax = None
    dimensions = [Dimension(
                    variables=variables_d1,
                    n_cases=n_cases,
                    divs=2,
                    lower=0,
                    upper=5,
                    label="0"),
                  Dimension(
                      variables=variables_d2,
                      n_cases=n_cases,
                      divs=1,
                      lower=1,
                      upper=6,
                      label="1"),
                  Dimension(
                      variables=variables_d3,
                      n_cases=n_cases,
                      divs=1,
                      lower=2,
                      upper=7,
                      label="2")]
    main(dimensions, n_samples, tolerance, ax, dummy)


if __name__ == '__main__':
    run()
