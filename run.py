
import numpy as np

from datagen.src.dimensions import Dimension
from datagen.src.start_app import start
from datagen.src.objective_function import dummy

try:
    from pycompss.api.task import task
    from pycompss.api.api import compss_wait_on
except ImportError:
    from datagen.dummies.task import task
    from datagen.dummies.api import compss_wait_on


@task()
def main():
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

        Apart from that, it can also be specified the number of samples and
        tolerance (maximum difference upper-lower bound of a dimension within a
        cell to be subdivided).
        """
    p_sg = np.array([(0, 2), (0, 1.5), (0, 1.5)])
    p_cig = np.array([(0, 1), (0, 1.5), (0, 1.5), (0, 2)])
    tau_f_g_for = np.array([(0, 2)])
    tau_v_g_for = np.array([(0, 2)])
    tau_p_g_for = np.array([(0, 2)])
    tau_q_g_for = np.array([(0, 2)])
    n_samples = 2
    n_cases = 3
    rel_tolerance = 0.1
    # max_depth = 5
    # ax = plt.figure().add_subplot(projection='3d')
    dimensions = [
        Dimension(variables=p_sg, n_cases=n_cases, divs=2, borders=(0, 5),
                  label="p_sg"),
        Dimension(variables=p_cig, n_cases=n_cases, divs=2, borders=(0, 6),
                  label="p_cig"),
        Dimension(variables=tau_f_g_for, n_cases=n_cases, divs=1,
                  borders=(0, 2), label="tau_f_g_for"),
        Dimension(variables=tau_v_g_for, n_cases=n_cases, divs=1,
                  borders=(0, 2), label="tau_v_g_for"),
        Dimension(variables=tau_p_g_for, n_cases=n_cases, divs=1,
                  borders=(0, 2), label="tau_p_g_for"),
        Dimension(variables=tau_q_g_for, n_cases=n_cases, divs=1,
                  borders=(0, 2), label="tau_q_g_for")
    ]

    ax = None
    start(dimensions, n_samples, rel_tolerance, ax, dummy)


if __name__ == "__main__":
    main()
