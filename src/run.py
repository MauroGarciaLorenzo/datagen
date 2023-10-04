import sys
sys.path.append("../")
from pycompss.api.task import task
from src.classes import Dimension
from src.main import main


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

        Apart from that, it can also be specified the number of samplesand the
        tolerance (maximum difference upper-lower bound of a dimension within a
        cell to be subdivided).
        """
    p_sg = [(0, 2), (0, 1.5), (0, 1.5)]
    p_cig = [(0, 1), (0, 1.5), (0, 1.5), (1, 2)]
    gfor = [(0, 1), (0, 1.5), (0, 1.5), (1, 2)]
    tau_f_g_for = [(0, 2)]
    tau_v_g_for = [(0, 2)]
    tau_p_g_for = [(0, 2)]
    tau_q_g_for = [(0, 2)]
    dim_min = [0, 1, 3]
    dim_max = [5, 6, 7]
    n_samples = 5
    n_cases = 3
    tolerance = 0.1
    # max_depth = 5
    divs = [2, 1, 1]
    # ax = plt.figure().add_subplot(projection='3d')
    dimensions = [Dimension(p_sg, n_cases, divs[0], dim_min[0],
                            dim_max[0], "p_sg"),
                  Dimension(p_cig, n_cases, divs[1], dim_min[1],
                            dim_max[1], "p_cig"),
                  Dimension(gfor, n_cases, divs[2], dim_min[1],
                            dim_max[1], "g_for"),
                  Dimension(tau_f_g_for, n_cases, divs[1], dim_min[2],
                            dim_max[2], "tau_f_g_for"),
                  Dimension(tau_v_g_for, n_cases, divs[1], dim_min[2],
                            dim_max[2], "tau_v_g_for"),
                  Dimension(tau_p_g_for, n_cases, divs[1], dim_min[2],
                            dim_max[2], "tau_p_g_for"),
                  Dimension(tau_q_g_for, n_cases, divs[1], dim_min[2],
                            dim_max[2], "tau_q_g_for")]
    ax = None
    main(dimensions, n_samples, tolerance, ax)


run()
