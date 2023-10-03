import sys
sys.path.append('../')

from src.classes import Dimension
from src.main import main


def run():
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
