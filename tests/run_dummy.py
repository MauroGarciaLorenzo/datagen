from src.classes import Dimension
from src.main import main


def run():
    variables_d1 = [(0, 2), (0, 1.5), (0, 1.5)]
    variables_d2 = [(0, 1), (0, 1.5), (0, 1.5), (1, 2)]
    variables_d3 = [(1, 3.5), (1, 3.5)]
    dim_min = [0, 1, 2]
    dim_max = [5, 6, 7]
    n_samples = 3
    n_cases = 2
    tolerance = 0.1
    # max_depth = 5
    divs = [2, 1, 1]
    # ax = plt.figure().add_subplot(projection='3d')
    ax = None
    dimensions = [Dimension(variables_d1, n_cases, divs[0], dim_min[0],
                            dim_max[0], "0"),
                  Dimension(variables_d2, n_cases, divs[1], dim_min[1],
                            dim_max[1], "1"),
                  Dimension(variables_d3, n_cases, divs[2], dim_min[2],
                            dim_max[2], "2")]
    main(dimensions, n_samples, tolerance, ax)
