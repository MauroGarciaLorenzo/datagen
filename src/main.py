from classes import Dimension
from utils import flatten_list
from sampling import explore_cell, gen_grid
from objective_function import dummy
from pycompss.api.task import task
from pycompss.api.api import compss_wait_on
import pandas as pd

from viz import print_results


@task(returns=1)
def main():
    variables_d1 = [(0, 2), (0, 1.5), (0, 1.5)]
    variables_d2 = [(0, 1), (0, 1.5), (0, 1.5), (1, 2)]
    variables_d3 = [(1, 3.5), (1, 3.5)]
    dim_min = [0, 1, 2]
    dim_max = [5, 6, 7]
    n_samples = 3
    n_cases = 2
    tolerance = 0.1
    max_depth = 5
    divs = [2, 1, 1]
    # ax = plt.figure().add_subplot(projection='3d')
    ax = None
    dimensions = [Dimension(variables_d1, n_cases, divs[0], dim_min[0], dim_max[0]),
                  Dimension(variables_d2, n_cases, divs[1], dim_min[1], dim_max[1]),
                  Dimension(variables_d3, n_cases, divs[2], dim_min[2], dim_max[2])]

    grid = gen_grid(dimensions)
    execution_logs = [None] * len(grid)
    depth = 0
    cases = None
    entropy = None
    list_cases_df = []
    for cell in range(len(grid)):
        dims = grid[cell].dimensions
        execution_logs[cell], cases_df = explore_cell(
            dummy, n_samples, entropy, tolerance, depth, cell, ax, dims, cases
        )  # for each cell in grid, explore_cell
        list_cases_df.append(cases_df)

    # implement reduce
    for cell in range(len(grid)):
        execution_logs[cell] = compss_wait_on(execution_logs[cell])
        list_cases_df[cell] = compss_wait_on(list_cases_df[cell])
    cases_df = pd.concat(list_cases_df, ignore_index=True)

    execution_logs = flatten_list(execution_logs)
    print_results(execution_logs, cases_df)


if __name__ == "__main__":
    main()
