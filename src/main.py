from classes import Dimension
from utils import flatten_list
from sampling import explore_cell, gen_grid
from objective_function import dummy
from pycompss.api.task import task
from pycompss.api.api import compss_wait_on
from viz import print_results
import pandas as pd


@task(returns=1)
def main(dimensions, n_samples, tolerance, ax):
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
