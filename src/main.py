from sampling import explore_cell, gen_grid
from objective_function import dummy
from pycompss.api.task import task
from pycompss.api.api import compss_wait_on

from utils import flatten_list
from viz import print_results
import pandas as pd


@task(returns=1)
def main(dimensions, n_samples, tolerance, ax):
    """In this method we work with dimensions (main axes), which represent a
    list of variables. For example, the value of each variable of a concrete
    dimension could represent the power supplied by a generator, while the
    value linked to that dimension should be the total sum of energy produced.

    The function explores each initial cells and obtains two objects from it:
        -execution_logs: dimensions, entropy, delta entropy and depth of each
                        cell.
        -cases_df: dataframe containing each case and associated stability
                taken during the execution.

    :param dimensions: List of dimensions involved
    :param n_samples: Number of different values for each dimension
    :param tolerance: Maximum size for a cell to be subdivided
    :param ax: Plottable object
    """
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

