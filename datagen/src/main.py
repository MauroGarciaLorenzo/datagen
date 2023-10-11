
import pandas as pd

from .sampling import gen_grid, explore_children
from .viz import print_results

try:
    from pycompss.api.task import task
    from pycompss.api.api import compss_wait_on
except ImportError:
    from datagen.dummies import task
    from datagen.dummies import compss_wait_on


@task(returns=1)
def main(dimensions, n_samples, tolerance, ax, func):
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
    cases_df, execution_logs = explore_children(ax,
                                                cases_df=None,
                                                children_grid=grid,
                                                depth=0,
                                                dims_df=pd.DataFrame(),
                                                func=func,
                                                n_samples=n_samples,
                                                tolerance=tolerance)

    print_results(execution_logs, cases_df)
    print("")

