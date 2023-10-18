#  Copyright 2002-2023 Barcelona Supercomputing Center (www.bsc.es)

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""This module is the main entry point for the application. The main
goal is to explore the various cells (or combinations of dimensions) and
produce both a record of execution logs and a DataFrame containing specific
cases and their associated stability."""

import pandas as pd

from .sampling import gen_grid, explore_grid
from .viz import print_results

try:
    from pycompss.api.task import task
    from pycompss.api.api import compss_wait_on
except ImportError:
    from datagen.dummies.task import task
    from datagen.dummies.api import compss_wait_on


def start(dimensions, n_samples, rel_tolerance, ax, func):
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
    :param rel_tolerance: Fraction of the dimension's range that will be used
        as the minimum size of new cells in grid generation. 
        e.g., if rel_tolerance = 0.1 the dimension's tolerance will be 10 % of
        its range
    :param ax: Plottable object
    :param func: Objective function
    """
    for dim in dimensions:
        dim.tolerance = (dim.borders[1] - dim.borders[0]) * rel_tolerance
    grid = gen_grid(dimensions)
    cases_df, execution_logs = explore_grid(ax, cases_df=None, grid=grid,
                                            depth=0, dims_df=pd.DataFrame(),
                                            func=func, n_samples=n_samples)
    print_results(execution_logs, cases_df)
    print("")

