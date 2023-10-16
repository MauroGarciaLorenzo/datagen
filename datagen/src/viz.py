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

"""This module is responsible for visualization-related functionality. It
provides functions to print detailed information about each cell's
dimensions and the results obtained during the exploration. It helps in
understanding and visualizing the stability of each case, the entropy,
delta entropy, and depth of the explored cells. Additionally, there's a
function to plot sample data.
"""


def print_grid(grid):
    """Shows each cell's dimensions

    :param grid: list of cells
    """
    print("")
    for i in range(len(grid)):
        print("------", "casilla", i, "------")
        print("samples casilla", grid[i].n_samples)
        for j in grid[i].dimensions:
            print("        variables:", j.variables)
            print("        cases", j.n_cases)
            print("        divisiones", j.divs)
            print("        limites", j.borders)
            print("")
        print("")
        print("")


def print_results(execution_logs, cases_df):
    """Shows the dataframe obtained by the application and the logs for each
    cell: dimensions, entropy, delta entropy and depth

    :param execution_logs: dimensions, entropy, delta entropy and depth of each
                        cell.
    :param cases_df: dataframe containing every case evaluated by the program
                and each evaluation (stability)
    """
    print("samples-stability:")
    print(cases_df)
    print("number of cells: ", len(execution_logs))
    for r in execution_logs:
        print("dimensions: ", r[0])
        print("entropy: ", r[1])
        print("delta entropy: ", r[2])
        print("depth: ", r[3])
        print("")
        print("")


def plot_sample(ax, x, y, z):
    ax.scatter(x, y, z)
