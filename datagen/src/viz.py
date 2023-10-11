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
