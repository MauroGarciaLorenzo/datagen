def print_grid(grid):
    print("")
    for i in range(len(grid)):
        print("------", "casilla", i, "------")
        print("samples casilla", grid[i].n_samples)
        for j in grid[i].dimensions:
            print("        variables:", j.variables)
            print("        case", j.n_case)
            print("        divisiones", j.divs)
            print("        limites", j.borders)
            print("")
        print("")
        print("")


def plot_sample(ax, x, y, z):
    ax.scatter(x, y, z)
