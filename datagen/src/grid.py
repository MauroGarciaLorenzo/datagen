import numpy as np

from datagen.src.dimensions import Dimension, Cell

def gen_grid(dimensions):
    """
    Generate grid. Every cell is made out of a list of the Dimension objects
    involved in the problem, with the only difference that the lower and upper
    bounds change for each cell.

    :param dimensions: Involved dimensions
    :return: Grid
    """
    independent_dims_subdv = [dim for dim in dimensions if dim.independent_dimension if dim.divs > 1]
    independent_dims_no_subdv = [dim for dim in dimensions if dim.independent_dimension if dim.divs == 1]
    dependent_dims = [dim for dim in dimensions if not dim.independent_dimension]
    n_dims = len(independent_dims_subdv)
    ini = tuple(dim.borders[0] for dim in independent_dims_subdv if dim.divs > 1)
    fin = tuple(dim.borders[1] for dim in independent_dims_subdv if dim.divs > 1)
    div = tuple(dim.divs for dim in independent_dims_subdv if dim.divs > 1)
    total_div = int(np.prod(div))
    if total_div == 0:
        raise Exception(f"Too many divisions defined. div: {div}")
    grid = []
    for i in range(total_div):
        div_indices = np.unravel_index(i, div)
        lower = [
            ini[j] + (fin[j] - ini[j]) / div[j] * div_indices[j]
            for j in range(n_dims)]
        upper = [
            ini[j] + (fin[j] - ini[j]) / div[j] * (div_indices[j] + 1)
            for j in range(n_dims)
        ]
        dims = []
        for j in range(len(independent_dims_subdv)):
            dims.append(
                Dimension(
                    variable_borders=independent_dims_subdv[j].variable_borders,
                    n_cases=independent_dims_subdv[j].n_cases,
                    divs=independent_dims_subdv[j].divs,
                    borders=(lower[j], upper[j]),
                    label=independent_dims_subdv[j].label,
                    tolerance=independent_dims_subdv[j].tolerance,
                    cosphi=independent_dims_subdv[j].cosphi)
            )
        dims.extend(independent_dims_no_subdv)
        dims.extend(dependent_dims)
        grid.append(Cell(dims))
    return grid
