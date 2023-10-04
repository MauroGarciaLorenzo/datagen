def check_dims(dimensions, tolerance):
    """This method check if the size of every dimension is smaller than the
    tolerance declared.

    :param dimensions: Cell dimensions
    :param tolerance: maximum difference upper-lower bound of a dimension
                    within a cell to be subdivided
    :return: True if tolerance is bigger than this difference, false otherwise
    """
    for d in dimensions:
        if (d.borders[1] - d.borders[0]) < tolerance:
            return False
    return True


def flatten_list(data):
    """This method extracts the values of the list given, obtaining one element
     for each cell.

    :param data: list of logs of the children cells
    :return: flattened list
    """
    flattened_list = []
    for item in data:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)
    return flattened_list
