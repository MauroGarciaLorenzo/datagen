def check_dims(dimensions, tolerance):
    for d in dimensions:
        if (d.borders[1] - d.borders[0]) < tolerance:
            return False

    return True


def flatten_list(data):
    flattened_list = []
    for item in data:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)
    return flattened_list
