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

"""This utility module provides functions to check the size of the
dimensions against a specified tolerance to ensure they haven't become too
small. It also includes a function to flatten nested lists into a single
list, which is useful for processing the list of logs generated during the
exploration.
"""


def check_dims(dimensions):
    """This method checks if the size of every dimension is smaller than the
    tolerance declared.

    :param dimensions: Cell dimensions
    :return: True if tolerance is bigger than this difference, false otherwise
    """
    for d in dimensions:
        if (d.borders[1] - d.borders[0]) < d.tolerance:
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
