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
import os

import pandas as pd


def combine_columns_by_prefix(df, sum_columns):
    """Combine columns in a dataframe by common prefixes."""
    result_df = pd.DataFrame()

    for col in df.columns:
        prefix = col.split("_Var")[0]
        if prefix not in result_df.columns:
            result_df[prefix] = sum_columns[prefix]
        else:
            result_df[prefix] += sum_columns[prefix]

    return result_df

def clean_dir(directory):
    if os.path.exists(directory):
        files = os.listdir(directory)
        for file in files:
            path = os.path.join(directory, file)
            if os.path.isfile(path):
                os.remove(path)
    else:
        os.makedirs(directory, exist_ok=True)


def get_dimension(label, dimensions):
    if label == "g_for" or label == "g_fol":
        dim = next(
            (d for d in dimensions
             if d.label == "p_cig"), None)
    else:
        dim = next((d for d in dimensions
                    if d.label == label), None)
    return dim


def check_dims(dimensions):
    """This method checks if the size of every dimension is smaller than the
    tolerance declared.

    :param dimensions: Cell dimensions
    :return: True if tolerance is bigger than this difference, false otherwise
    """
    for d in dimensions:
        if d.independent_dimension:
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


def save_results(cases_df, dims_df, execution_logs):
    result_dir = "results"
    cases_df.to_csv(os.path.join(result_dir, "cases_df.csv"), index=False)

    dims_df.to_csv(os.path.join(result_dir, "dims_df.csv"), index=False)

    with open(os.path.join(result_dir, "execution_logs.txt"), "w") as log_file:
        for log_entry in execution_logs:
            log_file.write("Dimensions:\n")
            for dim in log_entry[0]:
                log_file.write(f"{dim}\n")
            log_file.write(f"Entropy: {log_entry[1]}\n")
            log_file.write(f"Delta Entropy: {log_entry[2]}\n")
            log_file.write(f"Depth: {log_entry[3]}\n")
            log_file.write("\n")
