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


def save_results(cases_df, dims_df, execution_logs, output_dataframes, seed):
    result_dir = f"results/seed{str(seed)}"

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    cases_df.to_csv(os.path.join(result_dir, "cases_df.csv"), index=False)

    dims_df.to_csv(os.path.join(result_dir, "dims_df.csv"), index=False)

    for key, value in output_dataframes.items():
        value.to_csv(os.path.join(result_dir, f"{key}.csv"))

    with open(os.path.join(result_dir, "execution_logs.txt"), "w") as log_file:
        for log_entry in execution_logs:
            log_file.write("Dimensions:\n")
            for dim in log_entry[0]:
                log_file.write(f"{dim}\n")
            log_file.write(f"Entropy: {log_entry[1]}\n")
            log_file.write(f"Delta Entropy: {log_entry[2]}\n")
            log_file.write(f"Depth: {log_entry[3]}\n")
            log_file.write("\n")


def get_case_results(T_EIG, d_grid):
    df_op = pd.DataFrame()

    T_buses = d_grid['T_buses']
    for i in T_buses.index:
        bus = T_buses.loc[i, 'bus']
        df_op.loc[0, 'V' + str(bus)] = T_buses.loc[i, 'Vm']
        df_op.loc[0, 'theta' + str(bus)] = T_buses.loc[i, 'theta']

    T_gens = d_grid['T_gen']
    for i in T_gens.index:
        col_name = T_gens.loc[i, 'element'] + str(T_gens.loc[i, 'bus'])
        for var in ['P', 'Q', 'Sn']:
            df_op.loc[0, var + '_' + col_name] = T_gens.loc[i, var]

    T_load = d_grid['T_load']
    for i in T_load.index:
        bus = T_load.loc[i, 'bus']
        df_op.loc[0, 'PL' + str(bus)] = T_load.loc[i, 'P']
        df_op.loc[0, 'QL' + str(bus)] = T_load.loc[i, 'Q']

    # add control parameters

    T_EIG = T_EIG.set_index('mode')
    T_EIG = T_EIG.T
    df_real = T_EIG.loc[['real']].reset_index(drop=True)
    df_imag = T_EIG.loc[['imag']].reset_index(drop=True)
    df_freq = T_EIG.loc[['freq']].reset_index(drop=True)
    df_damp = T_EIG.loc[['damp']].reset_index(drop=True)

    return df_op, df_real, df_imag, df_freq, df_damp

def concat_dataframes(dict_list):
    result_dict = {}

    for d in dict_list:
        for key, value in d.items():
            if key not in result_dict:
                result_dict[key] = value
            else:
                result_dict[key] = pd.concat([result_dict[key], value])

    return result_dict