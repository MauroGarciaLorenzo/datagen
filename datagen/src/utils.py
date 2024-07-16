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
import yaml
import sys

import pandas as pd
from stability_analysis.data import get_data_path

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


def write_dataframes_to_excel(df_dict, path, filename):
    excel_file_path = os.path.join(path, filename)
    # Create a Pandas Excel writer using xlsxwriter as the engine
    with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
        # Iterate over each key-value pair in the dictionary
        for sheet_name, df in df_dict.items():
            # Write each DataFrame to a separate sheet with the sheet name as the key
            if isinstance(df, dict):
                df = pd.DataFrame(df)
            if isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                print(f'Warning: Not writing {sheet_name}. '
                      f'Not a DataFrame or Series')


def save_dataframes(output_dataframes_array, path_results, seed):
    index = 0
    for dataframe in output_dataframes_array:
        for key, value in dataframe.items():
            cu = os.environ.get("COMPUTING_UNITS")
            filename = f"cu{cu}_case_{str(index)}_{key}_seed{str(seed)}.xlsx"
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", flush=True)
            print(key, flush=True)
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", flush=True)
            print(value, flush=True)
            print("", flush=True)
            print("", flush=True)
            if not os.path.exists(path_results):
                os.makedirs(path_results)
            if isinstance(value, dict):
                write_dataframes_to_excel(value, path_results, filename)
            else:
                pd.DataFrame.to_excel(value,
                                      os.path.join(path_results, filename))
        index += 1


def save_results(cases_df, dims_df, execution_logs, output_dataframes, seed, dst_dir):
    if dst_dir is None: dst_dir = ""
    if dst_dir != "":
        dst_dir += "/"
    result_dir = f"{dst_dir}results/seed{str(seed)}"

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    cases_df.to_csv(os.path.join(result_dir, "cases_df.csv"), index=False)
    dims_df.to_csv(os.path.join(result_dir, "dims_df.csv"), index=False)

    case = 0
    for key, value in output_dataframes.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(os.path.join(result_dir, f"case{case}_{key}.csv"))
        else:
            for k, v in value.items():
                if isinstance(v, pd.DataFrame):
                    value.to_csv(os.path.join(result_dir, f"case{case}_{k}.csv"))
                else:
                    print("Wrong format for output dataframes")
        case += 1

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


def get_args():
    working_dir, path_data, setup_path = parse_args()
    if not working_dir:
        working_dir = ""
    if not path_data:
        path_data = get_data_path()
    print("Working directory: ", working_dir)
    print("Path data: ", path_data)
    if setup_path:
        setup = load_yaml(setup_path)
    else:
        current_directory = os.path.dirname(__file__)
        setup = load_yaml(f"{current_directory}/../../setup/default_setup.yaml")

    n_pf = setup["n_pf"]
    voltage_profile = setup["voltage_profile"]
    v_min_v_max_delta_v = setup["v_min_v_max_delta_v"]
    loads_power_factor = setup["loads_power_factor"]
    generators_power_factor = setup["generators_power_factor"]
    n_samples = setup["n_samples"]
    n_cases = setup["n_cases"]
    rel_tolerance = setup["rel_tolerance"]
    max_depth = setup["max_depth"]
    seed = setup["seed"]
    grid_name = setup["grid_name"]
    print(f'N_PF: {n_pf}')
    print(f'Voltage profile: {voltage_profile}')
    print(f'V min, V max, Delta V: {v_min_v_max_delta_v}')
    print(f'Loads power factor: {loads_power_factor}')
    print(f'Generators power factor: {generators_power_factor}')
    print(f'Number of samples: {n_samples}')
    print(f'Number of cases: {n_cases}')
    print(f'Relative tolerance: {rel_tolerance}')
    print(f'Max depth: {max_depth}')
    print(f'Seed: {seed}')
    print(f'Grid name: {grid_name}')
    return generators_power_factor, grid_name, loads_power_factor, n_cases, n_pf, n_samples, path_data, seed, v_min_v_max_delta_v, voltage_profile, working_dir


def parse_args():
    working_dir = None
    path_data = None
    setup_content = None

    args = sys.argv[1:]
    while args:
        arg = args.pop(0)
        if arg.startswith('--working_dir='):
            working_dir = arg.split('=', 1)[1]
        elif arg.startswith('--path_data='):
            path_data = arg.split('=', 1)[1]
        elif arg.startswith('--setup='):
            setup_content = arg.split('=', 1)[1]
        else:
            print(f"Error: Argument not recognized {arg}")
            sys.exit(1)

    return working_dir, path_data, setup_content


def load_yaml(content):
    try:
        content = os.path.expanduser(content)
        # Try to interpret content as a file path
        with open(content, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        try:
            # If not a file, try to interpret content as a YAML string
            return yaml.safe_load(content)
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML content: {exc}")
            sys.exit(1)
