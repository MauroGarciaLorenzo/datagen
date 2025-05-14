from typing import Sequence

import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)

from datagen.src.constants import NAN_COLUMN_NAME


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


def sort_df_rows_by_another(df1, df2, column_name):
    """
    Sorts rows of df2 based on the ordering of values in df1 for a specified
    column.

    Args:
        df1 (pd.DataFrame): First DataFrame containing the ordering.
        df2 (pd.DataFrame): Second DataFrame to be sorted based on df1.
        column_name (str): The name of the column of interest in both
            DataFrames.

    Returns:
        pd.DataFrame: A new DataFrame sorted according to df1's order.
    """
    ordered_index = df1.set_index(column_name).index
    sorted_df2 = df2.set_index(column_name).loc[
        ordered_index].reset_index()
    return sorted_df2


def sort_df_last_columns(df):
    """
    Sort dataframe columns so that the selected columns appear at the end.
    """
    cols = df.columns.tolist()
    cols_at_the_end = ["case_id", "Stability"]
    for remove_col in cols_at_the_end:
        cols.remove(remove_col)
    for add_col in cols_at_the_end:
        cols.append(add_col)
    df = df[cols]
    return df


def concat_df_dict(*dicts):
    """
    Receive a list of dictionaries of dataframes and concatenate the dataframes
    to return a unique dictionary of dataframes. Only deal with dataframes
    inside the dictionary, do not handle other types of data.
    Example:
      >> dict_list = [{'a': df1, 'b': df2}, {'a': df3, 'b': df4}]
      >> result_dict = {'a': pd.concat([df1, df3]), 'b': pd.concat([df2, df4])}

    TODO: we can simplify this function now that we do not need to preserve
     the row order on each dataframe, since they are sorted at the end of the
     run thanks to the use of a unique ID per case
    """
    # Return empty dict if all contents are empty
    if all_empty(dicts):
        return {}

    # Consider case of list inside a list
    if isinstance(dicts, Sequence):
        if len(dicts) == 1 and isinstance(dicts[0], Sequence):
            dicts = dicts[0]

    # Collect columns of the different dataframes from the first correct
    # item in the sequence
    cols_dict = {}
    n_subdicts = None
    for d in dicts:
        # Null check
        if not d:
            raise ValueError(
                f"Input must be a list of non-empty dictionaries "
                f"that contain dataframes. Received type {type(d)}.")
        # Check all cases have the same number of sub-dictionaries
        if n_subdicts is not None:
            if n_subdicts != len(d):
                raise ValueError("Differing number of sub-dictionaries while "
                                 "concatenating.")
        else:
            n_subdicts = len(d)
        # Get number of columns from each sub-dict
        if d is None:
            continue
        if isinstance(d, dict):
            for df_label, df in d.items():
                # Skip if columns were already filled
                if df_label in cols_dict:
                    continue
                # Store columns if the dataframe is not empty/undefined
                if isinstance(df, pd.DataFrame) and not df.empty:
                    if df.columns[0] != NAN_COLUMN_NAME:
                        cols_dict[df_label] = df.columns.to_list()
        else:
            raise ValueError("Input must be a list of dictionaries")

    # Concatenate items missing in cols_dict that are all NaN dataframes
    output_dict = {}
    if len(cols_dict) != n_subdicts:
        for d in dicts:
            for df_label, df in d.items():
                # Skip if already in cols_dict: next loop will deal with it
                if df_label in cols_dict:
                    continue
                # Concatenate
                if df_label not in output_dict:
                    output_dict[df_label] = df
                else:
                    output_dict[df_label] = pd.concat(
                        [output_dict[df_label], df], axis=0,
                        ignore_index=True)

    # Do normal concatenation case with well-formed dataframes
    if len(cols_dict) != 0:
        for d in dicts:
            for df_label, cols in cols_dict.items():
                # Get row to be appended
                if d[df_label].columns[0] == NAN_COLUMN_NAME:
                    # 'd' only contains NaN dataframes with no column names
                    n = len(d[df_label])
                    m = len(cols)
                    to_append = pd.DataFrame(np.full((n, m), np.nan),
                                             columns=cols)
                    # Add content of additional columns that may not be NaN
                    if len(d[df_label].columns) > 1:
                        missing_cols_df = d[df_label].drop(NAN_COLUMN_NAME,
                                                           axis=1)
                        to_append[missing_cols_df.columns] = missing_cols_df
                elif isinstance(d, dict):
                    # Normal case: directly append the content of the input df
                    to_append = d[df_label]
                else:
                    raise ValueError(
                        f"Input must be a list of dictionaries of dataframes. "
                        f"Received type {type(d)} with entry '{df_label}' of "
                        f"type {type(d[df_label])}.")

                # Concatenate dataframes
                if df_label not in output_dict:
                    output_dict[df_label] = to_append
                else:
                    output_dict[df_label] = pd.concat(
                        [output_dict[df_label], to_append], axis=0,
                        ignore_index=True)

    return output_dict


def all_empty(data):
    if isinstance(data, (list, tuple)):  # Check for list or tuple
        return all(all_empty(item) for item in data)
    elif isinstance(data, dict):  # Check for dictionary
        # Dict is empty when no keys exist
        return all_empty(list(data.values())) and not data
    else:  # For non-container types, consider them "non-empty" if present
        return False
