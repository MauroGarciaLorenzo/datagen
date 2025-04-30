import os

import pandas as pd

from datagen.src.data_ops import sort_df_rows_by_another, sort_df_last_columns


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
        if dataframe is None:
            continue
        for key, value in dataframe.items():
            cu = os.environ.get("COMPUTING_UNITS")
            filename = f"cu{cu}_case_{str(index)}_{key}_seed{str(seed)}"
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", flush=True)
            print(key, flush=True)
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", flush=True)
            print(value, flush=True)
            print("", flush=True)
            print("", flush=True)
            if not os.path.exists(path_results):
                os.makedirs(path_results)
            if isinstance(value, dict):
                write_dataframes_to_excel(
                    value, path_results, f"{filename}.xlsx")
            else:
                pd.DataFrame.to_csv(
                    value, os.path.join(path_results, f"{filename}.csv"))
        index += 1


def save_results(cases_df, dims_df, execution_logs, output_dataframes,
                 dst_dir, execution_time):
    """
    Save set of results including the main cases/dims dataframes plus
    execution logs and all elements inside the output_dataframes dictionary,
    mostly dataframes. These dataframe rows must follow the same sorting
    according to the case ids.
    """
    if dst_dir is None:
        dst_dir = ""

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    cases_df.to_csv(os.path.join(dst_dir, "cases_df.csv"))
    dims_df.to_csv(os.path.join(dst_dir, "dims_df.csv"))
    execution_time_df = pd.DataFrame(
        {"execution_time": [execution_time]})

    execution_time_df.to_csv((os.path.join(dst_dir, "execution_time.csv")), index=False)

    for key, value in output_dataframes.items():
        if isinstance(value, pd.DataFrame):
            # All dataframes should have the same sorting
            sorted_df = sort_df_rows_by_another(cases_df, value, "case_id")
            # Sort columns at the end
            sorted_df = sort_df_last_columns(sorted_df)
            # Save dataframe
            sorted_df.to_csv(os.path.join(dst_dir, f"case_{key}.csv"))
        else:
            for k, v in value.items():
                if isinstance(v, pd.DataFrame):
                    value.to_csv(os.path.join(dst_dir, f"case_{k}.csv"))
                else:
                    print("Wrong format for output dataframes")

    if execution_logs!=None:
        with open(os.path.join(dst_dir, "execution_logs.txt"), "w") as log_file:
            for log_entry in execution_logs:
                log_file.write("Dimensions:\n")
                for dim in log_entry[0]:
                    log_file.write(f"{dim}\n")
                log_file.write(f"Entropy: {log_entry[1]}\n")
                log_file.write(f"Delta Entropy: {log_entry[2]}\n")
                log_file.write(f"Depth: {log_entry[3]}\n")
                log_file.write("\n")
