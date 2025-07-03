from os import path

import pandas as pd

from datagen.src.objective_function import power_flow
from stability_analysis.data import get_data_path
from stability_analysis.preprocess import preprocess_data, read_data, \
    process_raw
from stability_analysis.powerflow import GridCal_powerflow


# Paths to data
path_data = get_data_path()
path_raw = path.join(path_data, "raw")
path_results = path.join(path_data, "results")

# File names

# IEEE 9
raw = "ieee9_6"
excel = "IEEE_9_headers"
excel_data = "IEEE_9"
excel_op = "OperationData_IEEE_9"

# IEEE 118
# raw = "IEEE118busREE_Winter Solved_mod_PQ"
# # excel = "IEEE_118bus_TH" # THÉVENIN
# # excel = "IEEE_118_01" # SG
# excel = "IEEE_118_FULL"
# excel_data = "IEEE_118_FULL"
# excel_op = "OperationData_IEEE_118"

# TEXAS 2000 bus
# raw = "ACTIVSg2000_solved_noShunts"
# excel = "texas_2000"


raw_file = path.join(path_raw, raw + ".raw")
# excel_raw = path.join(path_raw, raw + ".xlsx")
excel_sys = path.join(path_data, "cases/" + excel + ".xlsx")  # empty
excel_sg = path.join(path_data, "cases/" + excel_data + "_data_sg.xlsx")
excel_vsc = path.join(path_data, "cases/" + excel_data + "_data_vsc.xlsx")
excel_op = path.join(path_data, "cases/" + excel_op + ".xlsx")
# %% READ RAW FILE

# Read raw file
d_raw_data = process_raw.read_raw(raw_file)

d_raw_data['generator']['Region'] = 1
d_raw_data['load']['Region'] = 1
d_raw_data['branch']['Region'] = 1
d_raw_data['results_bus']['Region'] = 1

# Preprocess input raw data to match excel file format
preprocess_data.preprocess_raw(d_raw_data)

# Write to excel file
# preprocess_data.raw2excel(d_raw_data,excel_raw)

# Create GridCal Model
gridCal_grid = GridCal_powerflow.create_model(path_raw, raw_file)

# %% READ OPERATION EXCEL FILE

d_op = read_data.read_data(excel_op)

# %% READ EXCEL FILE

# Read data of grid elements from Excel file
d_grid, d_grid_0 = read_data.read_sys_data(excel_sys)

# TO BE DELETED
d_grid = read_data.tempTables(d_grid)

# # Read simulation configuration parameters from Excel file
# sim_config = read_data.get_simParam(excel_sys)

# %% READ EXEC FILES WITH SG AND VSC CONTROLLERS PARAMETERS

d_sg = read_data.read_data(excel_sg)

d_vsc = read_data.read_data(excel_vsc)

# Parse input
kwargs = {
    "func_params": {
        "d_raw_data": d_raw_data,
        "d_op": d_op,
        "gridCal_grid": gridCal_grid,
        "d_grid": d_grid,
    }
}

case = pd.read_csv("cases_IEEE9_example.csv", index_col=0).iloc[0]

# Run power flow
pf_results = power_flow(case, **kwargs)
print("Bus naming:", pf_results.results.bus_names)
print("Resulting voltages:", pf_results.results.voltage)
