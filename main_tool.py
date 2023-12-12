# import numpy as np
# import matplotlib.pyplot as plt
# import control as ct
# import pandas as pd

from os import path, getcwd
from tool.preprocess import preprocess_data, read_data, process_raw, parameters,read_op_data_excel, admittance_matrix
from tool.powerflow import GridCal_powerflow, process_powerflow, slack_bus, fill_d_grid_after_powerflow
from tool.state_space import generate_NET, build_ss, generate_elements
from tool.opal import process_opal
from tool.analysis import small_signal
from tool.preprocess.utils import *
from tool.random_operating_point import random_OP
from tool.modify_GridCal_grid import assign_StaticGen_to_grid,assign_PQ_Loads_to_grid
from GridCalEngine.Core.DataStructures import numerical_circuit

# %% SET FILE NAMES AND PATHS

# Paths to data

path_data = path.abspath(getcwd()) + "\\tool\\data" 
path_raw = path.abspath(getcwd()) + "\\tool\\data\\raw"
path_results = path.abspath(getcwd()) + "\\tool\\data\\results"

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
excel_sys = path.join(path_data, "cases/" + excel + ".xlsx") #empty 
excel_sg = path.join(path_data, "cases/" + excel_data + "_data_sg.xlsx") 
excel_vsc = path.join(path_data, "cases/" + excel_data + "_data_vsc.xlsx") 
excel_op = path.join(path_data, "cases/" + excel_op + ".xlsx") 
# %% READ RAW FILE

# Read raw file
d_raw_data = process_raw.read_raw(raw_file)

d_raw_data['generator']['Region']=1
d_raw_data['load']['Region']=1
d_raw_data['branch']['Region']=1
d_raw_data['results_bus']['Region']=1

# Preprocess input raw data to match excel file format
preprocess_data.preprocess_raw(d_raw_data)

# Write to excel file
# preprocess_data.raw2excel(d_raw_data,excel_raw)

# Create GridCal Model
GridCal_grid = GridCal_powerflow.create_model(path_raw, raw_file)

#%% READ OPERATION EXCEL FILE

d_op = read_op_data_excel.read_operation_data_excel(excel_op)

#%% GENERATE RANDOM OPERATING POINT

d_raw_data, d_op= random_OP.random_operating_point(d_op, d_raw_data, GridCal_grid,
                                             n_reg=1, loads_power_factor=0.95,
                                             generators_power_factor=0.95,
                                             all_gfor=False)
              
#%% MODIFY GRIDCAL_GRID

assign_StaticGen_to_grid.assign_StaticGen(GridCal_grid, d_raw_data, d_op)
assign_PQ_Loads_to_grid.assign_PQ_load(GridCal_grid, d_raw_data)

# %% READ EXCEL FILE

# Read data of grid elements from Excel file
d_grid, d_grid_0 = read_data.read_data(excel_sys)

# TO BE DELETED
d_grid = read_data.tempTables(d_grid) 

# # Read simulation configuration parameters from Excel file
# sim_config = read_data.get_simParam(excel_sys)

# %% POWER-FLOW

# Receive system status from OPAL
#d_grid, GridCal_grid, data_old = process_opal.update_OP_from_RT(d_grid, GridCal_grid, data_old)
    
# Get Power-Flow results with GridCal
pf_results = GridCal_powerflow.run_powerflow(GridCal_grid)

print('Converged:',pf_results.convergence_reports[0].converged_[0])

# Update PF results and operation point of generator elements
d_pf = process_powerflow.update_OP(GridCal_grid, pf_results)

#%% FILL d_grid

d_grid, d_pf = fill_d_grid_after_powerflow.fill_d_grid(d_grid, GridCal_grid, d_pf, d_raw_data, d_op)

# d_grid['T_user']=d_grid['T_user'][0:]

# %% READ PARAMETERS

# Get parameters of generator units from excel files & compute pu base
d_grid = parameters.get_params(d_grid, excel_sg, excel_vsc)

# Assign slack bus and slack element
d_grid = slack_bus.assign_slack(d_grid)

# Compute reference angle (delta_slk)
d_grid, REF_w, num_slk, delta_slk = slack_bus.delta_slk(d_grid)

# %% GENERATE STATE-SPACE MODEL

# Generate AC & DC NET State-Space Model
l_blocks, l_states, d_grid = generate_NET.generate_SS_NET_blocks(d_grid, delta_slk)

# Generate generator units State-Space Model
l_blocks, l_states = generate_elements.generate_SS_elements(d_grid, delta_slk, l_blocks, l_states)


# %% BUILD FULL SYSTEM STATE-SPACE MODEL

# Define full system inputs and ouputs
var_in = ['NET_Rld1']
var_out = ['GFOR3_w'] #['all']

# Build full system state-space model
inputs, outputs = build_ss.select_io(l_blocks, var_in, var_out)
ss_sys = build_ss.connect(l_blocks, l_states, inputs, outputs)

# %% SMALL-SIGNAL ANALYSIS

T_EIG = small_signal.FEIG(ss_sys, True)
T_EIG.head

# write to excel
T_EIG.to_excel(path.join(path_results, "EIG_" + excel + ".xlsx"))