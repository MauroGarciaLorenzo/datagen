from os import path, getcwd
import numpy as np
from matplotlib import pyplot as plt

from datagen.src.dimensions import Dimension
from datagen.src.start_app import start
from datagen.src.objective_function import dummy

try:
    from pycompss.api.task import task
    from pycompss.api.api import compss_wait_on
except ImportError:
    from datagen.dummies.task import task
    from datagen.dummies.api import compss_wait_on

#%%
path2tool='C:\\Users\\Francesca\\Documents\\SS_pyTOOL'

# %% SET FILE NAMES AND PATHS

# Paths to data

path_data = path2tool + "\\tool\\data" 
path_raw = path2tool + "\\tool\\data\\raw"
path_results = path2tool + "\\tool\\data\\results"

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
#%%
@task()
def main():
    """
    In this method we work with dimensions (main axes), which represent a
    list of variables. For example, the value of each variable of a concrete
    dimension could represent the power supplied by a generator, while the
    value linked to that dimension should be the total sum of energy produced.

    For each dimension it must be declared:
        -variables: list of variables represented by tuples containing its
                lower and upper borders.
        -n_cases: number of cases taken for each sample (each sample represents
                the total sum of a dimension). A case is a combination of
                variables where all summed together equals the sample.
        -divs: number of divisions in that dimension. It will be the growth
                order of the number of cells
        -lower: lower bound of the dimension (minimum value of a sample)
        -upper: upper bound of the dimension (maximum value of a sample)
        -label: dimension identifier

    Apart from that, it can also be specified the number of samples and
    the relative tolerance (indicates the portion of the size of the original
    dimension). For example, if we have a dimension of size 10 and relative
    tolerance is 0.5, the smallest cell in this dimension will have size 5.
    Lastly, user should provide the objective function. As optional parameters,
    the user can define:
        -use_sensitivity: a boolean indicating whether sensitivity analysis is
        used or not.
        -ax: plot axes in case it is desired to show stability points and cells
        divisions (dimensions length must be 2). Plots saved in
        "datagen/results/figures".
        -plot_boxplot: a boolean indicating whether boxplots for each variable
        must be obtained or not. Plots saved in "datagen/results/figures".
    """
    
    # p_sg = [(0, 2), (0, 1.5), (0, 1.5)]
    # p_cig = [(0, 1), (0, 1.5), (0, 1.5), (0, 2)]
    # tau_f_g_for = [(0., 2)]
    # tau_v_g_for = [(0., 2)]
    # tau_p_g_for = [(0., 2)]
    # tau_q_g_for = [(0., 2)]
    
    p_sg = [(d_op['Generators']['Pmin'].iloc[i],d_op['Generators']['Pmax_SG'].iloc[i]) for i in range(len(d_op['Generators']))]
    p_cig= [(d_op['Generators']['Pmin'].iloc[i],d_op['Generators']['Pmax_CIG'].iloc[i]) for i in range(len(d_op['Generators']))]
    
    loads_power_factor=0.95,
    generators_power_factor=0.95
    
    tau_f_g_for = [(0., 2)]
    tau_v_g_for = [(0., 2)]
    tau_p_g_for = [(0., 2)]
    tau_q_g_for = [(0., 2)]
    
    
    n_samples = 3
    n_cases = 3

    rel_tolerance = 0.01
    max_depth = 3
    dimensions = [
        Dimension(variables=p_sg, n_cases=n_cases, divs=2, borders=(d_op['Generators']['Pmin'].sum(), d_op['Generators']['Pmax_SG'].sum()),
                  label="p_sg"),
        Dimension(variables=p_cig, n_cases=n_cases, divs=1, borders=(d_op['Generators']['Pmin'].sum(), d_op['Generators']['Pmax_CIG'].sum()),
                  label="p_cig")#,
        ]
    #     Dimension(variables=tau_f_g_for, n_cases=n_cases, divs=1,
    #               borders=(0, 2), label="tau_f_g_for"),
    #     Dimension(variables=tau_v_g_for, n_cases=n_cases, divs=1,
    #               borders=(0, 2), label="tau_v_g_for"),
    #     Dimension(variables=tau_p_g_for, n_cases=n_cases, divs=1,
    #               borders=(0, 2), label="tau_p_g_for"),
    #     Dimension(variables=tau_q_g_for, n_cases=n_cases, divs=1,
    #               borders=(0, 2), label="tau_q_g_for")
    # ]

    fig, ax = plt.subplots()
    use_sensitivity = True
    cases_df, dims_df, execution_logs = \
        start(dimensions, n_samples, rel_tolerance, dummy, max_depth,
              use_sensitivity=use_sensitivity, ax=ax, divs_per_cell=2, seed=1)

if __name__ == "__main__":
    main()
