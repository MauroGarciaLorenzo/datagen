import os.path
import sys

import numpy as np
from matplotlib import pyplot as plt

from datagen.src.dimensions import Dimension
from datagen.src.start_app import start
from datagen.src.objective_function_ACOPF import *

# from datagen.src.objective_function import small_signal_stability

try:
    from pycompss.api.task import task
    from pycompss.api.api import compss_wait_on
except ImportError:
    from datagen.dummies.task import task
    from datagen.dummies.api import compss_wait_on

from os import path, getcwd
from stability_analysis.data import get_data_path
from stability_analysis.preprocess import preprocess_data, read_data, \
    process_raw, parameters, admittance_matrix
from stability_analysis.powerflow import GridCal_powerflow, process_powerflow, \
    slack_bus, fill_d_grid_after_powerflow
from stability_analysis.state_space import generate_NET, build_ss, \
    generate_elements
from stability_analysis.opal import process_opal
from stability_analysis.analysis import small_signal
from stability_analysis.preprocess.utils import *
# from stability_analysis.random_operating_point import random_OP
from stability_analysis.modify_GridCal_grid import assign_Generators_to_grid, \
    assign_PQ_Loads_to_grid
# from GridCalEngine.Core.DataStructures import numerical_circuit

# from datagen.src.SS_stab import small_signal_stability

@task()
def main():
    # %% SET FILE NAMES AND PATHS

    # Paths to data
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    path_data = get_data_path()
    path_raw = path.join(path_data, "raw")

    # File names

    gridname='IEEE118'#'IEEE9'#

    if gridname == 'IEEE9':
    # # # IEEE 9
        raw = "ieee9_6"
        excel = "IEEE_9_headers"
        excel_data = "IEEE_9"
        excel_op = "OperationData_IEEE_9"

    elif gridname=='IEEE118':
        # IEEE 118
        raw = "IEEE118busREE_Winter_Solved_mod_PQ_91Loads"
        # excel = "IEEE_118bus_TH" # THÉVENIN
        # excel = "IEEE_118_01" # SG
        excel = "IEEE_118_FULL_headers"
        excel_data = "IEEE_118_FULL"
        excel_op = "OperationData_IEEE_118"
        excel_lines_ratings = "IEEE_118_Lines"

    # TEXAS 2000 bus
    # raw = "ACTIVSg2000_solved_noShunts"
    # excel = "texas_2000"


    raw_file = path.join(path_raw, raw + ".raw")
    # excel_raw = path.join(path_raw, raw + ".xlsx")
    excel_sys = path.join(path_data, "cases/" + excel + ".xlsx") #empty
    excel_sg = path.join(path_data, "cases/" + excel_data + "_data_sg.xlsx")
    excel_vsc = path.join(path_data, "cases/" + excel_data + "_data_vsc.xlsx")
    excel_op = path.join(path_data, "cases/" + excel_op + ".xlsx")

    if gridname == 'IEEE118':
        excel_lines_ratings = path.join(path_data, "cases/" + excel_lines_ratings + ".csv")

    # %% READ OPERATION EXCEL FILE

    d_op = read_data.read_data(excel_op)

    # %% READ RAW FILE

    # Read raw file
    d_raw_data = process_raw.read_raw(raw_file)

    if gridname == 'IEEE9':
        # For the IEEE 9-bus system
        d_raw_data['generator']['Region']=1
        d_raw_data['load']['Region']=1
        d_raw_data['branch']['Region']=1
        d_raw_data['results_bus']['Region']=1

    elif gridname == 'IEEE118':
        # FOR the 118-bus system
        d_raw_data['generator']['Region']=d_op['Generators']['Region']
        d_raw_data['load']['Region']=d_op['Loads']['Region']
        # d_raw_data['branch']['Region']=1
        d_raw_data['results_bus']['Region']=d_op['Buses']['Region']
        d_raw_data['generator']['MBASE']=d_op['Generators']['Snom']

        lines_ratings=pd.read_csv(excel_lines_ratings)


    # Preprocess input raw data to match excel file format
    preprocess_data.preprocess_raw(d_raw_data)

    # Write to excel file
    # preprocess_data.raw2excel(d_raw_data,excel_raw)

    #%% Create GridCal Model
    GridCal_grid = GridCal_powerflow.create_model(path_raw, raw_file)

    for line in GridCal_grid.lines:
        bf = int(line.bus_from.code)
        bt = int(line.bus_to.code)

        line.rate=lines_ratings.loc[lines_ratings.query('Bus_from == @bf and Bus_to == @bt').index[0],'Max Flow (MW)']
        # print(line.rate)

    for trafo in GridCal_grid.transformers2w:
        bf = int(trafo.bus_from.code)
        bt = int(trafo.bus_to.code)

        trafo.rate=lines_ratings.loc[lines_ratings.query('Bus_from == @bf and Bus_to == @bt').index[0],'Max Flow (MW)']
        # print(trafo.rate)

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

    p_sg=[]
    p_cig=[]

    for i in range(len(d_op['Generators'])):
      p_sg.append((d_op['Generators']['Pmin_SG'].iloc[i], d_op['Generators']['Pmax_SG'].iloc[i]))
      p_cig.append((d_op['Generators']['Pmin_CIG'].iloc[i],d_op['Generators']['Pmax_CIG'].iloc[i]))

    p_loads = list(d_op['Loads']['Load_Participation_Factor'])

    loads_power_factor = 0.98
    generators_power_factor = 0.98

    n_samples = 1
    n_cases = 1

    rel_tolerance = 0.01
    max_depth = 2
    dimensions = dict()
    dimensions = [
          Dimension(label="p_sg", variable_borders=p_sg,
                    n_cases=n_cases, divs=2,
                    borders=(d_op['Generators']['Pmin_SG'].sum(),
                    d_op['Generators']['Pmax_SG'].sum()),
                    independent_dimension=True, cosphi=generators_power_factor),
          Dimension(label="p_cig", variable_borders=p_cig,
                    n_cases=n_cases, divs=1,
                    borders=(d_op['Generators']['Pmin_CIG'].sum(),
                    d_op['Generators']['Pmax_CIG'].sum()),
                    independent_dimension=True,
                    cosphi=generators_power_factor),
          Dimension(label="perc_g_for", variable_borders=[(0,1)],
                    n_cases=n_cases, divs=1, borders=(0, 1),
                    independent_dimension=True, cosphi=None),
          Dimension(label="p_load", values=p_loads,
                    n_cases=n_cases, divs=1,
                    independent_dimension=False,
                    cosphi=loads_power_factor)
                  ]

    for d in list(d_op['Generators']['BusNum']):
        dimensions.append(Dimension(label='tau_droop_f_gfor_'+str(d), n_cases=n_cases,
                                    divs=1, borders=(0.01,0.2),
                                    independent_dimension=True,
                                    cosphi=None))

        dimensions.append(Dimension(label='tau_droop_u_gfor_'+str(d), n_cases=n_cases,
                                    divs=1, borders=(0.01,0.2),
                                    independent_dimension=True,
                                    cosphi=None))

        dimensions.append(Dimension(label='tau_droop_f_gfol_'+str(d), n_cases=n_cases,
                                    divs=1, borders=(0.01,0.2),
                                    independent_dimension=True,
                                    cosphi=None))

        dimensions.append(Dimension(label='tau_droop_u_gfol_'+str(d), n_cases=n_cases,
                                    divs=1, borders=(0.01,0.2),
                                    independent_dimension=True,
                                    cosphi=None))


    voltage_profile=True
    v_min_v_max_delta_v=[0.95,1.05,0.02]

    N_pf=1
    stability_array = []
    output_dataframes_array = []

    func_params = {"N_pf":N_pf, "d_raw_data": d_raw_data, "d_op":d_op,
                   "GridCal_grid":GridCal_grid, "d_grid":d_grid, "d_sg":d_sg,
                    "d_vsc":d_vsc, "voltage_profile":voltage_profile,
                    "v_min_v_max_delta_v":v_min_v_max_delta_v}
    seed = 17

    cases_df, dims_df, execution_logs, output_dataframes = start(
                dimensions=dimensions, n_samples=n_samples,
        rel_tolerance=rel_tolerance, func=feasible_power_flow_ACOPF,
        max_depth=max_depth, seed=seed, func_params=func_params, log_dir=log_dir
    )

    stability_array = compss_wait_on(stability_array)
    output_dataframes_array = compss_wait_on(output_dataframes_array)


if __name__ == "__main__":
    main()
# d_pf_original, d_pf, d_raw_data = feasible_power_flow(case=case,
#                                          d_raw_data=d_raw_data,
#                                          d_op=d_op,
#                                          GridCal_grid=GridCal_grid,
#                                          d_grid=d_grid, d_sg=d_sg,
#                                          d_vsc=d_vsc,
#                                          # voltage_profile=voltage_profile,
#                                          # v_min_v_max_delta_v=v_min_v_max_delta_v
#                                          V_set=V_set
#                                          )

# stability, output_dataframes= small_signal_stability(case=case, d_raw_data = d_raw_data,
#                                                      d_op = d_op,
#                                                      GridCal_grid = GridCal_grid,
#                                                      d_grid = d_grid,
#                                                      d_sg = d_sg,
#                                                      d_vsc = d_vsc,
#                                                      d_pf = d_pf,
#                                                      )