import math
import time
import numpy as np

from .utils import get_case_results
from stability_analysis.operating_point_from_datagenerator import datagen_OP
from stability_analysis.modify_GridCal_grid import assign_Generators_to_grid,assign_PQ_Loads_to_grid, assign_SlackBus_to_grid
from stability_analysis.powerflow import GridCal_powerflow, process_powerflow, slack_bus, fill_d_grid_after_powerflow
from stability_analysis.preprocess import preprocess_data, read_data, process_raw, parameters,read_op_data_excel, admittance_matrix
from stability_analysis.state_space import generate_NET, build_ss, generate_elements
from stability_analysis.analysis import small_signal


# file where objective function is declared (dummy test)
def dummy(case, **kwargs):
    time.sleep(0.0001)
    return round(math.sin(sum(case)) * 0.5 + 0.5), {}

def matmul(case, **kwargs):
    t0 = time.time()
    while(time.time() - t0 < 0.0000):
        m0 = np.random.randint(0, 101, size=(1000, 1000))
        m1 = np.random.randint(0, 101, size=(1000, 1000))
        x = np.dot(m0, m1)
    return round(math.sin(sum(case)) * 0.5 + 0.5), {}

def dummy_linear(case, **kwargs):
    total_sum = sum(case)
    return total_sum/19 > 0.5, {}  # 19 => maximum value among all upper borders


def small_signal_stability(case, **kwargs):
    d_raw_data = kwargs.get("d_raw_data", None)
    d_op = kwargs.get("d_op", None)
    GridCal_grid = kwargs.get("GridCal_grid", None)
    d_grid = kwargs.get("d_grid", None)
    d_sg = kwargs.get("d_sg", None)
    d_vsc = kwargs.get("d_vsc", None)

    d_raw_data, d_op = datagen_OP.generated_operating_point(case, d_raw_data,
                                                            d_op)
    
    d_raw_data, slack_bus_num = choose_slack_bus(d_raw_data)
    
    assign_SlackBus_to_grid.assign_slack_bus(GridCal_grid, slack_bus_num)
    assign_Generators_to_grid.assign_StaticGen(GridCal_grid, d_raw_data, d_op)
    assign_PQ_Loads_to_grid.assign_PQ_load(GridCal_grid, d_raw_data)

    # %% POWER-FLOW

    # Receive system status from OPAL
    # d_grid, GridCal_grid, data_old = process_opal.update_OP_from_RT(d_grid, GridCal_grid, data_old)

    # Get Power-Flow results with GridCal
    pf_results = GridCal_powerflow.run_powerflow(GridCal_grid)

    print('Converged:', pf_results.convergence_reports[0].converged_[0])

    # Update PF results and operation point of generator elements
    d_pf = process_powerflow.update_OP(GridCal_grid, pf_results, d_raw_data)

    # %% FILL d_grid

    d_grid, d_pf = fill_d_grid_after_powerflow.fill_d_grid(d_grid,
                                                           GridCal_grid, d_pf,
                                                           d_raw_data, d_op)

    # %% READ PARAMETERS

    # Get parameters of generator units from excel files & compute pu base
    d_grid = parameters.get_params(d_grid, d_sg, d_vsc)
    
    d_grid = update_control(case, d_grid)

    # Assign slack bus and slack element
    d_grid = slack_bus.assign_slack(d_grid)

    # Compute reference angle (delta_slk)
    d_grid, REF_w, num_slk, delta_slk = slack_bus.delta_slk(d_grid)

    # %% GENERATE STATE-SPACE MODEL

    # Generate AC & DC NET State-Space Model
    l_blocks, l_states, d_grid = generate_NET.generate_SS_NET_blocks(d_grid,
                                                                     delta_slk)

    # Generate generator units State-Space Model
    l_blocks, l_states = generate_elements.generate_SS_elements(d_grid,
                                                                delta_slk,
                                                                l_blocks,
                                                                l_states)

    # %% BUILD FULL SYSTEM STATE-SPACE MODEL

    # Define full system inputs and ouputs
    var_in = ['NET_Rld1']
    var_out = ['all']  # ['GFOR3_w'] #

    # Build full system state-space model
    inputs, outputs = build_ss.select_io(l_blocks, var_in, var_out)
    ss_sys = build_ss.connect(l_blocks, l_states, inputs, outputs)

    # %% SMALL-SIGNAL ANALYSIS

    T_EIG = small_signal.FEIG(ss_sys, True)
    T_EIG.head

    # write to excel
    # T_EIG.to_excel(path.join(path_results, "EIG_" + excel + ".xlsx"))

    if max(T_EIG['real'] >= 0):
        stability = 0
    else:
        stability = 1

    df_op, df_real, df_imag, df_freq, df_damp = (
        get_case_results(T_EIG=T_EIG, d_grid=d_grid))
    output_dataframes = {
        "df_op":df_op, "df_real":df_real, "df_imag":df_imag,
        "df_freq":df_freq, "df_damp":df_damp
    }
    return stability, output_dataframes

def update_control(case, d_grid):
    case_index=case.index

    for i in range(0,len(d_grid['T_VSC'])):
        mode=d_grid['T_VSC'].loc[i,'mode']
        bus=d_grid['T_VSC'].loc[i,'bus']
        
        control_p_mode=[cc for cc in case.index if 'tau' and mode.lower() in cc]
        control_p_mode_bus=[cc for cc in control_p_mode if str(bus) in cc]
        
        control_p_labels=[''.join(filter(lambda x: not x.isdigit(), cc))[:-1].replace(mode.lower(),'')[:-1] for cc in control_p_mode_bus ]
        
        for control_p,control_p_bus in zip(control_p_labels,control_p_mode_bus):
            d_grid['T_VSC'].loc[i,control_p]=case[control_p_bus]
    
    return d_grid
        
def choose_slack_bus(d_raw_data):
    T_generators=d_raw_data['generator'].query('P_SG !=0 or P_GFOR!=0')
    T_generators['deltaP']=T_generators['MBASE']-T_generators['PG']
    T_generators=T_generators.sort_values(by='deltaP', ascending=False).reset_index(drop=True)
    slack_bus=T_generators.loc[0,'I']
    d_raw_data['data_global'].loc[0,'ref_bus']=slack_bus

    if T_generators.loc[0,'P_SG']!=0:
        d_raw_data['data_global'].loc[0,'ref_element']='SG'
    elif T_generators.loc[0,'P_GFOR']!=0:
        d_raw_data['data_global'].loc[0,'ref_element']='GFOR'
    else:
        raise RuntimeError('Error: missing generator at slack bus')
    return d_raw_data, slack_bus

    