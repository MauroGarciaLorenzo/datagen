import math
import time
import numpy as np
import copy

from datagen.src import sampling

from .utils import get_case_results
from stability_analysis.operating_point_from_datagenerator import datagen_OP
from stability_analysis.modify_GridCal_grid import assign_Generators_to_grid,assign_PQ_Loads_to_grid, assign_SlackBus_to_grid
from stability_analysis.powerflow import GridCal_powerflow, process_powerflow, slack_bus, fill_d_grid_after_powerflow
from stability_analysis.preprocess import preprocess_data, read_data, process_raw, parameters,read_op_data_excel, admittance_matrix
from stability_analysis.state_space import generate_NET, build_ss, generate_elements
from stability_analysis.analysis import small_signal

from stability_analysis.powerflow import check_feasibility

from GridCalEngine.Simulations.PowerFlow.power_flow_options import ReactivePowerControlMode, SolverType
from GridCalEngine.Simulations.OPF.NumericalMethods import ac_opf
from GridCalEngine.Simulations.OPF.NumericalMethods.ac_opf import run_nonlinear_opf, ac_optimal_power_flow
from GridCalEngine.Core.DataStructures.numerical_circuit import compile_numerical_circuit_at
import GridCalEngine.api as gce

from .utils_obj_fun import *

def feasible_power_flow_ACOPF(case,N_pf, **kwargs):
    d_raw_data = kwargs.get("d_raw_data", None)
    d_op = kwargs.get("d_op", None)
    GridCal_grid = kwargs.get("GridCal_grid", None)
    d_grid = kwargs.get("d_grid", None)
    d_sg = kwargs.get("d_sg", None)
    d_vsc = kwargs.get("d_vsc", None)
    voltage_profile = kwargs.get("voltage_profile", None)
    v_min_v_max_delta_v = kwargs.get("v_min_v_max_delta_v", None)
    V_set = kwargs.get("V_set", None)

    if voltage_profile != None and v_min_v_max_delta_v == None:
        print('Error: Voltage profile option selected but v_min, v_max, and delta_v are missing')
        return None, None, None

    if voltage_profile != None and V_set != None:
        print('Error: Both Voltage profile and V_set option is selected. Choose only one of them')
        return None, None, None

    if voltage_profile == None and V_set == None:
        print('Error: Neither Voltage profile or V_set option is selected. Choose one of them')
        return None, None, None

    d_raw_data, d_op = datagen_OP.generated_operating_point(case, d_raw_data,
                                                            d_op)
    d_raw_data, slack_bus_num = choose_slack_bus(d_raw_data)
    i_slack=int(d_raw_data['generator'].query('I == @slack_bus_num').index[0])

    # slack_bus_num=80
    assign_SlackBus_to_grid.assign_slack_bus(GridCal_grid, slack_bus_num)

    if voltage_profile != None:
        vmin = v_min_v_max_delta_v[0]
        vmax = v_min_v_max_delta_v[1]
        delta_v = v_min_v_max_delta_v[2]

        voltage_profile_list, indx_id = sampling.gen_voltage_profile(vmin, vmax, delta_v, d_raw_data, slack_bus_num,
                                                                     GridCal_grid)

        assign_Generators_to_grid.assign_PVGen(GridCal_grid=GridCal_grid, d_raw_data=d_raw_data, d_op=d_op,
                                               voltage_profile_list=voltage_profile_list, indx_id=indx_id)

    elif V_set != None:
        assign_Generators_to_grid.assign_PVGen(GridCal_grid=GridCal_grid, d_raw_data=d_raw_data, d_op=d_op, V_set=V_set)

    assign_PQ_Loads_to_grid.assign_PQ_load(GridCal_grid, d_raw_data)
    
    # %% Run 1st POWER-FLOW

    # Receive system status from OPAL
    # d_grid, GridCal_grid, data_old = process_opal.update_OP_from_RT(d_grid, GridCal_grid, data_old)

    # Get Power-Flow results with GridCal
    pf_results = GridCal_powerflow.run_powerflow(GridCal_grid)

    print('Converged:', pf_results.convergence_reports[0].converged_[0])
    

    # Update PF results and operation point of generator elements
    d_pf_original = process_powerflow.update_OP(GridCal_grid, pf_results, d_raw_data)

    d_pf_original = additional_info_results(d_pf_original, i_slack, pf_results)

    nc = compile_numerical_circuit_at(GridCal_grid)
    nc.generator_data.cost_0[:] = 0
    nc.generator_data.cost_1[:] = 0
    nc.generator_data.cost_2[:] = 0
    pf_options = gce.PowerFlowOptions(solver_type=gce.SolverType.NR, verbose=1, tolerance=1e-8, max_iter=50)
#    d_opf_results = ac_optimal_power_flow(Pref=np.array(d_pf_original['pf_gen']['P']), slack_bus_num=i_slack, nc=nc, pf_options=pf_options, plot_error=True)
    d_opf_results = ac_optimal_power_flow(nc=nc, pf_options=pf_options, plot_error=True)

    d_opf = return_d_opf(d_raw_data, d_opf_results)

    path='./datagen/src/results/'
    write_csv(d_pf_original, path, N_pf, 'PF_orig')
    write_csv(d_opf, path, N_pf, 'OPF')

    print('')

#    return_d_opf()

def small_signal_stability(case, **kwargs):
    d_raw_data = kwargs.get("d_raw_data", None)
    d_op = kwargs.get("d_op", None)
    GridCal_grid = kwargs.get("GridCal_grid", None)
    d_grid = kwargs.get("d_grid", None)
    d_sg = kwargs.get("d_sg", None)
    d_vsc = kwargs.get("d_vsc", None)
    d_pf = kwargs.get("d_pf", None)

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
        "df_op": df_op, "df_real": df_real, "df_imag": df_imag,
        "df_freq": df_freq, "df_damp": df_damp
    }
    return stability, output_dataframes


def return_d_opf(d_raw_data, d_opf_results):
    df_opf_bus = pd.DataFrame(
        {'bus': d_raw_data['results_bus']['I'], 'Vm': d_opf_results.Vm, 'theta': d_opf_results.Va})
    df_opf_gen_pre = pd.DataFrame(
        {'bus': d_raw_data['generator']['I'], 'P': d_opf_results.Pg, 'Q': d_opf_results.Qg})
    df_opf_gen = pd.merge(df_opf_gen_pre, df_opf_bus[['bus', 'Vm', 'theta']], on='bus', how='left')
    d_opf = {'df_opf_bus': df_opf_bus, 'df_opf_gen': df_opf_gen}
    return d_opf


def write_csv(d, path, N_pf, filename):
    for df_name, df in d.items():
        filename=filename+'_'+str(N_pf)+'_'+str(df_name)
        pd.DataFrame.to_csv(df,path+filename+".csv")
