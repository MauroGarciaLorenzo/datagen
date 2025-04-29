
from stability_analysis.optimal_power_flow import process_optimal_power_flow

from GridCalEngine.Simulations.OPF.NumericalMethods.ac_opf import run_nonlinear_opf, ac_optimal_power_flow
from GridCalEngine.DataStructures.numerical_circuit import compile_numerical_circuit_at
import GridCalEngine.api as gce

from .constants import NAN_COLUMN_NAME, OUTPUT_DF_NAMES, COMPUTING_TIME_NAMES
from .utils_obj_fun import *
from .sampling import gen_voltage_profile

try:
    from pycompss.api.task import task
    from pycompss.api.api import compss_wait_on
except ImportError:
    from datagen.dummies.task import task
    from datagen.dummies.api import compss_wait_on

import time

from GridCalEngine.Simulations.PowerFlow.power_flow_worker import multi_island_pf_nc


def feasible_power_flow_ACOPF(case, **kwargs):
    """
    Runs the alternating current optimal power flow (ACOPF) stability analysis.
    :param case: pandas DataFrame with the case parameters
    :param kwargs: dictionary with additional parameters
    :return: stability: 0 if the system is stable, 1 otherwise
    :return: output_dataframes: Mandatory dictionary with at least the
        entries that contain dataframes (None entries if feasibility fails)
    """
    print("EV", flush=True)
    func_params = kwargs.get("func_params")
    generator = kwargs.get("generator", None)
    dimensions = kwargs.get("dimensions", None)

    n_pf = func_params.get("n_pf", None)
    d_raw_data = func_params.get("d_raw_data", None)
    d_op = func_params.get("d_op", None)
    gridCal_grid = func_params.get("gridCal_grid", None)
    d_grid = func_params.get("d_grid", None)
    d_sg = func_params.get("d_sg", None)
    d_vsc = func_params.get("d_vsc", None)
    voltage_profile = func_params.get("voltage_profile", None)
    v_min_v_max_delta_v = func_params.get("v_min_v_max_delta_v", None)
    v_set = func_params.get("v_set", None)

    # Remove the id and make sure case is fully numeric
    case_id = case["case_id"]
    case = case.drop("case_id")
    case = case.astype(float)

    # Initialize essential output dataframes to None
    computing_times = pd.DataFrame(
        {name: np.nan for name in COMPUTING_TIME_NAMES}, index=[0])
    output_dataframes = {}
    for df_name in OUTPUT_DF_NAMES:
        output_dataframes[df_name] = pd.DataFrame({NAN_COLUMN_NAME: [np.nan]})

    if voltage_profile is not None and v_min_v_max_delta_v is None:
        raise ValueError('Voltage profile option selected but v_min, v_max, '
                         'and delta_v are missing')
    if voltage_profile is not None and v_set is not None:
        raise ValueError('Both Voltage profile and v_set option is selected. '
                         'Choose only one of them')
    if voltage_profile is None and v_set is None:
        raise ValueError('Neither Voltage profile or v_set option is selected.'
                         ' Choose one of them')

    d_raw_data, d_op = datagen_OP.generated_operating_point(case, d_raw_data,
                                                            d_op)
    d_raw_data, slack_bus_num = choose_slack_bus(d_raw_data)
    i_slack=int(d_raw_data['generator'].query('I == @slack_bus_num').index[0])

    # slack_bus_num=80
    assign_SlackBus_to_grid.assign_slack_bus(gridCal_grid, slack_bus_num)

    if voltage_profile != None:
        vmin = v_min_v_max_delta_v[0]
        vmax = v_min_v_max_delta_v[1]
        delta_v = v_min_v_max_delta_v[2]

        voltage_profile_list, indx_id = gen_voltage_profile(vmin, vmax, delta_v, d_raw_data, slack_bus_num,
                                                                     gridCal_grid, generator=generator)

        assign_Generators_to_grid.assign_PVGen(GridCal_grid=gridCal_grid, d_raw_data=d_raw_data, d_op=d_op,
                                               voltage_profile_list=voltage_profile_list, indx_id=indx_id)

    elif v_set != None:
        assign_Generators_to_grid.assign_PVGen(GridCal_grid=gridCal_grid, d_raw_data=d_raw_data, d_op=d_op, V_set=v_set)

    assign_PQ_Loads_to_grid.assign_PQ_load(gridCal_grid, d_raw_data)

    # %% Run 1st POWER-FLOW

    # Receive system status from OPAL
    # d_grid, gridCal_grid, data_old = process_opal.update_OP_from_RT(d_grid, gridCal_grid, data_old)

    # Get Power-Flow results with GridCal
    pf_results = GridCal_powerflow.run_powerflow(gridCal_grid,Qconrol_mode=ReactivePowerControlMode.Direct)

    print('Converged:', pf_results.convergence_reports[0].converged_[0])


    # Update PF results and operation point of generator elements
    d_pf_original = process_powerflow.update_OP(gridCal_grid, pf_results, d_raw_data)
    d_pf_original['info']=pd.DataFrame()
    d_pf_original = additional_info_PF_results(d_pf_original, i_slack, pf_results, n_pf)

    nc = compile_numerical_circuit_at(gridCal_grid)
    nc.generator_data.cost_0[:] = 0
    nc.generator_data.cost_1[:] = 0
    nc.generator_data.cost_2[:] = 0
    pf_options = gce.PowerFlowOptions(solver_type=gce.SolverType.NR, verbose=1, tolerance=1e-8, control_q=ReactivePowerControlMode.Direct)#, max_iter=100)
    opf_options = gce.OptimalPowerFlowOptions(solver=gce.SolverType.NR, verbose=0, ips_tolerance=1e-4, ips_iterations=200)

#    d_opf_results = ac_optimal_power_flow(Pref=np.array(d_pf_original['pf_gen']['P']), slack_bus_num=i_slack, nc=nc, pf_options=pf_options, plot_error=True)

    start = time.perf_counter()

    pf_results = multi_island_pf_nc(nc=nc, options=pf_options)

    d_opf_results = ac_optimal_power_flow(nc= nc,
                                          pf_options= pf_options,
                                          opf_options= opf_options,
                                          # debug: bool = False,
                                          #use_autodiff = True,
                                          pf_init= True,
                                          Sbus_pf= pf_results.Sbus,
                                          voltage_pf= pf_results.voltage,
                                          plot_error= False)


    end = time.perf_counter()
    computing_times['time_powerflow'] = end - start

    d_opf = process_optimal_power_flow.update_OP(gridCal_grid, d_opf_results, d_raw_data)
    d_opf['info']=pd.DataFrame()
    d_opf = additional_info_OPF_results(d_opf,i_slack, n_pf, d_opf_results)

    if not d_opf_results.converged:
        # Exit function
        stability = -1
        output_dataframes = postprocess_obj_func(
            output_dataframes, case_id, stability,
            df_computing_times=computing_times)
        return stability, output_dataframes

    #########################################################################


    d_grid, d_opf = fill_d_grid_after_powerflow.fill_d_grid(d_grid,
                                                           gridCal_grid, d_opf,
                                                           d_raw_data, d_op)

    p_sg = np.sum(d_grid['T_gen'].query('element == "SG"')['P']) * 100
    p_cig = np.sum(d_grid['T_gen'].query('element != "SG"')['P']) * 100
    if p_cig!=0:
        perc_gfor = np.sum(d_grid['T_gen'].query('element == "GFOR"')['P']) / p_cig*100
    else:
        perc_gfor=0
        
    if dimensions:
        valid_point = True
        for d in dimensions:
            if d.label == "p_sg":
                if p_sg < d.borders[0] or p_sg > d.borders[1]:
                    valid_point = False
            if d.label == "p_cig":
                if p_cig < d.borders[0] or p_cig > d.borders[1]:
                    valid_point = False
            if d.label == "perc_g_for":
                if perc_gfor < d.borders[0] or perc_gfor > d.borders[1]:
                    valid_point = False
        if not valid_point:
            # Exit function
            stability = -2
            output_dataframes = postprocess_obj_func(
                output_dataframes,case_id, stability,
                df_computing_times=computing_times)
            return stability, output_dataframes

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
    start = time.perf_counter()

    """
    connect_fun: 'append_and_connect' (default) or 'interconnect'. 
        'append_and_connect': Uses a function that bypasses linearization; 
        'interconnect': use original ct.interconnect function. 
    save_ss_matrices: bool. Default is False. 
        If True, write on csv file the A, B, C, D matrices of the state space.
        False default option
    """
    connect_fun = 'append_and_connect'
    save_ss_matrices = False

    l_blocks, l_states, d_grid = generate_NET.generate_SS_NET_blocks(
        d_grid, delta_slk, connect_fun, save_ss_matrices)

    end = time.perf_counter()
    computing_times['time_generate_SS_net'] = end - start

    start = time.perf_counter()

    # Generate generator units State-Space Model
    l_blocks, l_states = generate_elements.generate_SS_elements(
        d_grid, delta_slk, l_blocks, l_states, connect_fun, save_ss_matrices)
    end = time.perf_counter()
    computing_times['time_generate_SS_elem'] = end - start

    # %% BUILD FULL SYSTEM STATE-SPACE MODEL

    # Define full system inputs and ouputs
    var_in = ['NET_Rld1']
    var_out = ['all'] #['all']  # ['GFOR3_w'] #

    # Build full system state-space model
    start = time.perf_counter()

    inputs, outputs = build_ss.select_io(l_blocks, var_in, var_out)
    ss_sys = build_ss.connect(l_blocks, l_states, inputs, outputs, connect_fun,
                              save_ss_matrices)

    end = time.perf_counter()
    computing_times['time_connect'] = end - start


    # %% SMALL-SIGNAL ANALYSIS

    start = time.perf_counter()


    T_EIG = small_signal.FEIG(ss_sys, False)
    T_EIG.head

    end = time.perf_counter()
    computing_times['time_eig'] = end - start


    # write to excel
    # T_EIG.to_excel(path.join(path_results, "EIG_" + excel + ".xlsx"))

    if max(T_EIG['real'] >= 0):
        stability = 0
    else:
        stability = 1

    # Obtain all participation factors
    # df_PF = small_signal.FMODAL(ss_sys, plot=False)
    # # Obtain the participation factors for the selected modes
    # T_modal, df_PF = small_signal.FMODAL_REDUCED(ss_sys, plot=True, modeID = [1,3,11])
    # # Obtain the participation factors >= tol, for the selected modes
    start = time.perf_counter()

    T_modal, df_PF = small_signal.FMODAL_REDUCED_tol(ss_sys, plot=False, modeID = np.arange(1,23), tol = 0.3)

    end = time.perf_counter()
    computing_times['time_partfact'] = end - start

    # Collect output dataframes
    df_op, df_real, df_imag, df_freq, df_damp = (
        get_case_results(T_EIG=T_EIG, d_grid=d_grid))
    output_dataframes['df_op'] = df_op
    output_dataframes['df_real'] = df_real
    output_dataframes['df_imag'] = df_imag
    output_dataframes['df_freq'] = df_freq
    output_dataframes['df_damp'] = df_damp
    output_dataframes['df_computing_times'] = computing_times
    # Do not include objects that are not dataframes and are not single-row
    # output_dataframes['d_grid'] = d_grid
    # output_dataframes['d_opf'] = d_opf
    # output_dataframes['d_pf_original'] = d_pf_original
    
    # Exit function
    output_dataframes = postprocess_obj_func(output_dataframes, case_id,
                                             stability)
    return stability, output_dataframes


def postprocess_obj_func(output_dataframes, case_id, stability,
                         **update_output_dataframes):
    """
    Do tasks that always need to be performed before exiting the objective
    function.

    You can pass an arbitrary number of key-value arguments as a utility to
    update output_dataframes with new dataframes, for instance:

    >> output_dataframes = \
    >>     postprocess_obj_func(output_dataframes, case_id, stability,
    >>         df_op=df_op, df_computing_times=computing_times,
    >>         df_real=df_real, df_imag=df_imag)

    """
    # Update output dataframes
    for df_name, updated_df in update_output_dataframes.items():
        output_dataframes[df_name] = updated_df

    # Apply operations to extra dataframes
    for df_name, df in output_dataframes.items():
        # Append unique_id
        df['case_id'] = case_id
        # Append stability result
        df['Stability'] = stability

    # Check that the keys of df_names and output_dataframes match
    if set(OUTPUT_DF_NAMES) != set(output_dataframes.keys()):
        raise ValueError(
            'The keys of "output_dataframes" do not match the expected keys.')

    return output_dataframes


def return_d_opf(d_raw_data, d_opf_results):
    df_opf_bus = pd.DataFrame(
        {'bus': d_raw_data['results_bus']['I'], 'Vm': d_opf_results.Vm, 'theta': d_opf_results.Va})
    df_opf_gen_pre = pd.DataFrame(
        {'bus': d_raw_data['generator']['I'], 'P': d_opf_results.Pg, 'Q': d_opf_results.Qg})
    df_opf_gen = pd.merge(df_opf_gen_pre, df_opf_bus[['bus', 'Vm', 'theta']], on='bus', how='left')
    d_opf = {'opf_bus': df_opf_bus, 'opf_gen': df_opf_gen,'opf_load': d_raw_data['load']} 
    return d_opf


def write_csv(d, path, n_pf, filename_start):
    for df_name, df in d.items():
        filename=filename_start+'_'+str(n_pf)+'_'+str(df_name)
        pd.DataFrame.to_csv(df,path+filename+".csv")
        
def write_xlsx(d, path, filename):
    with pd.ExcelWriter(path+filename+".xlsx") as writer:
        for df_name, df in d.items():
            if isinstance(df, pd.DataFrame):                
                df.to_excel(writer, sheet_name=df_name)


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