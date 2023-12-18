import math
import time
from StabilityAnalysis.operating_point_from_datagenerator import datagen_OP
from StabilityAnalysis.modify_GridCal_grid import assign_StaticGen_to_grid,assign_PQ_Loads_to_grid
from StabilityAnalysis.powerflow import GridCal_powerflow, process_powerflow, slack_bus, fill_d_grid_after_powerflow
from StabilityAnalysis.preprocess import preprocess_data, read_data, process_raw, parameters,read_op_data_excel, admittance_matrix

# file where objective function is declared (dummy test)
def dummy(case):
    time.sleep(10)
    return round(math.sin(sum(case)) * 0.5 + 0.5)


def dummy_linear(case):
    total_sum = sum(case)
    return total_sum/19 > 0.5  # 19 => maximum value among all upper borders

def small_signal_stability(case,d_raw_data, d_op, GridCal_grid, d_grid, d_sg, d_vsc):
    
    d_raw_data, d_op =datagen_OP.generated_operating_point(case,d_raw_data, d_op)
    
    assign_StaticGen_to_grid.assign_StaticGen(GridCal_grid, d_raw_data, d_op)
    assign_PQ_Loads_to_grid.assign_PQ_load(GridCal_grid, d_raw_data)

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

    # %% READ PARAMETERS

    # Get parameters of generator units from excel files & compute pu base
    d_grid = parameters.get_params(d_grid, d_sg, d_vsc)

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
    var_out = ['all']# ['GFOR3_w'] #
    
    # Build full system state-space model
    inputs, outputs = build_ss.select_io(l_blocks, var_in, var_out)
    ss_sys = build_ss.connect(l_blocks, l_states, inputs, outputs)
    
    # %% SMALL-SIGNAL ANALYSIS
    
    T_EIG = small_signal.FEIG(ss_sys, True)
    T_EIG.head
    
    # write to excel
    T_EIG.to_excel(path.join(path_results, "EIG_" + excel + ".xlsx"))
    
    if max(T_EIG['real']>=0):
        stability =0
    else:
        stability =1
        
    return stability, T_EIG, d_pf