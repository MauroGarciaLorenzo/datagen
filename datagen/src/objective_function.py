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

import sys
sys.path.append('../../../GridCal/src/')
from GridCalEngine.Simulations.PowerFlow.power_flow_options import ReactivePowerControlMode, SolverType

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

# @task(return=3)
def feasible_power_flow(case, **kwargs):
    d_raw_data = kwargs.get("d_raw_data", None)
    d_op = kwargs.get("d_op", None)
    GridCal_grid = kwargs.get("GridCal_grid", None)
    d_grid = kwargs.get("d_grid", None)
    d_sg = kwargs.get("d_sg", None)
    d_vsc = kwargs.get("d_vsc", None)
    voltage_profile=kwargs.get("voltage_profile",None)
    v_min_v_max_delta_v=kwargs.get("v_min_v_max_delta_v",None)
    V_set=kwargs.get("V_set",None)
    
    if voltage_profile!=None and v_min_v_max_delta_v==None:
        print('Error: Voltage profile option selected but v_min, v_max, and delta_v are missing')        
        return None, None, None

    if voltage_profile!=None and V_set!=None:
        print('Error: Both Voltage profile and V_set option is selected. Choose only one of them')        
        return None, None, None
    
    if voltage_profile==None and V_set==None:
        print('Error: Neither Voltage profile or V_set option is selected. Choose one of them')        
        return None, None, None

    d_raw_data, d_op = datagen_OP.generated_operating_point(case, d_raw_data,
                                                            d_op) 
    
    d_raw_data, slack_bus_num = choose_slack_bus(d_raw_data)
    # slack_bus_num=80
    assign_SlackBus_to_grid.assign_slack_bus(GridCal_grid, slack_bus_num)
    
    if voltage_profile!=None:
        vmin=v_min_v_max_delta_v[0]
        vmax=v_min_v_max_delta_v[1]
        delta_v=v_min_v_max_delta_v[2]
        
        voltage_profile_list, indx_id=sampling.gen_voltage_profile(vmin,vmax,delta_v,d_raw_data,slack_bus_num,GridCal_grid)
    
        assign_Generators_to_grid.assign_PVGen(GridCal_grid=GridCal_grid, d_raw_data=d_raw_data, d_op=d_op, voltage_profile_list=voltage_profile_list, indx_id=indx_id)
    
    elif V_set!=None:
        assign_Generators_to_grid.assign_PVGen(GridCal_grid=GridCal_grid, d_raw_data=d_raw_data, d_op=d_op, V_set=V_set)
        
    assign_PQ_Loads_to_grid.assign_PQ_load(GridCal_grid, d_raw_data)

    # %% Run 1st POWER-FLOW

    # Receive system status from OPAL
    # d_grid, GridCal_grid, data_old = process_opal.update_OP_from_RT(d_grid, GridCal_grid, data_old)

    # Get Power-Flow results with GridCal
    pf_results = GridCal_powerflow.run_powerflow(GridCal_grid)

    print('Converged:', pf_results.convergence_reports[0].converged_[0])
    

    # Update PF results and operation point of generator elements
    d_pf = process_powerflow.update_OP(GridCal_grid, pf_results, d_raw_data)
    
    d_pf_original=copy.deepcopy(d_pf)
    
    if pf_results.convergence_reports[0].converged_[0] == False:
        print('No Convergent solution')
        return None, None, None
    
    d_raw_data=update_raw_data(d_raw_data,d_pf)
    # check violations
    Bus_voltage_violation, Gen_Q_Min_limit_violation, Gen_Q_Max_limit_violation, Gen_pf_violation, Gen_V_violation, Gen_pf_above095= check_feasibility.check_violations(d_pf, d_op)

    sum_diff_0= sum_cosphi_underdev(Gen_pf_violation)

    if voltage_profile!=None:

        J_f_pos=calculate_grad(sum_diff_0=sum_diff_0, Gen_pf_violation_0=Gen_pf_violation, d_raw_data=d_raw_data,
                           d_op=d_op, delta_p=10, GridCal_grid=GridCal_grid,
                           voltage_profile_list=voltage_profile_list,indx_id=indx_id)
    elif V_set!=None:

        J_f_pos=calculate_grad(sum_diff_0=sum_diff_0, Gen_pf_violation_0=Gen_pf_violation, d_raw_data=d_raw_data,
                           d_op=d_op, delta_p=10, GridCal_grid=GridCal_grid,
                           V_set=V_set)

    Gen_pf_violation['J_f']=J_f_pos.reshape(-1,1)

    # if len(Gen_pf_violation)==0:
    #     break

#%% IF UNFEASIBLE
    
    num_gen_viol=len(Gen_pf_violation)
    d_raw_data_gen_prev_iter=d_raw_data['generator'].copy(deep=True)
    Gen_pf_violation_prev_iter=Gen_pf_violation.copy(deep=True)
    
    print('min cosphi',min(Gen_pf_violation['Val']))
    print('P slack', d_pf['pf_gen'].query('bus == @slack_bus_num')['P'])
    n_iter=0
    molt=1
    alpha_incr=0.1
    alpha_decr=0.1

    while len(Gen_pf_violation)>0 and n_iter<100:
        print(n_iter)
        
        more_than_one_gen_violating=len(Gen_pf_violation)>1 
        all_gen_violatin_at_1_pu=np.all(np.round(Gen_pf_violation.query('GenBus != @slack_bus_num')['P_p_u'],4)==1)
        increasing_gen_power_positive_effect=len(Gen_pf_violation.query('GenBus != @slack_bus_num and J_f >0 and P_p_u <0.9999'))!=0
        slack_injecting_positive_power=d_pf['pf_gen'].loc[d_pf['pf_gen'].query('bus == @slack_bus_num').index[0],'P']>0
        
        if slack_injecting_positive_power == False:      
            
            d_raw_data['generator'] = d_raw_data_gen_prev_iter
            
            if voltage_profile!=None:
            
                assign_Generators_to_grid.assign_PVGen(GridCal_grid=GridCal_grid, d_raw_data=d_raw_data, d_op=d_op, voltage_profile_list=voltage_profile_list, indx_id=indx_id)
            
            elif V_set!=None:
                assign_Generators_to_grid.assign_PVGen(GridCal_grid=GridCal_grid, d_raw_data=d_raw_data, d_op=d_op, V_set=V_set)
              
            # Get Power-Flow results with GridCal
            pf_results = GridCal_powerflow.run_powerflow(GridCal_grid,SolverType.IWAMOTO,ReactivePowerControlMode.Iterative)
         
            print('Converged:',pf_results.convergence_reports[0].converged_[0])
            
            if pf_results.convergence_reports[0].converged_[0] == False:
            
                pf_results = GridCal_powerflow.run_powerflow(GridCal_grid)#,SolverType.IWAMOTO,ReactivePowerControlMode.Iterative)
            
                print('Converged:',pf_results.convergence_reports[0].converged_[0])       
                
                if pf_results.convergence_reports[0].converged_[0] == False:
                    print('No Convergent solution')
                    return None, None, None
    
         
        # Update PF results and operation point of generator elements
            d_pf = process_powerflow.update_OP(GridCal_grid, pf_results, d_raw_data)
            
            d_raw_data=update_raw_data(d_raw_data,d_pf)

            Bus_voltage_violation, Gen_Q_Min_limit_violation, Gen_Q_Max_limit_violation, Gen_pf_violation, Gen_V_violation, Gen_pf_above095 = check_feasibility.check_violations(d_pf, d_op)
         
            try:
                print('min cosphi',min(Gen_pf_violation['Val']))
                print('P slack', d_pf['pf_gen'].query('bus == @slack_bus_num')['P'])
            except: 
                print('P slack', d_pf['pf_gen'].query('bus == @slack_bus_num')['P'])
       
        
        if more_than_one_gen_violating and all_gen_violatin_at_1_pu==False and increasing_gen_power_positive_effect and slack_injecting_positive_power :
                        
            d_raw_data, d_raw_data_gen_prev_iter,num_gen_viol=proportional_increase_P(alpha_incr,d_pf,d_raw_data, d_op, Gen_pf_violation, slack_bus_num,num_gen_viol,d_raw_data_gen_prev_iter)   
        
            if voltage_profile!=None:
            
                assign_Generators_to_grid.assign_PVGen(GridCal_grid=GridCal_grid, d_raw_data=d_raw_data, d_op=d_op, voltage_profile_list=voltage_profile_list, indx_id=indx_id)
            
            elif V_set!=None:
                assign_Generators_to_grid.assign_PVGen(GridCal_grid=GridCal_grid, d_raw_data=d_raw_data, d_op=d_op, V_set=V_set)
                                    
            # Get Power-Flow results with GridCal
            pf_results = GridCal_powerflow.run_powerflow(GridCal_grid,SolverType.IWAMOTO,ReactivePowerControlMode.Iterative)
        
            print('Converged:',pf_results.convergence_reports[0].converged_[0])
            
            # if pf_results.convergence_reports[0].converged_[0] == False:
            #     return None, None, None
            
            if pf_results.convergence_reports[0].converged_[0] == False:
            
                pf_results = GridCal_powerflow.run_powerflow(GridCal_grid)#,SolverType.IWAMOTO,ReactivePowerControlMode.Iterative)
            
                print('Converged:',pf_results.convergence_reports[0].converged_[0])
                
                if pf_results.convergence_reports[0].converged_[0] == False:
                    print('No Convergent solution')
                    return None, None, None
            
            # d_pf_prev=copy.deepcopy(d_pf)
            # Update PF results and operation point of generator elements
            d_pf = process_powerflow.update_OP(GridCal_grid, pf_results, d_raw_data)
            
            d_raw_data=update_raw_data(d_raw_data,d_pf)

            Bus_voltage_violation, Gen_Q_Min_limit_violation, Gen_Q_Max_limit_violation, Gen_pf_violation, Gen_V_violation, Gen_pf_above095 = check_feasibility.check_violations(d_pf, d_op)
            
            if len(Gen_pf_violation)==0:
                return d_pf_original, d_pf, d_raw_data 

                
            sum_diff_1= sum_cosphi_underdev(Gen_pf_violation)

            c1=1
            rho=1.1
            expected_decrease = c1 * alpha_incr * np.dot(J_f_pos, -J_f_pos.reshape(-1,1))

            if sum_diff_1 <=sum_diff_0+expected_decrease:
                alpha_incr=alpha_incr
            else:
                alpha_incr=alpha_incr*rho
    
            if voltage_profile!=None:
            
                J_f_pos=calculate_grad(sum_diff_0=sum_diff_1, Gen_pf_violation_0=Gen_pf_violation, d_raw_data=d_raw_data,
                                   d_op=d_op, delta_p=10, GridCal_grid=GridCal_grid,
                                   voltage_profile_list=voltage_profile_list,indx_id=indx_id)
            elif V_set!=None:
            
                J_f_pos=calculate_grad(sum_diff_0=sum_diff_1, Gen_pf_violation_0=Gen_pf_violation, d_raw_data=d_raw_data,
                                   d_op=d_op, delta_p=10, GridCal_grid=GridCal_grid,
                                   V_set=V_set)
            Gen_pf_violation['J_f']=J_f_pos.reshape(-1,1)
            sum_diff_0=sum_diff_1
            try:
                print('min cosphi',min(Gen_pf_violation['Val']))
                print('P slack', d_pf['pf_gen'].query('bus == @slack_bus_num')['P'])
            except: 
                print('P slack', d_pf['pf_gen'].query('bus == @slack_bus_num')['P'])
                
        if increasing_gen_power_positive_effect==False: #and all_gen_violatin_at_1_pu==False more_than_one_gen_violating and
                        
            max_cosphi=0.95#max(Gen_pf_above095['Val'])
            Gen_pf_above095_red=Gen_pf_above095.query('Val > @max_cosphi').reset_index(drop=True)
            # Gen_pf_above095_red=Gen_pf_above095.sort_values(by='Val',ascending=False).reset_index(drop=True)
            
            if voltage_profile!=None:

                J_f_neg=calculate_grad(sum_diff_0=sum_diff_0, Gen_pf_violation_0=Gen_pf_above095_red, d_raw_data=d_raw_data,
                                   d_op=d_op, delta_p=-10, GridCal_grid=GridCal_grid,
                                   voltage_profile_list=voltage_profile_list,indx_id=indx_id)
            elif V_set!=None:

                J_f_neg=calculate_grad(sum_diff_0=sum_diff_0, Gen_pf_violation_0=Gen_pf_above095_red, d_raw_data=d_raw_data,
                                   d_op=d_op, delta_p=-10, GridCal_grid=GridCal_grid,
                                   V_set=V_set)

                
            Gen_pf_above095_red['J_f']=J_f_neg.reshape(-1,1)
            exclude_gen=list(Gen_pf_above095_red.query('J_f <=0')['GenBus'])
            
            # Gen_high_cosphi=d_pf['pf_gen'].query('cosphi >= @max_cosphi and bus!=@exclude_gen').reset_index(drop=True)
            
            Gen_high_cosphi=d_pf['pf_gen'].query('cosphi >= @max_cosphi and bus!=@exclude_gen').sort_values(by='cosphi',ascending=False).reset_index(drop=True)
            # try:
            #     Gen_high_cosphi=Gen_high_cosphi.iloc[0:5]
            # except:
            #     Gen_high_cosphi=Gen_high_cosphi
            
            
            d_raw_data= proportional_deincrease_P(d_pf,d_raw_data,d_op,Gen_high_cosphi,alpha_decr)
            if voltage_profile!=None:
            
                assign_Generators_to_grid.assign_PVGen(GridCal_grid=GridCal_grid, d_raw_data=d_raw_data, d_op=d_op, voltage_profile_list=voltage_profile_list, indx_id=indx_id)
            
            elif V_set!=None:
                assign_Generators_to_grid.assign_PVGen(GridCal_grid=GridCal_grid, d_raw_data=d_raw_data, d_op=d_op, V_set=V_set)
            
            
            # Get Power-Flow results with GridCal
            pf_results = GridCal_powerflow.run_powerflow(GridCal_grid,SolverType.IWAMOTO,ReactivePowerControlMode.Iterative)
        
            print('Converged:',pf_results.convergence_reports[0].converged_[0])
            
            # if pf_results.convergence_reports[0].converged_[0] == False:
            #     return None, None, None
            
            if pf_results.convergence_reports[0].converged_[0] == False:
            
                pf_results = GridCal_powerflow.run_powerflow(GridCal_grid)#,SolverType.IWAMOTO,ReactivePowerControlMode.Iterative)
            
                print('Converged:',pf_results.convergence_reports[0].converged_[0])
                
                if pf_results.convergence_reports[0].converged_[0] == False:
                    print('No Convergent solution')
                    return None, None, None
            
            d_pf_prev=copy.deepcopy(d_pf)
            # Update PF results and operation point of generator elements
            d_pf = process_powerflow.update_OP(GridCal_grid, pf_results, d_raw_data)
            
            d_raw_data=update_raw_data(d_raw_data,d_pf)

            Bus_voltage_violation, Gen_Q_Min_limit_violation, Gen_Q_Max_limit_violation, Gen_pf_violation, Gen_V_violation, Gen_pf_above095 = check_feasibility.check_violations(d_pf, d_op)
        
            if len(Gen_pf_violation)==0:
                return d_pf_original, d_pf, d_raw_data 

            sum_diff_1= sum_cosphi_underdev(Gen_pf_violation)
            
            # c1=1
            # rho=1.1
            # expected_decrease = c1 * alpha_decr * np.dot(J_f_neg, -J_f_neg.reshape(-1,1))

            # if sum_diff_1 <=sum_diff_0+expected_decrease:
            #     alpha_decr=alpha_decr
            # else:
            #     alpha_decr=alpha_decr*rho
            
            # sum_diff_1=sum_diff_0
            
            if voltage_profile!=None:

                J_f_pos=calculate_grad(sum_diff_0=sum_diff_1, Gen_pf_violation_0=Gen_pf_violation, d_raw_data=d_raw_data,
                                   d_op=d_op, delta_p=10, GridCal_grid=GridCal_grid,
                                   voltage_profile_list=voltage_profile_list,indx_id=indx_id)
            elif V_set!=None:
        
                J_f_pos=calculate_grad(sum_diff_0=sum_diff_1, Gen_pf_violation_0=Gen_pf_violation, d_raw_data=d_raw_data,
                                   d_op=d_op, delta_p=10, GridCal_grid=GridCal_grid,
                                   V_set=V_set)

            

            Gen_pf_violation['J_f']=J_f_pos.reshape(-1,1)
        
            try:
                print('min cosphi',min(Gen_pf_violation['Val']))
                print('P slack', d_pf['pf_gen'].query('bus == @slack_bus_num')['P'])
            except: 
                print('P slack', d_pf['pf_gen'].query('bus == @slack_bus_num')['P'])
                
        
        print('sum_diff ',sum_diff_1)
        print('gen viol ', len(Gen_pf_violation))
              
        
        n_iter=n_iter+1
        if n_iter == 4:#10*molt:
            print('stop')
            molt=molt+1
    
        # if more_than_one_gen_violating==False and all_gen_violatin_at_1_pu:
        #     print('stop')
            
            
        # if len(Gen_pf_violation)>1 and d_pf['pf_gen'].loc[d_pf['pf_gen'].query('bus == @slack_bus_num').index[0],'P']<0:
            
        #     d_raw_data['generator'] = d_raw_data_gen_prev_iter
            
        #     assign_Generators_to_grid.assign_PVGen(GridCal_grid, d_raw_data, d_op)
             
        #     # Get Power-Flow results with GridCal
        #     pf_results = GridCal_powerflow.run_powerflow(GridCal_grid,SolverType.IWAMOTO,ReactivePowerControlMode.Iterative)
         
        #     print('Converged:',pf_results.convergence_reports[0].converged_[0])
            
        #     if pf_results.convergence_reports[0].converged_[0] == False:
            
        #         pf_results = GridCal_powerflow.run_powerflow(GridCal_grid)#,SolverType.IWAMOTO,ReactivePowerControlMode.Iterative)
            
        #         print('Converged:',pf_results.convergence_reports[0].converged_[0])       
                
        #         if pf_results.convergence_reports[0].converged_[0] == False:
        #             print('No Convergent solution')
        #             return None, None, None
         
        #     # Update PF results and operation point of generator elements
        #     d_pf = process_powerflow.update_OP(GridCal_grid, pf_results, d_raw_data)
            
        #     d_raw_data=update_raw_data(d_raw_data,d_pf)

        #     Bus_voltage_violation, Gen_Q_Min_limit_violation, Gen_Q_Max_limit_violation, Gen_pf_violation, Gen_V_violation, Gen_pf_above095 = check_feasibility.check_violations(d_pf, d_op)
         
        #     try:
        #         print('min cosphi',min(Gen_pf_violation['Val']))
        #         print('P slack', d_pf['pf_gen'].query('bus == @slack_bus_num')['P'])
        #     except: 
        #         print('P slack', d_pf['pf_gen'].query('bus == @slack_bus_num')['P'])
             
        #     max_cosphi=max(d_pf['pf_gen']['cosphi'])*0.99
        #     Gen_high_cosphi=d_pf['pf_gen'].query('cosphi >= @max_cosphi').reset_index(drop=True)
            
        #     d_raw_data, d_raw_data_gen_prev_iter = proportional_deincrease_P(d_pf,d_raw_data,d_op,Gen_high_cosphi)
        #     assign_Generators_to_grid.assign_PVGen(GridCal_grid, d_raw_data, d_op)
             
            
        #     # Get Power-Flow results with GridCal
        #     pf_results = GridCal_powerflow.run_powerflow(GridCal_grid,SolverType.IWAMOTO,ReactivePowerControlMode.Iterative)
            
        #     print('Converged:',pf_results.convergence_reports[0].converged_[0])
            
        #     if pf_results.convergence_reports[0].converged_[0] == False:
            
        #         pf_results = GridCal_powerflow.run_powerflow(GridCal_grid)#,SolverType.IWAMOTO,ReactivePowerControlMode.Iterative)
            
        #         print('Converged:',pf_results.convergence_reports[0].converged_[0])
             
        #         if pf_results.convergence_reports[0].converged_[0] == False:
        #             print('No Convergent solution')
        #             return None, None, None
            
        #     # Update PF results and operation point of generator elements
        #     d_pf = process_powerflow.update_OP(GridCal_grid, pf_results, d_raw_data)
            
        #     d_raw_data=update_raw_data(d_raw_data,d_pf)

        #     Bus_voltage_violation, Gen_Q_Min_limit_violation, Gen_Q_Max_limit_violation, Gen_pf_violation, Gen_V_violation, Gen_pf_above095 = check_feasibility.check_violations(d_pf, d_op)
            
        #     try:
        #         print('min cosphi',min(Gen_pf_violation['Val']))
        #         print('P slack', d_pf['pf_gen'].query('bus == @slack_bus_num')['P'])
        #     except: 
        #         print('P slack', d_pf['pf_gen'].query('bus == @slack_bus_num')['P'])
             
        # if len(Gen_pf_violation)==1 and Gen_pf_violation.loc[0,'GenBus']==slack_bus_num:    
        #     d_raw_data, d_raw_data_gen_prev_iter = deincrease_P_only_1Gen(d_pf,d_raw_data,d_op)
        #     assign_Generators_to_grid.assign_PVGen(GridCal_grid, d_raw_data, d_op)
             
            
        #     # Get Power-Flow results with GridCal
        #     pf_results = GridCal_powerflow.run_powerflow(GridCal_grid,SolverType.IWAMOTO,ReactivePowerControlMode.Iterative)
            
        #     print('Converged:',pf_results.convergence_reports[0].converged_[0])
            
        #     if pf_results.convergence_reports[0].converged_[0] == False:
            
        #         pf_results = GridCal_powerflow.run_powerflow(GridCal_grid)#,SolverType.IWAMOTO,ReactivePowerControlMode.Iterative)
            
        #         print('Converged:',pf_results.convergence_reports[0].converged_[0])
                
        #         if pf_results.convergence_reports[0].converged_[0] == False:
        #             print('No Convergent solution')
        #             return None, None, None
            
            
        #     # Update PF results and operation point of generator elements
        #     d_pf = process_powerflow.update_OP(GridCal_grid, pf_results, d_raw_data)
            
        #     d_raw_data=update_raw_data(d_raw_data,d_pf)

        #     Bus_voltage_violation, Gen_Q_Min_limit_violation, Gen_Q_Max_limit_violation, Gen_pf_violation, Gen_V_violation, Gen_pf_above095 = check_feasibility.check_violations(d_pf, d_op)
            
        #     try:
        #         print('min cosphi',min(Gen_pf_violation['Val']))
        #         print('P slack', d_pf['pf_gen'].query('bus == @slack_bus_num')['P'])
        #     except: 
        #         print('P slack', d_pf['pf_gen'].query('bus == @slack_bus_num')['P'])
             
        # if len(Gen_pf_violation)==1 and Gen_pf_violation.loc[0,'GenBus']!=slack_bus_num:    
        #     alpha=0.1
        #     d_raw_data, d_raw_data_gen_prev_iter,num_gen_viol=proportional_increase_P(alpha,d_pf,d_raw_data, d_op, Gen_pf_violation, slack_bus_num,num_gen_viol,d_raw_data_gen_prev_iter)   
        #     assign_Generators_to_grid.assign_PVGen(GridCal_grid, d_raw_data, d_op)
             
            
        #     # Get Power-Flow results with GridCal
        #     pf_results = GridCal_powerflow.run_powerflow(GridCal_grid,SolverType.IWAMOTO,ReactivePowerControlMode.Iterative)
            
        #     print('Converged:',pf_results.convergence_reports[0].converged_[0])
            
        #     if pf_results.convergence_reports[0].converged_[0] == False:
            
        #         pf_results = GridCal_powerflow.run_powerflow(GridCal_grid)#,SolverType.IWAMOTO,ReactivePowerControlMode.Iterative)
            
        #         print('Converged:',pf_results.convergence_reports[0].converged_[0])
                
        #         if pf_results.convergence_reports[0].converged_[0] == False:
        #             print('No Convergent solution')
        #             return None, None, None
            
            
        #     # Update PF results and operation point of generator elements
        #     d_pf = process_powerflow.update_OP(GridCal_grid, pf_results, d_raw_data)
            
        #     d_raw_data=update_raw_data(d_raw_data,d_pf)

        #     Bus_voltage_violation, Gen_Q_Min_limit_violation, Gen_Q_Max_limit_violation, Gen_pf_violation, Gen_V_violation, Gen_pf_above095 = check_feasibility.check_violations(d_pf, d_op)
            
        #     try:
        #         print('min cosphi',min(Gen_pf_violation['Val']))
        #         print('P slack', d_pf['pf_gen'].query('bus == @slack_bus_num')['P'])
        #     except: 
        #         print('P slack', d_pf['pf_gen'].query('bus == @slack_bus_num')['P'])
             

        # if len(Gen_pf_violation)>1 and np.all(np.round(Gen_pf_violation.query('GenBus != @slack_bus_num')['P_p_u'],4)==1):
        #     # d_raw_data, d_raw_data_gen_prev_iter = deincrease_P_only_1Gen(d_pf,d_raw_data,d_op)
            
        #     max_cosphi=max(d_pf['pf_gen']['cosphi'])*0.99
        #     Gen_high_cosphi=d_pf['pf_gen'].query('cosphi >= @max_cosphi').reset_index(drop=True)
            
        #     d_raw_data, d_raw_data_gen_prev_iter = proportional_deincrease_P(d_pf,d_raw_data,d_op,Gen_high_cosphi)
          
        #     assign_Generators_to_grid.assign_PVGen(GridCal_grid, d_raw_data, d_op)
             
            
        #     # Get Power-Flow results with GridCal
        #     pf_results = GridCal_powerflow.run_powerflow(GridCal_grid,SolverType.IWAMOTO,ReactivePowerControlMode.Iterative)
            
        #     print('Converged:',pf_results.convergence_reports[0].converged_[0])
            
        #     if pf_results.convergence_reports[0].converged_[0] == False:
            
        #         pf_results = GridCal_powerflow.run_powerflow(GridCal_grid)#,SolverType.IWAMOTO,ReactivePowerControlMode.Iterative)
            
        #         print('Converged:',pf_results.convergence_reports[0].converged_[0])
                
        #         if pf_results.convergence_reports[0].converged_[0] == False:
        #             print('No Convergent solution')
        #             return None, None, None
            
            
        #     # Update PF results and operation point of generator elements
        #     d_pf = process_powerflow.update_OP(GridCal_grid, pf_results, d_raw_data)
            
        #     d_raw_data=update_raw_data(d_raw_data,d_pf)

        #     Bus_voltage_violation, Gen_Q_Min_limit_violation, Gen_Q_Max_limit_violation, Gen_pf_violation, Gen_V_violation, Gen_pf_above095 = check_feasibility.check_violations(d_pf, d_op)
            
        #     try:
        #         print('min cosphi',min(Gen_pf_violation['Val']))
        #         print('P slack', d_pf['pf_gen'].query('bus == @slack_bus_num')['P'])
        #     except: 
        #         print('P slack', d_pf['pf_gen'].query('bus == @slack_bus_num')['P'])

        
    # d_raw_data=update_raw_data(d_pf)    
    return d_pf_original, d_pf, d_raw_data 

    # %% FILL d_grid

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

# def proportional_increase_P(d_pf,d_raw_data,d_op,Gen_pf_violation,slack_bus_num, num_gen_viol,d_raw_data_gen_prev_iter):
#     min_cosphi=min(Gen_pf_violation.query('GenBus != @slack_bus_num and P_p_u!=1')['Val'])
#     max_delta_cosphi=0.95-min_cosphi
    
#     if len(Gen_pf_violation)<=num_gen_viol:
        
#         d_raw_data_gen_prev_iter=d_raw_data['generator'].copy(deep=True)
#         num_gen_viol=len(Gen_pf_violation)

#     d_raw_data_gen_prev_iter=d_raw_data['generator'].copy(deep=True)
#     # num_gen_viol=len(Gen_pf_violation)
#     P_sum=0
#     for i in range(len(Gen_pf_violation)):
#         cosphi=Gen_pf_violation.loc[i,'Val']
#         delta_cosphi=0.95-cosphi
#         delta_P=0.1/max_delta_cosphi*delta_cosphi
        
#         bus=Gen_pf_violation.loc[i,'GenBus']
#         P_pu= d_pf['pf_gen'].loc[d_pf['pf_gen'].query('bus==@bus').index[0],'P']*100/d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus').index[0],'Pmax']

#         if P_pu+delta_P<1:
#             P_pu=P_pu+delta_P
#         else:
#             P_pu=1
#         d_raw_data['generator'].loc[d_raw_data['generator'].query('I ==@bus').index[0],'PG']=P_pu*d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus').index[0],'Pmax']

#         #P_sum=P_sum+print(float(P_pu*d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus').index[0],'Pmax']))
#     return d_raw_data, d_raw_data_gen_prev_iter,num_gen_viol
    

def proportional_deincrease_P(d_pf,d_raw_data,d_op,Gen_high_cosphi,alpha):
    max_cosphi=max(Gen_high_cosphi['cosphi'])
    max_delta_cosphi=max_cosphi- 0.95
    # d_raw_data_gen_prev_iter=d_raw_data['generator'].copy(deep=True)
    
    for i in range(len(Gen_high_cosphi)):
        cosphi=Gen_high_cosphi.loc[i,'cosphi']
        delta_cosphi=cosphi-0.95
        delta_P=alpha/max_delta_cosphi*delta_cosphi
        
        
        bus=Gen_high_cosphi.loc[i,'bus']
        P_pu= d_pf['pf_gen'].loc[d_pf['pf_gen'].query('bus==@bus').index[0],'P']*100/d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus').index[0],'Pmax']

        if P_pu-delta_P>0:
            P_pu=P_pu-delta_P
        else:
            P_pu=0
        d_raw_data['generator'].loc[d_raw_data['generator'].query('I ==@bus').index[0],'PG']=P_pu*d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus').index[0],'Pmax']

    return d_raw_data#, d_raw_data_gen_prev_iter
       
   
def deincrease_P_only_1Gen(d_pf,d_raw_data,d_op):
    max_cosphi=max(d_pf['pf_gen']['cosphi'])
    i_max=np.argmax(d_pf['pf_gen']['cosphi'])
    max_delta_cosphi=max_cosphi- 0.95
    d_raw_data_gen_prev_iter=d_raw_data['generator'].copy(deep=True)
    
    delta_P=0.05
        
    bus=d_pf['pf_gen'].loc[i_max,'bus']
    P_pu= d_pf['pf_gen'].loc[d_pf['pf_gen'].query('bus==@bus').index[0],'P']*100/d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus').index[0],'Pmax']

    if P_pu-delta_P>0:
        P_pu=P_pu-delta_P
    else:
        P_pu=0
    d_raw_data['generator'].loc[d_raw_data['generator'].query('I ==@bus').index[0],'PG']=P_pu*d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus').index[0],'Pmax']

    return d_raw_data, d_raw_data_gen_prev_iter
   
def update_raw_data(d_raw_data,d_pf):
    
    d_raw_data['generator']['PG']=d_pf['pf_gen']['P']*100
    d_raw_data['generator']['QG']=d_pf['pf_gen']['Q']*100
    
    return d_raw_data

def proportional_increase_P(alpha,d_pf,d_raw_data,d_op,Gen_pf_violation,slack_bus_num, num_gen_viol,d_raw_data_gen_prev_iter):
    Gen_pf_violation_red=Gen_pf_violation.query('J_f>0').reset_index(drop=True)
    min_cosphi=min(Gen_pf_violation_red.drop(np.where(np.isclose(Gen_pf_violation_red['P_p_u'],1))[0],axis=0).query('GenBus != @slack_bus_num')['Val'])
    max_delta_cosphi=0.95-min_cosphi
    
    if len(Gen_pf_violation)<=num_gen_viol:
        
        d_raw_data_gen_prev_iter=d_raw_data['generator'].copy(deep=True)
        num_gen_viol=len(Gen_pf_violation)
        Gen_pf_violation_prev_iter=Gen_pf_violation.copy(deep=True)

    # d_raw_data_gen_prev_iter=d_raw_data['generator'].copy(deep=True)
    # num_gen_viol=len(Gen_pf_violation)
    #alpha=0.1
    delta_P_tot=0
    while delta_P_tot<10:
        delta_P_tot_prev=delta_P_tot
        for i in range(len(Gen_pf_violation_red)):
            cosphi=Gen_pf_violation_red.loc[i,'Val']
            delta_cosphi=0.95-cosphi
            delta_P=alpha/max_delta_cosphi*delta_cosphi
            
            bus=Gen_pf_violation_red.loc[i,'GenBus']
            if bus == slack_bus_num:
                continue
            P_pu= d_pf['pf_gen'].loc[d_pf['pf_gen'].query('bus==@bus').index[0],'P']*100/d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus').index[0],'Pmax']
    
            if P_pu+delta_P<1:
                P_pu=P_pu+delta_P
            else:
                P_pu=1
            d_raw_data['generator'].loc[d_raw_data['generator'].query('I ==@bus').index[0],'PG']=P_pu*d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus').index[0],'Pmax']

        buses=list(set(list(Gen_pf_violation_red['GenBus']))-set([slack_bus_num]))
        delta_P_tot=np.sum(d_raw_data['generator'].query('I == @buses')['PG'])-np.sum(d_pf['pf_gen'].query('bus == @buses')['P'])*100
        delta_delta=abs(delta_P_tot_prev-delta_P_tot)
        if delta_delta < 1e-3:
            return d_raw_data, d_raw_data_gen_prev_iter,num_gen_viol
        alpha=alpha+0.05
        print('delta_p_tot',delta_P_tot)
    return d_raw_data, d_raw_data_gen_prev_iter,num_gen_viol

def sum_cosphi_underdev(Gen_pf_violation):
    sum_diff=0.95*len(Gen_pf_violation)-np.sum(Gen_pf_violation['Val'])
    return sum_diff

def calculate_grad(sum_diff_0,Gen_pf_violation_0, d_raw_data, d_op, delta_p,GridCal_grid, **kwargs):
   
    voltage_profile_list=kwargs.get("voltage_profile_list",None)
    indx_id=kwargs.get("indx_id",None)
    V_set=kwargs.get("V_set",None)
   
    J_f=np.zeros([1,len(Gen_pf_violation_0)])
    
    for gen_count in range(0,len(Gen_pf_violation_0)): 
        d_raw_data_increase_gen=copy.deepcopy(d_raw_data)
        gen_bus=Gen_pf_violation_0.loc[gen_count,'GenBus']
        d_raw_data_increase_gen['generator'].loc[d_raw_data_increase_gen['generator'].query('I == @gen_bus').index[0],'PG']=d_raw_data_increase_gen['generator'].loc[d_raw_data_increase_gen['generator'].query('I == @gen_bus').index[0],'PG']+delta_p
        
        if voltage_profile_list!=None:
            assign_Generators_to_grid.assign_PVGen(GridCal_grid=GridCal_grid, d_raw_data=d_raw_data_increase_gen, d_op=d_op, voltage_profile_list=voltage_profile_list, indx_id=indx_id)
        elif V_set!=None:
            assign_Generators_to_grid.assign_PVGen(GridCal_grid=GridCal_grid, d_raw_data=d_raw_data_increase_gen, d_op=d_op, V_set=V_set)
                    
        # Get Power-Flow results with GridCal
        pf_results = GridCal_powerflow.run_powerflow(GridCal_grid,SolverType.IWAMOTO,ReactivePowerControlMode.Iterative)
    
        print('Converged:',pf_results.convergence_reports[0].converged_[0])
        
        # if pf_results.convergence_reports[0].converged_[0] == False:
        #     return None, None, None
        
        if pf_results.convergence_reports[0].converged_[0] == False:
        
            pf_results = GridCal_powerflow.run_powerflow(GridCal_grid)#,SolverType.IWAMOTO,ReactivePowerControlMode.Iterative)
        
            print('Converged:',pf_results.convergence_reports[0].converged_[0])
            
            if pf_results.convergence_reports[0].converged_[0] == False:
                print('No Convergent solution')
                return None, None, None
        
        # Update PF results and operation point of generator elements
        d_pf = process_powerflow.update_OP(GridCal_grid, pf_results, d_raw_data)
        
        Bus_voltage_violation, Gen_Q_Min_limit_violation, Gen_Q_Max_limit_violation, Gen_pf_violation, Gen_V_violation, Gen_pf_above095 = check_feasibility.check_violations(d_pf, d_op)
    
        J_f[0,gen_count]=sum_diff_0-sum_cosphi_underdev(Gen_pf_violation)
        
    return J_f