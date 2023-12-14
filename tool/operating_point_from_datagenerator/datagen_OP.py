import pandas as pd
import numpy as np
import random

def generated_operating_point(case,d_raw_data, d_op): #GridCal_grid
    
    d_raw_data=assign_loads_to_d_raw_data(d_raw_data,case['p_load'])
        
    d_raw_data=assign_gens_to_d_raw_data(d_raw_data, case['p_sg'],case['p_cig'])
        
    d_raw_data, d_op = assign_GFOL_GFOR(d_raw_data, d_op, case['p_gfor'], case['p_gfol'], case['p_cig'])
    
    d_raw_data = alphas_P(d_raw_data)
    
    return d_raw_data, d_op


def assign_loads_to_d_raw_data(d_raw_data,case_load):
    
    case_load_index=case_load.index
    active_power=[p for p in case_load_index if p.startswith('p')]
    reactive_power=[q for q in case_load_index if q.startswith('q')]
    
    d_raw_data['load']['PL']= np.array(case_load[active_power])
    d_raw_data['load']['QL']= np.array(case_load[reactive_power])
    
    d_raw_data['load']['P']=d_raw_data['load']['PL']/100
    d_raw_data['load']['Q']=d_raw_data['load']['QL']/100
   
    return d_raw_data

def assign_gens_to_d_raw_data(d_raw_data,case_sg,case_cig):
    case_sg_index=case_sg.index
    case_cig_index=case_cig.index
    
    active_power_sg=[p for p in case_sg_index if p.startswith('p')]
    reactive_power_sg=[q for q in case_sg_index if q.startswith('q')]
    
    active_power_cig=[p for p in case_cig_index if p.startswith('p')]
    reactive_power_cig=[q for q in case_cig_index if q.startswith('q')]
   
    d_raw_data['generator']['PG']=np.array(case_sg[active_power_sg])+np.array(case_cig[active_power_cig])
    d_raw_data['generator']['QG']=np.array(case_sg[reactive_power_sg])+np.array(case_cig[reactive_power_cig])
    
    d_raw_data['generator']['P_CIG']=np.array(case_cig[active_power_cig])
    d_raw_data['generator']['P_SG']=np.array(case_sg[active_power_sg])
    
    return d_raw_data


def assign_GFOL_GFOR(d_raw_data, d_op, case_gfor, case_gfol, case_cig): #GridCal_grid
                    
    # buses=GridCal_grid.get_buses()
    # for bus in buses:
    #     if bus.determine_bus_type()._value_ == 3 :
            
    #         slack_bus_name=bus._name
    #         break
    
    case_gfor_index=case_gfor.index
    case_gfol_index=case_gfol.index
    case_cig_index=case_cig.index

    active_power_gfor=[p for p in case_gfor_index if p.startswith('p')]
    reactive_power_gfor=[q for q in case_gfor_index if q.startswith('q')]
    
    active_power_gfol=[p for p in case_gfol_index if p.startswith('p')]
    reactive_power_gfol=[q for q in case_gfol_index if q.startswith('q')]

    active_power_cig=[p for p in case_cig_index if p.startswith('p')]

    d_raw_data['generator']['P_GFOR']= np.array(case_gfor[active_power_gfor])
    d_op['Generators']['Snom_GFOR']= d_op['Generators']['Snom_CIG']*np.array(case_gfor[active_power_gfor])/np.array(case_cig[active_power_cig])
    
    no_cig=np.where(np.array(case_cig[active_power_cig])==0)
    d_op['Generators']['Snom_GFOR'].iloc[no_cig]=0
    
    d_raw_data['generator']['P_GFOL']=np.array(case_gfol[active_power_gfol])
    d_op['Generators']['Snom_GFOL']= d_op['Generators']['Snom_CIG']-d_op['Generators']['Snom_GFOR']
    
    return d_raw_data, d_op

def alphas_P(d_raw_data):
    for el in ['SG','GFOR','GFOL']:
        d_raw_data['generator']['alpha_P_'+el]=d_raw_data['generator']['P_'+el]/d_raw_data['generator']['PG']
        
    return d_raw_data
            


   
    
    
    