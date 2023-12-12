import pandas as pd
import numpy as np
import random

def random_operating_point(d_op,d_raw_data, GridCal_grid,n_reg=1,loads_power_factor=0.95, generators_power_factor=0.95, all_gfor=False):
    Pd=random_demand(list(d_op['Demand'].MinLoad),list(d_op['Demand'].PeakLoad),n_reg)
    
    d_raw_data = assign_loads_to_d_raw_data(d_raw_data, d_op, n_reg, loads_power_factor,Pd)
    
    Pg=np.sum(Pd)

    n_gen=len(d_raw_data['generator'])    
    
    d_raw_data=assign_gens_to_d_raw_data(d_raw_data, d_op, n_gen, generators_power_factor, Pg)
    
    d_raw_data = assign_P_by_CIG_and_SG(d_raw_data, d_op, n_gen)
    
    d_raw_data, d_op = assign_Snom_GFOL_GFOR(d_raw_data, d_op, GridCal_grid, n_gen, all_gfor)
    
    d_raw_data = alphas_P(d_raw_data)
    
    return d_raw_data, d_op

    

def random_demand(Pd_min,Pd_max,n_reg):
    pi=random.uniform(0, 1)
    
    Pd=[]

    for p in range(0,n_reg):
        Pd.append(pi*(Pd_max[p]-Pd_min[p])+Pd_min[p])
    
    return Pd

def assign_loads_to_d_raw_data(d_raw_data,d_op,n_reg,loads_power_factor,Pd):
    for r in range(1,n_reg+1):
        pf=np.array(d_op['Loads'].query('Region == @r')['Load_Participation_Factor'])
        d_raw_data['load'].loc[d_raw_data['load'].query('Region == @r').index,'PL']=Pd[r-1]*pf
        
        d_raw_data['load']['P']=d_raw_data['load']['PL']/100
        
    d_raw_data['load']['QL']=d_raw_data['load']['PL']*np.sqrt(1-loads_power_factor**2)/loads_power_factor
    d_raw_data['load']['Q']=d_raw_data['load']['P']*np.sqrt(1-loads_power_factor**2)/loads_power_factor
    
    return d_raw_data

def assign_gens_to_d_raw_data(d_raw_data,d_op,n_gen,generators_power_factor,Pg):
    gamma= np.random.dirichlet(np.ones(n_gen))
    Pg_i=Pg*gamma
    Pg_tot_i=np.array(d_op['Generators']['Pmax'])
    ind_all=np.arange(0,n_gen)
    while any(Pg_tot_i<Pg_i):
        ind=np.where(Pg_tot_i<Pg_i)[0]
        P_exc=0
        for i in ind:
            P_delta=Pg_i[i]-Pg_tot_i[i]
            P_exc=P_exc+P_delta
            Pg_i[i]=Pg_tot_i[i]
        ind_non_exc=list(set(ind_all)-set(ind))
        Pg_i[ind_non_exc]=Pg_i[ind_non_exc]+P_exc/len(ind_non_exc)
        ind_all=list(set(ind_all)-set(ind))
    d_raw_data['generator']['PG']=Pg_i
    d_raw_data['generator']['QG']=Pg_i*np.sqrt(1-generators_power_factor**2)/generators_power_factor
       
    return d_raw_data

def assign_P_by_CIG_and_SG(d_raw_data, d_op, n_gen):
    alpha=np.random.random((n_gen,1)) # percentage of P injected by CIG
    
    alpha_max=np.ones([n_gen,1])
    alpha_max[d_op['Generators'].query('Pmax_CIG == 0').index]=0
    alpha_max[np.where(d_op['Generators']['Pmax_CIG']>d_raw_data['generator']['PG']),0]=d_raw_data['generator'].loc[np.where(d_op['Generators']['Pmax_CIG']>d_raw_data['generator']['PG']),'PG']/d_op['Generators'].loc[np.where(d_op['Generators']['Pmax_CIG']>d_raw_data['generator']['PG']),'Pmax_CIG']
    
    alpha=alpha_max[:,0]*alpha[:,0]
        
    d_raw_data['generator'][['P_CIG']]=d_op['Generators'][['Pmax_CIG']]*alpha.reshape(-1,1)
    d_raw_data['generator']['P_SG']=d_raw_data['generator']['PG']-d_raw_data['generator']['P_CIG']
    # print(any(d_raw_data['generator']['P_CIG']>d_op['Generators']['Pmax_CIG']))
    # print(any(d_raw_data['generator']['P_SG']>d_op['Generators']['Pmax_SG']))
    # print(any(d_raw_data['generator']['P_SG']<0))
    
    if any(d_raw_data['generator']['P_SG']>d_op['Generators']['Pmax_SG']):
        excess_sg=d_raw_data['generator']['P_SG']>d_op['Generators']['Pmax_SG']
        ind=excess_sg[excess_sg].index
        delta=d_raw_data['generator'].loc[ind,'P_SG']-d_op['Generators'].loc[ind,'Pmax_SG']
        d_raw_data['generator'].loc[ind,'P_CIG']=d_raw_data['generator'].loc[ind,'P_CIG']+delta
        d_raw_data['generator'].loc[ind,'P_SG']=d_raw_data['generator'].loc[ind,'P_SG']-delta
        
    # print(any(d_raw_data['generator']['P_SG']>d_op['Generators']['Pmax_SG']))
    
    return d_raw_data

def assign_Snom_GFOL_GFOR(d_raw_data, d_op, GridCal_grid, n_gen, all_gfor):
                    
    buses=GridCal_grid.get_buses()
    for bus in buses:
        if bus.determine_bus_type()._value_ == 3 :
            
            slack_bus_name=bus._name
            break
    
    if all_gfor:    
        beta=np.ones([n_gen,1])
    else:
        beta=np.random.random((n_gen,1))
        ind_slack=d_op['Generators'].query('BusName == @slack_bus_name').index[0]
        beta[ind_slack]=1
    
    d_raw_data['generator']['P_GFOR']=d_raw_data['generator']['P_CIG']*beta.ravel()
    d_op['Generators']['Snom_GFOR']= d_op['Generators']['Pmax_CIG']*beta.ravel()/0.8
    
    d_raw_data['generator']['P_GFOL']=d_raw_data['generator']['P_CIG']-d_raw_data['generator']['P_GFOR']
    d_op['Generators']['Snom_GFOL']= d_op['Generators']['Pmax_CIG']*(1-beta.ravel())/0.8
    
    return d_raw_data, d_op

def alphas_P(d_raw_data):
    for el in ['SG','GFOR','GFOL']:
        d_raw_data['generator']['alpha_P_'+el]=d_raw_data['generator']['P_'+el]/d_raw_data['generator']['PG']
        
    return d_raw_data
            


   
    
    
    