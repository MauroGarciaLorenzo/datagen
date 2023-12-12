# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 13:00:33 2023

@author: Francesca
"""

import numpy as np
import pandas as pd

from tool.state_space import generate_NET

def fill_d_grid(d_grid, GridCal_grid, d_pf, d_raw_data, d_op):
    
    d_raw_data= change_d_raw_data_columns_name(d_raw_data)
     
    # Calculate RX loads
    d_grid['T_load'] = calculate_R_X_loads(d_grid['T_load'], d_pf['pf_load'], GridCal_grid)
    
    d_grid['T_global']=d_raw_data['data_global'].copy(deep=True)    
    
    d_grid = fill_NET(d_grid, d_raw_data)
    
    d_grid = fill_GEN(d_grid, d_op, d_raw_data, d_pf, GridCal_grid)
    
    d_grid = fill_SG(d_grid, d_raw_data)

    d_grid = fill_VSC(d_grid, d_raw_data)
    
    d_grid = fill_BUSES(d_grid, d_pf)

    # Write power-flow data to generator elements tables: T_TH, T_SG, T_VSC, T_user    
    # T_nodes = generate_NET.generate_T_nodes(d_grid)
    
    # d_grid = PF2table(T_nodes, d_grid, pf_gen, GridCal_grid)
    
    return d_grid, d_pf

def change_d_raw_data_columns_name(d_raw_data):
    
    d_raw_data['branch']=d_raw_data['branch'].rename(columns={'I':'bus_from','J':'bus_to','STAT':'state'})
    d_raw_data['data_global']=d_raw_data['data_global'].rename(columns={'AREA':'Area','BASKV':'Vb_kV'})
        
    return d_raw_data
    

def calculate_R_X_loads(T_load, pf_load, GridCal_grid):
    
    T_load['V'] = pf_load["Vm"]
    T_load['theta'] = pf_load["theta"]
    T_load['P'] = pf_load["P"]
    T_load['Q'] = pf_load["Q"]
    
    T_load= Load_type(T_load, GridCal_grid)
    
    T_load.loc[T_load["type"] == "PQ", "R"] = T_load.loc[T_load["type"] == "PQ", "V"] ** 2 / T_load.loc[T_load["type"] == "PQ", "P"]
    T_load.loc[T_load["type"] == "PQ", "X"] = T_load.loc[T_load["type"] == "PQ", "V"] ** 2 / T_load.loc[T_load["type"] == "PQ", "Q"]  
    return T_load

def Load_type(T_load, GridCal_grid):
    loads=GridCal_grid.get_loads()
    l=0
    for load in loads:
        T_load.loc[l,'number']=l+1
        T_load.loc[l,'bus']=int(load.bus.code)
        T_load.loc[l,'state']=1
        
        if load.P !=0 and load.Q!=0:
            T_load.loc[l,'type']='PQ'
            
        l=l+1
            
    return T_load
 
def fill_NET(d_grid, d_raw_data):
    
    for c in d_grid['T_NET'].columns:
        if c in d_raw_data['branch'].columns:
            d_grid['T_NET'][c]=d_raw_data['branch'][c]
    
    d_grid['T_NET']['number']= np.arange(1,len(d_grid['T_NET'])+1)
    
    return d_grid
        
def fill_GEN(d_grid, d_op, d_raw_data, d_pf, GridCal_grid):
    j=0
    SG=['SG']
    VSC=['GFOL','GFOR']
    
    for gen in [SG,VSC]: 
        n=0
        for element in gen:
            ind=d_op['Generators'].query('Snom_{} !=0'.format(element)).index
            for i in ind: 
                d_grid['T_gen'].loc[j,'number']=n+1
                d_grid['T_gen'].loc[j,'bus']=d_op['Generators'].loc[i,'BusNum']
                d_grid['T_gen'].loc[j,'element']=element        
                d_grid['T_gen'].loc[j,'Sn']=d_op['Generators'].loc[i,'Snom_{}'.format(element)]
                d_grid['T_gen'].loc[j,'Area']=1
                d_grid['T_gen'].loc[j,'state']=1
                
                d_grid['T_gen'].loc[j,'P']= d_raw_data['generator'].loc[i,'alpha_P_'+element]*d_pf['pf_gen'].loc[i,'P']
                
                d_grid['T_gen'].loc[j,'Q']= d_grid['T_gen'].loc[j,'P']*np.tan(np.arccos(d_pf['pf_gen'].loc[i,'cosphi']))*np.sign(d_pf['pf_gen'].loc[i,'Q'])
            
                d_grid['T_gen'].loc[j,'V']= d_pf['pf_gen'].loc[i,'Vm']
                d_grid['T_gen'].loc[j,'theta']= d_pf['pf_gen'].loc[i,'theta']
                
                d_grid['T_gen'].loc[j,'type']= check_slack(d_grid['T_gen'].loc[j,'bus'], GridCal_grid)
                j=j+1
                n=n+1
                         
    return d_grid    

def check_slack(gen_bus, GridCal_grid):
    buses=GridCal_grid.get_buses()
    for bus in buses:
        if int(bus.code)==gen_bus:
            if bus.is_slack:
                gen_type=0
            else:
                gen_type=1
            break
    return gen_type

def fill_SG(d_grid, d_raw_data):
    df_sg=d_grid['T_gen'].query('element == "SG"')
    for c in d_grid['T_SG'].columns:
        if c in df_sg.columns:
            d_grid['T_SG'][c]=df_sg[c]
    buses=list(d_grid['T_SG']['bus'])
    Vn=list(d_raw_data['generator'].loc[d_raw_data['generator'].query('I == @buses').index,'BASKV'])
    d_grid['T_SG'].loc[d_grid['T_SG'].query('bus == @buses').index,'Vn']=Vn
    
    d_grid['T_SG']=d_grid['T_SG'].reset_index(drop=True)
    return d_grid

def fill_VSC(d_grid, d_raw_data):
    df_vsc=d_grid['T_gen'].query('element == "GFOR" or element == "GFOL"')
    for c in d_grid['T_VSC'].columns:
        if c in df_vsc.columns:
            d_grid['T_VSC'][c]=df_vsc[c]
    
    for b in list(d_grid['T_VSC']['bus']):
        d_grid['T_VSC'].loc[d_grid['T_VSC'].query('bus == @b').index,'Vn']= d_raw_data['generator'].loc[d_raw_data['generator'].query('I == @b').index[0],'BASKV']
    
    d_grid['T_VSC']['mode']=df_vsc['element']
    
    d_grid['T_VSC']=d_grid['T_VSC'].reset_index(drop=True)
    
    return d_grid

def fill_BUSES(d_grid, d_pf):
    d_grid['T_buses']=d_pf['pf_bus'][['bus','Vm','theta']]
    d_grid['T_buses']['Area']=1
    
    return d_grid
            
            
    

def PF2table(T_nodes, d_grid, pf_gen, GridCal_grid):    
    
    T_case = d_grid['T_case']
    l_gen = d_grid['gen_names']    
    Sbase = GridCal_grid.Sbase
    
    for xx in l_gen:
        
        T_xx = d_grid[f'T_{xx}']
                
        if T_xx.empty:
            pass
        else:    
            bus = T_xx['bus'].copy()
            Pt = T_xx['bus'].copy().to_frame()
            Qt = T_xx['bus'].copy().to_frame()
            V = T_xx['bus'].copy().to_frame()
            theta = T_xx['bus'].copy().to_frame() 
            Sn = T_xx[['bus','Sn']].copy() 
            
            for idx,bus in bus.items():
            
                Pt.at[idx,'P'] = pf_gen.loc[pf_gen.bus == bus, "P"].values[0]
                Qt.at[idx,'Q'] = pf_gen.loc[pf_gen.bus == bus, "Q"].values[0]
                Pt.reset_index(drop=True, inplace=True)
                Qt.reset_index(drop=True, inplace=True)
                
                V.at[idx,'Vm'] = pf_gen.loc[pf_gen.bus == bus, "Vm"].values[0]
                theta.at[idx,'theta'] = pf_gen.loc[pf_gen.bus == bus, "theta"].values[0]
                V.reset_index(drop=True, inplace=True)
                theta.reset_index(drop=True, inplace=True)    
                            
            element_columns = [col for col in T_nodes.columns if col.startswith('Element_')]
            T_nodes['single_gen'] = T_nodes[element_columns].apply(
                lambda row: sum(1 for element in row if isinstance(element, str) and 'load' not in element) == 1, axis=1)        
                                
            if not T_case.empty:
                
                # Multiple elements are represented in Power-flow as one element (REE approach) 
                # Total penetration is shared inside the generator according to T_case:
                
                # If only one element type XX connected to the bus, all pf_gen goes to XX  
                # otherwise, compute penetration of each element 
                    
                 # if single_gen = True --> keep P
                 # if single_gen = False --> T_case.loc[T_case['bus'] = bus, f'  ']
                                
                 if xx == 'VSC':
                     P = [Pt.loc[idx,'P'] if T_nodes.loc[T_nodes['Node'] == Pt.loc[idx,'bus'], 'single_gen'].bool() 
                          else Pt.loc[idx,'P']*T_case.loc[T_case['bus'] == Pt.loc[idx,'bus'], f'{T_xx["mode"][idx]}'].values[0] for idx in Pt.index]
                   
                     Q = [Qt.loc[idx,'Q'] if T_nodes.loc[T_nodes['Node'] == Qt.loc[idx,'bus'], 'single_gen'].bool() 
                          else Qt.loc[idx,'Q']*T_case.loc[T_case['bus'] == Qt.loc[idx,'bus'], f'{T_xx["mode"][idx]}'].values[0] for idx in Qt.index]   
                     
                     Sn_scaled = [Sn.loc[idx,'Sn'] if T_nodes.loc[T_nodes['Node'] == Sn.loc[idx,'bus'], 'single_gen'].bool() 
                                 else Sn.loc[idx,'Sn']*T_case.loc[T_case['bus'] == Sn.loc[idx,'bus'], f'{T_xx["mode"][idx]}'].values[0] for idx in Sn.index]  
                     
                 else:            
                     P = [Pt.loc[idx,'P'] if T_nodes.loc[T_nodes['Node'] == Pt.loc[idx,'bus'], 'single_gen'].bool() 
                          else Pt.loc[idx,'P']*T_case.loc[T_case['bus'] == Pt.loc[idx,'bus'], f'{xx}'].values[0] for idx in Pt.index]
                   
                     Q = [Qt.loc[idx,'Q'] if T_nodes.loc[T_nodes['Node'] == Qt.loc[idx,'bus'], 'single_gen'].bool() 
                          else Qt.loc[idx,'Q']*T_case.loc[T_case['bus'] == Qt.loc[idx,'bus'], f'{xx}'].values[0] for idx in Qt.index]    
                     
                     Sn_scaled = [Sn.loc[idx,'Sn'] if T_nodes.loc[T_nodes['Node'] == Sn.loc[idx,'bus'], 'single_gen'].bool() 
                                 else Sn.loc[idx,'Sn']*T_case.loc[T_case['bus'] == Sn.loc[idx,'bus'], f'{xx}'].values[0] for idx in Sn.index]                                  
                
            else:
                
            # Multiple elements are represented in Power-flow as multiple elements
            # The share that corresponds to each element is computed according to their size:
                
                # if single_gen = True --> keep P, Q
                # if single_gen = False --> get P, compute Q
                
                P = Pt['P'].copy()
                Q = Qt['Q'].copy()
                for idx,gen in enumerate(GridCal_grid.get_generators()):
                    if not T_nodes.loc[T_nodes['Node'] == Pt.loc[idx,'bus'], 'single_gen'].bool():
                        P[idx] = gen.get_properties_dict()['p']/Sbase
                        
                        bus_id = gen.get_properties_dict()['bus']                                                
                        S_base = gen.get_properties_dict()['snom']
                        S_total = sum([g.get_properties_dict()['snom'] if g.get_properties_dict()['bus']==bus_id else 0 for g in GridCal_grid.get_generators()])                        
                        Q[idx] =  Qt.loc[idx,'Q']*S_base/S_total               
                                             
            T_xx['P'] = P
            T_xx['Q'] = Q  
            T_xx['V'] = V['Vm']
            T_xx['theta'] = theta['theta']
            T_xx['Sn'] = Sn_scaled
            
            d_grid[f'T_{xx}'] = T_xx
        
    return d_grid
    
    