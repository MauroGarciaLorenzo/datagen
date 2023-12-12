import pandas as pd
import numpy as np
import re
import control as ct
from control.matlab import ss
from tool.preprocess import preprocess_data


def generate_SS_NET_blocks(d_grid, delta_slk):
    
    l_blocks = [] # Create list to store subsystems blocks
    l_states = [] # Create list to store subsystems state labels
    d_grid = xb2lc(d_grid) # Convert X,B columns to R,L


    # AC Grid   
    
    # Generates the Connectivity Matrix and the Table of nodes for the AC grid:
    connect_mtx, connect_mtx_PI, connect_mtx_rl = generate_general_connect_mtx(d_grid)
    T_nodes = generate_T_nodes(d_grid)    
    
    # RL NET Add aditional elements    
    # Add transformers
    connect_mtx_rl, T_NET_wTf, T_trafo_missing = add_trafo(d_grid, connect_mtx_rl)
    rl_T_nodes = generate_specific_T_nodes_v2(connect_mtx_rl, T_nodes);    
    # Add Thevenins 
    connect_mtx_rl, T_NET_wTf_wTh, T_TH_missing, rl_T_nodes = add_TH(d_grid['T_TH'], connect_mtx_rl, connect_mtx_PI, T_NET_wTf, rl_T_nodes)        
    
    # RL NET    
    rl_T_NET = get_specific_NET(connect_mtx_rl, T_NET_wTf_wTh)    
    # Get PI T_nodes
    PI_T_nodes = generate_specific_T_nodes_v2(connect_mtx_PI, T_nodes);
    # Generate the State-Space of the AC RL grid
    l_blocks = generate_general_rl_NET_v3(connect_mtx_rl, rl_T_nodes, PI_T_nodes, rl_T_NET, d_grid['T_global'], l_blocks, l_states) # NOT FINISHED !!!
       
    # PI NET     
    PI_T_NET = get_specific_NET(connect_mtx_PI, d_grid['T_NET'])
    # Generates the State-Space of the AC PI grid
    l_blocks, l_states = generate_general_PI_NET(connect_mtx_PI, connect_mtx_rl, PI_T_nodes, T_trafo_missing, d_grid, l_blocks, l_states)
        
    # Trafos 
    l_blocks, l_states = build_trafo(T_trafo_missing, d_grid['T_global']['fb'][0], l_blocks, l_states)
    
    # TH 
    l_blocks, l_states = build_TH(T_TH_missing, d_grid['T_global']['fb'][0], l_blocks, l_states)   
    
    # Loads
    l_blocks, l_states = build_load(d_grid['T_load'], connect_mtx_PI, d_grid['T_global']['fb'][0], delta_slk, l_blocks, l_states)
        
    
    # DC Grid
    
    # generate_DC_connectivity_matrix
    # generate_DC_NET
    print("DC grid SS not implemented yet")
    
    return l_blocks, l_states, d_grid


def xb2lc(d_grid):
    T_buses = d_grid['T_buses']
    T_buses['fb'] = d_grid['T_buses']['Area'].map(d_grid['T_global'].set_index('Area')['fb'])    
        
    d_grid['T_NET']['L'] = d_grid['T_NET']['X'] / (2 * np.pi * d_grid['T_NET']['bus_from'].map(T_buses.set_index('bus')['fb']))
    d_grid['T_NET']['C'] = d_grid['T_NET']['B'] / (2 * np.pi * d_grid['T_NET']['bus_from'].map(T_buses.set_index('bus')['fb'])) / 2

    d_grid['T_trafo']['L'] = d_grid['T_trafo']['X'] / (2 * np.pi * d_grid['T_trafo']['bus_from'].map(T_buses.set_index('bus')['fb']))
    d_grid['T_trafo']['C'] = d_grid['T_trafo']['B'] / (2 * np.pi * d_grid['T_trafo']['bus_from'].map(T_buses.set_index('bus')['fb']))

    d_grid['T_load']['L'] = d_grid['T_load']['X'] / (2 * np.pi * d_grid['T_load']['bus'].map(T_buses.set_index('bus')['fb']))
    
    d_grid['T_TH']['L'] = d_grid['T_TH']['X'] / (2 * np.pi * d_grid['T_TH']['Area'].map(T_buses.set_index('bus')['fb']))
    
    return d_grid



def generate_general_connect_mtx(d_grid):

    T_NET = d_grid['T_NET']
    
    # Connectivity Matrix generation:
    n_nodes = max([max(T_NET['bus_from']), max(T_NET['bus_to'])])
    connect_mtx_rl = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    connect_mtx_PI = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    for i in range(len(T_NET)):
        if T_NET['B'][i] == 0:
            connect_mtx_rl[T_NET['bus_from'][i]-1, T_NET['bus_to'][i]-1] = 1
        else:
            connect_mtx_PI[T_NET['bus_from'][i]-1, T_NET['bus_to'][i]-1] = 1
    connect_mtx_rl = connect_mtx_rl + connect_mtx_rl.T
    connect_mtx_PI = connect_mtx_PI + connect_mtx_PI.T
    connect_mtx = connect_mtx_PI + connect_mtx_rl
    
    return connect_mtx, connect_mtx_PI, connect_mtx_rl
    

def generate_T_nodes(d_grid):   
    
    T_NET = d_grid['T_NET']
    T_trafo = d_grid['T_trafo']
    T_load = d_grid['T_load']
    T_TH = d_grid['T_TH']
    T_SG = d_grid['T_SG']
    T_VSC = d_grid['T_VSC']
    T_MMC = d_grid['T_MMC']
    T_user = d_grid['T_user']
     
    # Nodes specification:
    tn = max(T_NET['bus_from'].max(), T_NET['bus_to'].max(), T_trafo['bus_from'].max(), T_trafo['bus_to'].max())  # Total node number
    
    nsb = np.zeros(tn)
    for node in range(1, tn+1):
        nsb[node-1] = sum([T_load['bus'].eq(node).sum(), T_TH['bus'].eq(node).sum(), T_SG['bus'].eq(node).sum(), 
                          T_VSC['bus'].eq(node).sum(), T_MMC['NodeAC'].eq(node).sum(),T_user['bus'].eq(node).sum()])
    nsb = int(max(nsb))
    
    sz = (tn, nsb+1)  # Table size
    data = np.empty(sz)
    data.fill(np.nan)
    varNames = ['Node'] + [f'Element_{i}' for i in range(1, nsb+1)]
    T_nodes = pd.DataFrame(data, columns=varNames)
        
    # Assign values to the T_nodes table
    T_nodes['Node'] = range(1, tn+1)
        
    for mmc in range(len(T_MMC)):
        emptyCellnotFound = True
        i = 1
        while(emptyCellnotFound):
            if pd.isna(T_nodes.loc[T_nodes['Node'] == T_MMC['NodeAC'][mmc], f'Element_{i}']).any(): #cell not empty
                emptyCellnotFound = False
            else:
                i = i+1
        T_nodes.loc[T_nodes['Node'] == T_MMC['NodeAC'][mmc], f'Element_{i}'] = f"MMC{int(T_MMC['number'][mmc])}"
        
                
    
    xx_list = [T_load, T_TH, T_SG, T_VSC, T_user]
    xx_names = ["load", "TH", "SG", "VSC", "user"]
    
    for idx, T_xx in enumerate(xx_list):
        for xx in range(len(T_xx)):
            emptyCellnotFound = True
            i = 1
            while(emptyCellnotFound):
                if pd.isna(T_nodes.loc[T_nodes['Node'] == T_xx['bus'][xx], f'Element_{i}']).any(): #cell not empty
                    emptyCellnotFound = False
                else:
                    i = i+1
            T_nodes.loc[T_nodes['Node'] == T_xx['bus'][xx], f'Element_{i}'] = f"{xx_names[idx]}{int(T_xx['number'][xx])}"
    
    return T_nodes


def add_trafo(d_grid, connect_mtx_rl):
    
    T_NET = d_grid['T_NET']
    T_trafo = d_grid['T_trafo']
    number = np.max(T_NET['number']) + 1
    
    missing = []
    for tf in range(len(T_trafo)):
        if np.sum(connect_mtx_rl[T_trafo['bus_from'][tf]-1]) > 0 or np.sum(connect_mtx_rl[T_trafo['bus_to'][tf]-1]) > 0:
            connect_mtx_rl[T_trafo['bus_from'][tf]-1, T_trafo['bus_to'][tf]-1] = 1
            connect_mtx_rl[T_trafo['bus_to'][tf]-1, T_trafo['bus_from'][tf]-1] = 1
            T_NET.loc[len(T_NET)] = [number, T_trafo['bus_from'][tf], T_trafo['bus_to'][tf], T_trafo['R'][tf], T_trafo['X'][tf], 0, T_trafo['L'][tf], T_trafo['C'][tf]]
            number += 1
        else:
            missing.append(tf)
    T_trafo_missing = T_trafo.iloc[missing]
    
    return connect_mtx_rl, T_NET, T_trafo_missing


def generate_specific_T_nodes_v2(connect_mtx_rl, T_nodes):
    
    lines = np.sum(connect_mtx_rl, axis=0)
    for idx,val in enumerate(lines):
        if val == 0:
            T_nodes = T_nodes.drop(idx)
    T_nodes_specific = T_nodes.copy()
    return T_nodes_specific


def add_TH(T_TH, connect_mtx_rl, connect_mtx_PI, T_NET_wTf, rl_T_nodes): 
    
    nodeACmax = connect_mtx_rl.shape[0] + 1
    number = np.max(T_NET_wTf['number']) + 1
    index_table = rl_T_nodes.shape[0]
    missing = []

    for th in range(T_TH.shape[0]):
        if np.sum(connect_mtx_rl[T_TH['bus'][th]-1, :]) > 0 and np.sum(connect_mtx_PI[T_TH['bus'][th]-1, :]) == 0:
            
            # Create new RL node            
            newCol = np.zeros((connect_mtx_rl.shape[0],1))
            newCol[T_TH['bus'][th]-1,0] = 1
            connect_mtx_rl = np.append(connect_mtx_rl,newCol,axis = 1)
            
            newRow = np.zeros((1,connect_mtx_rl.shape[1]))
            newRow[0,T_TH['bus'][th]-1] = 1
            connect_mtx_rl = np.append(connect_mtx_rl,newRow,axis = 0)
            
            # Add row to T_NET
            T_NET_wTf.loc[len(T_NET_wTf)] = [number, T_TH['bus'][th], nodeACmax, T_TH['R'][th], T_TH['X'][th], 0, 1, T_TH['L'][th], 0]
            
            rl_T_nodes.loc[index_table, 'Node'] = nodeACmax
            
            rl_T_nodes.loc[index_table, 'Element_1'] = 'Additional TH' + str(T_TH['number'][th])
            
            index_table += 1            
            nodeACmax += 1            
            number += 1
            
        else:
            missing.append(th)

    T_TH_missing = T_TH.iloc[missing, :]

    return connect_mtx_rl, T_NET_wTf, T_TH_missing, rl_T_nodes    
    
    
def get_specific_NET(connect_mtx, T_NET):
    index = []
    
    for i in range(connect_mtx.shape[0]):
        for j in range(connect_mtx.shape[1]):
            if connect_mtx[i, j] == 1:
                found_rows = T_NET[(T_NET['bus_from'] == i+1) & (T_NET['bus_to'] == j+1)]
                if not found_rows.empty:
                    index.append(found_rows.index[0])
    
    NET = T_NET.loc[index, :]
    
    return NET    


def generate_general_rl_NET_v3(connect_mtx_rl, rl_T_nodes, PI_T_nodes, rl_T_NET, T_global, l_blocks, l_states):
    
    # ---------------------------------------------------
    # LA DEIXO PER MÉS ENDEVANT. AL 118 NO HI HA BUSOS RL    
    # ---------------------------------------------------
        
    if connect_mtx_rl.any():
        # Order the buses of the rl lines
        rl_T_NET = preprocess_data.reorder_buses_lines(rl_T_NET)    
        
        print("Error: RL grid SS not implemented yet. Cannot build current system")
        
        # Append SS to l_blocks
        
    return l_blocks
        
    
def generate_general_PI_NET(connect_mtx_PI, connect_mtx_rl, PI_T_nodes, T_trafo_missing, d_grid, l_blocks, l_states):   
    
    def construccio_SS_Cn( Cn , c_x , c_u , c_y ,f):
        
        # Generate SS of capacitor bus
        w = 2*np.pi*f # grid frequency  
        
        Ac = np.array([[0, -w], [w, 0]])        
        Bc = np.array([[1 / Cn, 0], [0, 1 / Cn]])        
        Cc = np.array([[1, 0], [0, 1]])        
        Dc = np.zeros((2, 2))

        c = ss(Ac, Bc, Cc, Dc, inputs = c_u, outputs = c_y, states = c_x)    
        return c
           

    def construccio_SS_rl( Rxy , Lxy , rl_x , rl_u, rl_y,f):
        
        # Generate SS of RL line
        w = 2*np.pi*f # grid frequency
        
        Arl = np.array([[-Rxy/Lxy, -w], [w, -Rxy/Lxy]])        
        Brl = np.array([[1/Lxy, -1/Lxy, 0, 0], [0, 0, 1/Lxy, -1/Lxy]])        
        Crl = np.array([[1, 0], [0, 1]])        
        Drl = np.zeros((2, 4))

        rl = ss(Arl, Brl, Crl, Drl, inputs = rl_u, outputs = rl_y, states = rl_x)    
        return rl


    def construccio_SS_nus( Din , llista_u_nus , llista_y_nus):
     
        # State-space sum of currents in node
        Ain = np.empty(0)
        columns_Bin = Din.shape[1]
        Bin = np.empty((0, columns_Bin))
        Cin = np.empty((0, 2))
        in_ss = ss(Ain, Bin, Cin, Din, inputs = llista_u_nus, outputs = llista_y_nus)  
        return in_ss
    
       
    
    if connect_mtx_PI.any():
        T_NET = d_grid['T_NET']
        f = d_grid['T_global']['f_Hz'][0]
        
        # List of all the SS-blocks
        llista_SS_rl = []
        llista_SS_C = []
        llista_SS_nus = []
        
        # List of states and I/O names for the complete SS
        llista_x_AC = []
        llista_u_AC = []
        llista_y_AC = []
                
        # Iterate over the buses in Connectivity_Matrix_PI
        
        for ii in range(connect_mtx_PI.shape[0]):
            if sum(connect_mtx_PI[ii,:]) > 0:
                Cn = 0
                position = np.where(connect_mtx_PI[ii,:]==1)[0]
                for Node in position:
                    C1 = T_NET.loc[ (T_NET['bus_from'] == ii+1) & (T_NET['bus_to'] == Node+1), 'C']
                    C2 = T_NET.loc[ (T_NET['bus_from'] == Node+1) & (T_NET['bus_to'] == ii+1), 'C']  
                    
                    if C1.empty:
                        C1 = T_NET.loc[(T_NET['bus_from'] == Node+1) & (T_NET['bus_to'] == ii+1), 'C'].values[0] / 2
                    else:
                        C1 = C1.values[0] / 2
                    if C2.empty:
                        C2 = T_NET.loc[(T_NET['bus_from'] == ii+1) & (T_NET['bus_to'] == Node+1), 'C'].values[0] / 2
                    else:
                        C2 = C2.values[0] / 2
                        
                    Cn += C1 + C2
                    
                c_x = ['vc_q'+str(ii+1), 'vc_d'+str(ii+1)]
                c_u = ['ic_q'+str(ii+1), 'ic_d'+str(ii+1)]
                c_y = ['NET_vn'+str(ii+1)+'q', 'NET_vn'+str(ii+1)+'d']                                
                # Build SS of the condenser
                SS_C_i = construccio_SS_Cn(Cn , c_x , c_u , c_y, f);                 
                # Append SS of the condenser to the list
                llista_SS_C.append(SS_C_i)
                # Add outputs to global AC system:
                llista_y_AC.append('NET_vn'+str(ii+1)+'q')
                llista_y_AC.append('NET_vn'+str(ii+1)+'d')
                llista_x_AC.append('vc_q'+str(ii+1))
                llista_x_AC.append('vc_d'+str(ii+1))
                
                # Build D-matrix of the bus
                Din = []                
                # Create list for the bus-inputs
                llista_u_nus = []                
                # Create list for the bus-outputs
                llista_y_nus = ['ic_q'+str(ii+1), 'ic_d'+str(ii+1)]  
                
                # Find elements connected to bus
                
                for idx,row in T_trafo_missing.iterrows():
                    if row['bus_from'] == ii+1:
                        # Define bus inputs
                        llista_u_nus.append('Trafo' + str(int(row['number'])) + '_iq' + str(int(row['bus_from'])))
                        llista_u_nus.append('Trafo' + str(int(row['number'])) + '_id' + str(int(row['bus_from'])))
                        # Update Din
                        Din.append(np.array([[1, 0], [0, 1]])) #positive because it enters the bus
                        # Define global input of the system
                        llista_u_AC.append('Trafo' + str(int(row['number'])) + '_iq' + str(int(row['bus_from'])))
                        llista_u_AC.append('Trafo' + str(int(row['number'])) + '_id' + str(int(row['bus_from'])))
                    if row['bus_to'] == ii+1:
                        # Define bus inputs
                        llista_u_nus.append('Trafo' + str(int(row['number']))+ '_iq' + str(int(row['bus_to'])))
                        llista_u_nus.append('Trafo' + str(int(row['number']))+ '_id' + str(int(row['bus_to'])))
                        # Update Din
                        Din.append(np.array([[1, 0], [0, 1]])) #positive because it enters the bus
                        # Define global input of the system
                        llista_u_AC.append('Trafo' + str(int(row['number'])) + '_iq' + str(int(row['bus_to'])))   
                        llista_u_AC.append('Trafo' + str(int(row['number'])) + '_id' + str(int(row['bus_to'])))  
                        
                        
                PI_T_nodes_row = PI_T_nodes.loc[PI_T_nodes['Node'] == ii+1]
                for column in PI_T_nodes_row.columns[1:]:
                    value = PI_T_nodes_row[column]
                    if not pd.isna(value.values[0]):
                        match = re.match(r'([A-Za-z]+)(\d+)', value.values[0])
                        element = match.group(1)
                        number = match.group(2)
                                                
                        if element == 'load': 
                            # Define bus inputs
                            llista_u_nus.append('Load' + number + '_iq')
                            llista_u_nus.append('Load' + number + '_id')
                            # Update Din
                            Din.append(np.array([[1, 0], [0, 1]])) #positive because it enters the bus
                            # Define global input of the system
                            llista_u_AC.append('Load' + number + '_iq')   
                            llista_u_AC.append('Load' + number + '_id')  
                        
                        elif element == 'TH':
                            # Define bus inputs
                            llista_u_nus.append('TH' + number + '_iq')
                            llista_u_nus.append('TH' + number + '_id')
                            # Update Din
                            Din.append(np.array([[1, 0], [0, 1]])) #positive because it enters the bus
                            # Define global input of the system
                            llista_u_AC.append('TH' + number + '_iq')   
                            llista_u_AC.append('TH' + number + '_id')  
                            
                        elif element == 'SG':
                            # Define bus inputs
                            llista_u_nus.append('SG' + number + '_iq')
                            llista_u_nus.append('SG' + number + '_id')
                            # Update Din
                            Din.append(np.array([[1, 0], [0, 1]])) #positive because it enters the bus
                            # Define global input of the system
                            llista_u_AC.append('SG' + number + '_iq')   
                            llista_u_AC.append('SG' + number +'_id')  
                            
                        elif element == 'VSC':
                            mode = d_grid['T_VSC'].loc[(d_grid['T_VSC']['bus'] == ii+1) & (d_grid['T_VSC']['number'] == int(number)), "mode"].values[0]
                            # Define bus inputs
                            llista_u_nus.append(mode + number + '_iq')
                            llista_u_nus.append(mode + number + '_id')
                            # Update Din
                            Din.append(np.array([[1, 0], [0, 1]])) #positive because it enters the bus
                            # Define global input of the system
                            llista_u_AC.append(mode + number +'_iq')   
                            llista_u_AC.append(mode + number + '_id')                                 
                                
                        elif element == 'MMC':
                            # Define bus inputs
                            llista_u_nus.append('MMC' + number + '_idiffq')
                            llista_u_nus.append('MMC' + number + '_idiffd')
                            # Update Din
                            Din.append(np.array([[1, 0], [0, 1]])) #positive because it enters the bus
                            # Define global input of the system
                            llista_u_AC.append('MMC' + number + '_idiffq')   
                            llista_u_AC.append('MMC' + number +'_idiffd') 
                            
                        elif element == 'user':
                            # Define bus inputs
                            llista_u_nus.append('USER' + number + '_iq')
                            llista_u_nus.append('USER' + number + '_id')
                            # Update Din
                            Din.append(np.array([[1, 0], [0, 1]])) #positive because it enters the bus
                            # Define global input of the system
                            llista_u_AC.append('USER' + number + '_iq')   
                            llista_u_AC.append('USER' + number + '_id') 
                            
                        
                for jj in range(connect_mtx_PI.shape[1]):
                    if connect_mtx_PI[ii,jj] == 1:
                        if jj < ii:
                            # Build D-matrix of the bus
                            Din.append(np.array([[1, 0], [0, 1]])) #positive because it enters the bus
                            llista_u_nus.append('NET_iq_' + str(jj+1) + '_' + str(ii+1))
                            llista_u_nus.append('NET_id_' + str(jj+1) + '_' + str(ii+1))
                        elif jj > ii:
                            # Build D-matrix of the bus
                            Din.append(np.array([[-1, 0], [0, -1]])) #negative because it leaves the bus
                            llista_u_nus.append('NET_iq_' + str(ii+1) + '_' + str(jj+1))
                            llista_u_nus.append('NET_id_' + str(ii+1) + '_' + str(jj+1))     
                            # Set names of RL variables
                            rl_x = ['iq'+'_'+str(ii+1)+'_'+str(jj+1), 'id'+'_'+str(ii+1)+'_'+str(jj+1)]
                            rl_u = ['NET_vn'+str(ii+1)+'q', 'NET_vn'+str(jj+1)+'q', 'NET_vn'+str(ii+1)+'d', 'NET_vn'+str(jj+1)+'d'] 
                            rl_y = ['NET_iq_'+str(ii+1)+'_'+str(jj+1), 'NET_id_'+str(ii+1)+'_'+str(jj+1)]                    
                            # Build SS matrices of RL
                            SS_rl = construccio_SS_rl(T_NET.loc[(T_NET['bus_from']==ii+1) & (T_NET['bus_to']==jj+1),"R"].values[0] , T_NET.loc[(T_NET['bus_from']==ii+1) & (T_NET['bus_to']==jj+1),"L"].values[0] , rl_x , rl_u, rl_y,f);
                            # Append SS of the RL to the list
                            llista_SS_rl.append(SS_rl)                   
                            # Add outputs to global AC system:
                            llista_y_AC.append('NET_iq_'+str(ii+1)+'_'+str(jj+1))
                            llista_y_AC.append('NET_id_'+str(ii+1)+'_'+str(jj+1))         
                            llista_x_AC.append('iq'+'_'+str(ii+1)+'_'+str(jj+1))
                            llista_x_AC.append('id'+'_'+str(ii+1)+'_'+str(jj+1))
            
                
                for jj in range(connect_mtx_rl.shape[1]):
                    if connect_mtx_rl[ii,jj] == 1:
                        if jj < ii:
                            # Build D-matrix of the bus
                            Din.append(np.array([[1, 0], [0, 1]])) #positive because it enters the bus
                            llista_u_nus.append('NET_iq_' + str(jj+1) + '_' + str(ii+1))
                            llista_u_nus.append('NET_iq_' + str(jj+1) + '_' + str(ii+1))
                            if not 'NET_iq_' + str(jj+1) + '_' + str(ii+1) in llista_u_AC:
                                llista_u_AC.append('NET_iq_' + str(jj+1) + '_' + str(ii+1))  
                                llista_u_AC.append('NET_iq_' + str(jj+1) + '_' + str(ii+1))
                            
                        elif jj > ii: 
                            # Build D-matrix of the bus
                            Din.append(np.array([[-1, 0], [0, -1]])) #negative because it leaves the bus
                            llista_u_nus.append('NET_iq_' + str(ii+1) + '_' + str(jj+1))
                            llista_u_nus.append('NET_id_' + str(ii+1) + '_' + str(jj+1))
                            if not 'NET_iq_' + str(ii+1) + '_' + str(jj+1) in llista_u_AC:
                                llista_u_AC.append('NET_iq_' + str(ii+1) + '_' + str(jj+1)) 
                                llista_u_AC.append('NET_id_' + str(ii+1) + '_' + str(jj+1)) 
            
                # Build SS of the bus sum of currents
                Din = np.hstack(Din)
                SS_nus = construccio_SS_nus( Din , llista_u_nus , llista_y_nus)
                # Append SS of the bus sum of currents to the list
                llista_SS_nus.append(SS_nus)              
                       
        
        # Generate PI NET State-Space
        PI_NET = ct.interconnect(llista_SS_rl+llista_SS_C+llista_SS_nus, states = llista_x_AC, inputs=llista_u_AC, outputs=llista_y_AC, check_unused = False) 
        #Append SS to l_blocks 
        l_blocks.append(PI_NET)
        l_states.extend(llista_x_AC)
              
    return l_blocks, l_states
        


def build_trafo(T_trafo_missing, fb, l_blocks, l_states):
        
    for _,row in T_trafo_missing.iterrows():
        L = row['L']
        w_n = 2*np.pi*fb
        number = int(row['number'])
        nodeA = int(row['bus_from'])
        nodeB = int(row['bus_to'])
        R = row['R']
        
        A = np.array([[-R / L, -w_n],
                      [w_n, -R / L]])

        B = np.array([[1 / L, 0, -1 / L, 0],
                      [0, 1 / L, 0, -1 / L]])
        
        C = np.array([[-1, 0],
                      [0, -1],
                      [1, 0],
                      [0, 1]])
        
        D = np.zeros((4, 4))
        
        x = ['Trafo'+str(number)+'_iq', 'Trafo'+str(number)+'_id']
        u = ['NET_vn'+str(nodeA)+'q', 'NET_vn'+str(nodeA)+'d', 'NET_vn'+str(nodeB)+'q', 'NET_vn'+str(nodeB)+'d']
        y = ['Trafo'+str(number)+'_iq'+str(nodeA), 'Trafo'+str(number)+'_id'+str(nodeA),'Trafo'+str(number)+'_iq'+str(nodeB), 'Trafo'+str(number)+'_id'+str(nodeB)]
        
        trafo = ss(A, B, C, D, inputs = u, outputs = y, states = x)
        l_blocks.append(trafo)
        l_states.extend(x)        
    return l_blocks, l_states


def build_TH(T_TH_missing, fb, l_blocks, l_states):
        
    for _,row in T_TH_missing.iterrows():
        w_n = 2*np.pi*fb
        number = int(row['number'])
        nodeA = int(row['bus'])
        R = row['R']
        L = row['L']
        
        if L == 0:             
            A = np.empty(0)    
            B = np.empty((0,4))            
            C = np.empty((0, 2))            
            D = np.array([[1, 0, -1, 0], [0, 1, 0, -1]])*(1/R)
            
            u = ['TH'+str(number)+'_vnq', 'TH'+str(number)+'_vnd','NET_vn'+str(nodeA)+'q', 'NET_vn'+str(nodeA)+'d']
            y = ['TH'+str(number)+'_iq', 'TH'+str(number)+'_id']
            
            th = ss(A, B, C, D, inputs = u, outputs = y)
            l_blocks.append(th)
            
        else:     
            A = np.array([[-R / L, -w_n], [w_n, -R / L]])    
            B = np.array([[1 / L, 0, -1 / L, 0], [0, 1 / L, 0, -1 / L]])            
            C = np.array([[1, 0],[0, 1]])            
            D = np.zeros((2, 4))
            
            x = ['TH'+str(number)+'_iq_'+str(nodeA), 'TH'+str(number)+'_id_'+str(nodeA)]
            u = ['TH'+str(number)+'_vnq', 'TH'+str(number)+'_vnd','NET_vn'+str(nodeA)+'q', 'NET_vn'+str(nodeA)+'d']
            y = ['TH'+str(number)+'_iq', 'TH'+str(number)+'_id']

            th = ss(A, B, C, D, inputs = u, outputs = y, states = x)
            l_blocks.append(th)
            l_states.extend(x)
        
    return l_blocks, l_states


def build_load(T_load, connect_mtx_PI, f, delta_slk, l_blocks, l_states):
    
    
    def build_Load_in_PI_R_addR(row,delta_slk):
        
        R = row['R']
        number = int(row['number'])
        nodeAC = int(row['bus'])
        
        # Calculate vq0 vd0
        theta0 = row['theta'] * np.pi / 180 - delta_slk
        vq0 = row['V'] * np.cos(theta0) * np.sqrt(2 / 3)
        vd0 = -row['V'] * np.sin(theta0) * np.sqrt(2 / 3)
        
        Ar = np.empty(0)
        Br = np.empty((0,3))
        Cr = np.empty((0,2))
        Dr = np.array([[1/R, 0, -vq0/(R**2)],[0, 1/R, -vd0/(R**2)]])
        
        ur = ['NET_vn' + str(nodeAC) + 'q', 'NET_vn' + str(nodeAC) + 'd', 'NET_Rld' + str(number)]
        yr = ['Load' + str(number) + '_irq', 'Load' + str(number) + '_ird']
        
        load = ss(Ar, Br, Cr, Dr, inputs = ur, outputs = yr)        
        
        return load
    
    
    def build_Load_in_PI_addR(row,f,delta_slk):
    
        R = row['R']
        L = row['L']
        number = int(row['number'])
        nodeAC = int(row['bus'])     
        w = 2*np.pi*f
        
        # Calculate vq0 vd0
        theta0 = row['theta'] * np.pi / 180 - delta_slk
        vq0 = row['V'] * np.cos(theta0) * np.sqrt(2 / 3)
        vd0 = -row['V'] * np.sin(theta0) * np.sqrt(2 / 3)   
        
        Al = np.array([[0, -w],[w, 0]])    
        Bl = np.array([[1/L, 0],[0, 1/L]])    
        Cl = np.array([[1, 0],[0, 1]])    
        Dl = np.array([[0, 0],[0, 0]])
    
        xl = ['Load' + str(number) + '_ilq', 'Load' + str(number) + '_ild']
        ul = ['NET_vn' + str(nodeAC) + 'q', 'NET_vn' + str(nodeAC) + 'd']
        yl = ['Load' + str(number) + '_ilq', 'Load' + str(number) + '_ild']
        
        SS_l = ss(Al, Bl, Cl, Dl, inputs = ul, outputs = yl, states = xl)
        
        Ar = np.empty(0)   
        Br = np.empty((0,3))
        Cr = np.empty((0,2))  
        Dr = np.array([[1/R, 0, -vq0/(R**2)],[0, 1/R, -vd0/(R**2)]])
    
        ur = ['NET_vn' + str(nodeAC) + 'q', 'NET_vn' + str(nodeAC) + 'd', 'NET_Rld' + str(number)]    
        yr = ['Load' + str(number) + '_irq', 'Load' + str(number) + '_ird']
        
        SS_r = ss(Ar, Br, Cr, Dr, inputs = ur, outputs = yr)
        
        Anus = np.empty(0)
        Bnus = np.empty((0, 4))
        Cnus = np.empty((0, 2))        
        Dnus = np.array([[-1, -1, 0, 0],[0, 0, -1, -1]])
    
        unus = ['Load' + str(number) + '_irq','Load' + str(number) + '_ilq', 
                'Load' + str(number) + '_ird','Load' + str(number) + '_ild']    
        ynus = ['Load' + str(number) + '_iq','Load' + str(number) + '_id']
        
        SS_nus = ss(Anus, Bnus, Cnus, Dnus, inputs = unus, outputs = ynus)       
        
        uLoad = ['NET_vn' + str(nodeAC) + 'q', 'NET_vn' + str(nodeAC) + 'd', 'NET_Rld' + str(number)]
        load = ct.interconnect([SS_l, SS_r, SS_nus], states = xl, inputs=uLoad, outputs=ynus, check_unused = False) 
    
        return load, xl
    
    
    def build_Load_in_rl_R_addR(row,delta_slk):        

        print("Load in RL not implemented yet")        

        return load
        
    
    def build_Load_in_rl(row,f):
        
        print("Load in RL not implemented yet")
        
        return load, states
    
    
    for _,row in T_load.iterrows():
        
        bus = row['bus']
        L = row['L']
        
        if sum(connect_mtx_PI[bus-1,:]) > 0:
            if L == 0:
                load = build_Load_in_PI_R_addR(row,delta_slk) 
                l_blocks.append(load)
            else:
                load, states = build_Load_in_PI_addR(row,f,delta_slk)
                l_blocks.append(load)
                l_states.extend(states)
                
        else:
            if L == 0:
                load = build_Load_in_rl_R_addR(row,delta_slk)
                l_blocks.append(load)
            else:
                load, states = build_Load_in_rl(row,f)  
                l_blocks.append(load)
                l_states.extend(states)
   
    return l_blocks, l_states