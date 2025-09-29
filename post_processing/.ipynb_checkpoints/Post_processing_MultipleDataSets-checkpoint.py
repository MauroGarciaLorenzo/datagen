#!/usr/bin/env python
# coding: utf-8

# # Data Generation Post-processing -- Data Sets Comparison

# In[1]:


from matplotlib import offsetbox
from collections import defaultdict
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
import os
from utils_pp_standalone import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA, KernelPCA
#import seaborn as sns
from scipy.stats import pointbiserialr
from collections import defaultdict
import re


# In[2]:


plt.rcParams.update({"figure.figsize": [8, 4],
                     "text.usetex": True,
                     "font.family": "serif",
                     "font.serif": "Computer Modern",
                     "axes.labelsize": 20,
                     "axes.titlesize": 20,
                     'figure.titlesize': 20,
                     "legend.fontsize": 20,
                     "xtick.labelsize": 16,
                     "ytick.labelsize": 16,
                     "savefig.dpi": 130,
                    'legend.fontsize': 20,
                     'legend.handlelength': 2,
                     'legend.loc': 'upper right'})


# ## Data description

# In[7]:


path = '../results/'

dir_names=[dir_name for dir_name in os.listdir(path) if '_4909' in dir_name]# or '_3518' in dir_name]# if dir_name.startswith('datagen') and 'zip' not in dir_name]#
results_dataframes_datasets=dict()
dataset_ID_list=[]

df_op = 'case_df_op' #df_op
for dir_name in dir_names:
    path_results = os.path.join(path, dir_name)

    dataset_ID = dir_name[-5:]
    dataset_ID_list.append(dataset_ID)
    results_dataframes_datasets[dataset_ID], csv_files = open_csv(
        path_results, ['cases_df.csv', df_op+'.csv'])

    perc_stability(results_dataframes_datasets[dataset_ID][df_op], dir_name)
      


# **Data Set 7664**:
# - Entropy <u>is not</u> considered as cut-off criteria in the data generation process
# - Fasibility ratio <u>is</u> considered as cut-off criteria in the data generation process, rel_tolerance = 0.01 (if feasibility ratio in the cell is less then 0.01 the cell dies? correct?)
# - max_depth = 7
# 
# **Data Set 7665**:
# - Entropy <u>is</u> considered as cut-off criteria in the data generation process, (what was the threshold for entropy decrese to kill the cell?)
# - Fasibility ratio <u>is also</u> considered as cut-off criteria in the data generation process, rel_tolerance = 0.01 (if feasibility ratio in the cell is less then 0.01 the cell dies? correct?)
# - max_depth = 7
# 

# In[8]:


columns_in_df_DS = dict()

for dataset_ID in dataset_ID_list:
    columns_in_df = dict()
    for key, item in results_dataframes_datasets[dataset_ID].items():
        columns_in_df[key] = list(results_dataframes_datasets[dataset_ID][key].columns)
    columns_in_df_DS[dataset_ID]=columns_in_df


# In[9]:


def print_columns_groups(key, columns_list):
    # Group columns by the alphabetic prefix
    groups = defaultdict(list)
    for col in columns_list:
        match = re.match(r"([A-Za-z_]+)", col)  # extract the prefix before any digit
        prefix = match.group(1) if match else col
        groups[prefix].append(col)
    
    print(key+':\n')

    # Print grouped columns
    for prefix, cols in groups.items():
        print(f"{prefix}: {cols[0]},...,{cols[-1]}; N. elements: {len(cols)}\n")
        
for key, item in columns_in_df_DS[dataset_ID_list[0]].items():
    print_columns_groups(key, item)


# ### General Description
# 
# - cases_df: sampled quantities
# 
#     - p_X_Var, q_X_Var: N. elements: 53, P and Q in each generation unit from each typer of element (X = [sg, cig, g_for, g_fol]) (=0 where the element is not present) [MW, Mvar]
# 
#     - perc_g_for_Var: N. elements: 1
# 
#     - tau_droop_f_X_,tau_droop_u_X_ : N. elements: 53, taus de cada convertidor (X = [gfor, gfol])
# 
#     - p_load_Var, q_load_Var: N. elements: 91, P and Q of each load [MW, Mvar]
# 
#     - case_id: N. elements: 1
# 
#     - Stability: N. elements: 1 =1: stable, =0 unstable, = -1: unfeasible, =-2 feasible but to a point out of the sampling cell
#  
# - df_op: after power flow quantities
# 
#     - V, theta:  N. elements: 118
#     - P_SG, Q_SG, Sn_SG: N. elements: 47, P and Q injected by SG in p.u. and installed capacity in MVA (of the SGs effectively present in the grid)
#     - P_X, Q_X, Sn_X: N. elements: 18, P and Q injected by X in p.u. and installed capacity in MVA. X is GFOL or GFOR, if the converter is not included in the grid there is a NaN value.
# 
#     - PL, QL: N. elements: 91  P and Q of each load p.u.
# 
#     - case_id: case_id,...,case_id; N. elements: 1
# 
#     - Stability: N. elements: 1 =1: stable, =0 unstable, = -1: unfeasible, =-2 feasible but to a point out of the sampling cell
# 

# In[10]:


# %% ---- FILL NAN VALUES WITH NULL ---

for dataset_ID in dataset_ID_list:

    results_dataframes_datasets[dataset_ID][df_op] = results_dataframes_datasets[dataset_ID][df_op].fillna(0)
    
    # ---- FIX VALUES ----
    
    Sn_cols = [col for col in results_dataframes_datasets[dataset_ID][df_op]
               if col.startswith('Sn')]
    results_dataframes_datasets[dataset_ID][df_op][Sn_cols] = results_dataframes_datasets[dataset_ID][df_op][Sn_cols]/100 #p.u. system base 100 MVA
    
    theta_cols = [col for col in results_dataframes_datasets[dataset_ID][df_op]
                  if col.startswith('theta')]
    # Adjust angles greater than 180°
    results_dataframes_datasets[dataset_ID][df_op][theta_cols] = results_dataframes_datasets[dataset_ID][df_op][theta_cols] - \
        (results_dataframes_datasets[dataset_ID][df_op][theta_cols] > 180) * 360
    
    results_dataframes_datasets[dataset_ID][df_op][theta_cols] = results_dataframes_datasets[dataset_ID][df_op][theta_cols] * np.pi/180
    
    # add total demand variables
    PL_cols = [
        col for col in results_dataframes_datasets[dataset_ID][df_op].columns if col.startswith('PL')]
    results_dataframes_datasets[dataset_ID][df_op]['PD'] = results_dataframes_datasets[dataset_ID][df_op][PL_cols].sum(
        axis=1)
    
    QL_cols = [
        col for col in results_dataframes_datasets[dataset_ID][df_op].columns if col.startswith('QL')]
    results_dataframes_datasets[dataset_ID][df_op]['QD'] = results_dataframes_datasets[dataset_ID][df_op][QL_cols].sum(
        axis=1)


# ### Data Set Composition

# In[11]:


for dataset_ID in dataset_ID_list:

    perc_stability(results_dataframes_datasets[dataset_ID][df_op], dir_name)


# In[12]:


# %% ---- SELECT ONLY FEASIBLE CASES ----
case_id_feasible_DS = dict()
case_id_Unfeasible_DS = dict()
case_id_Unfeasible1_DS = dict()
case_id_Unfeasible2_DS = dict()

for dataset_ID in dataset_ID_list:

    # from data frame with power flow results: df_op
    results_dataframes_datasets[dataset_ID]['df_op_feasible'] = results_dataframes_datasets[dataset_ID][df_op].query(
        'Stability >= 0')
    
    # from data frame with sampled quantities: cases_df
    results_dataframes_datasets[dataset_ID]['cases_df_feasible'] = results_dataframes_datasets[dataset_ID]['cases_df'].query(
        'Stability >= 0')
    case_id_feasible_DS[dataset_ID] = list(results_dataframes_datasets[dataset_ID]['df_op_feasible']['case_id'])
    
    # ---- SELECT ONLY UNFEASIBLE CASES (from data frame with sampled quantities: cases_df)----
    
    results_dataframes_datasets[dataset_ID]['cases_df_unfeasible'] = results_dataframes_datasets[dataset_ID]['cases_df'].query('Stability < 0')
    results_dataframes_datasets[dataset_ID]['cases_df_unfeasible_1'] = results_dataframes_datasets[dataset_ID]['cases_df'].query('Stability == -1')
    results_dataframes_datasets[dataset_ID]['cases_df_unfeasible_2'] = results_dataframes_datasets[dataset_ID]['cases_df'].query('Stability == -2')
    
    case_id_Unfeasible_DS[dataset_ID] = list(results_dataframes_datasets[dataset_ID]['cases_df_unfeasible']['case_id'])
    case_id_Unfeasible1_DS[dataset_ID] = list(results_dataframes_datasets[dataset_ID]['cases_df_unfeasible_1']['case_id'])
    case_id_Unfeasible2_DS[dataset_ID] = list(results_dataframes_datasets[dataset_ID]['cases_df_unfeasible_2']['case_id'])


# In[13]:


def create_dimensions_caseid_df(df_dict, df_name, list_of_var, list_of_var_names, Sbase=1):
    dimensions_caseid = pd.DataFrame(columns = list_of_var_names + ['case_id','Stability'])
    for name_dim in  list_of_var_names:
        dimensions_caseid[name_dim] =  df_dict[df_name][list_of_var[name_dim]].sum(axis=1)*Sbase
    dimensions_caseid['case_id'] =  df_dict[df_name]['case_id']
    dimensions_caseid['Stability'] = list(df_dict[df_name]['Stability'])

    return dimensions_caseid


# In[14]:


dimensions_caseid_feasible_DS=dict()
dimensions_caseid_feasible_sampled_DS=dict()
dimensions_caseid_unfeasible_DS=dict()
dimensions_caseid_unfeasible1_DS=dict()
dimensions_caseid_unfeasible2_DS=dict()

for dataset_ID in dataset_ID_list:
    
    p_sg_var=[var for var in results_dataframes_datasets[dataset_ID]['df_op_feasible'].columns if var.startswith('P_SG')]
    p_cig_var=[var for var in results_dataframes_datasets[dataset_ID]['df_op_feasible'].columns if var.startswith('P_GFOR') or var.startswith('P_GFOL')]
    p_gfor_var=[var for var in results_dataframes_datasets[dataset_ID]['df_op_feasible'].columns if var.startswith('P_GFOR')]
    p_gfol_var=[var for var in results_dataframes_datasets[dataset_ID]['df_op_feasible'].columns if var.startswith('P_GFOL')]

    list_of_var = dict()
    list_of_var['p_sg'] =  p_sg_var
    list_of_var['p_cig'] =  p_cig_var
    list_of_var['p_gfor'] =  p_gfor_var
    list_of_var['p_gfol'] =  p_gfol_var
    
    dimensions_caseid_feasible_DS[dataset_ID] = create_dimensions_caseid_df(results_dataframes_datasets[dataset_ID], 'df_op_feasible', list_of_var, ['p_sg', 'p_cig', 'p_gfor','p_gfol'], Sbase=100)
    
    p_sg_var=[var for var in results_dataframes_datasets[dataset_ID]['cases_df_unfeasible'].columns if var.startswith('p_sg')]
    p_cig_var=[var for var in results_dataframes_datasets[dataset_ID]['cases_df_unfeasible'].columns if var.startswith('p_cig')]
    p_gfor_var=[var for var in results_dataframes_datasets[dataset_ID]['cases_df_unfeasible'].columns if var.startswith('P_GFOR')]
    p_gfol_var=[var for var in results_dataframes_datasets[dataset_ID]['cases_df_unfeasible'].columns if var.startswith('P_GFOL')]
    
    list_of_var = dict()
    list_of_var['p_sg'] =  p_sg_var
    list_of_var['p_cig'] =  p_cig_var
    list_of_var['p_gfor'] =  p_gfor_var
    list_of_var['p_gfol'] =  p_gfol_var
    
    dimensions_caseid_feasible_sampled_DS[dataset_ID] = create_dimensions_caseid_df(results_dataframes_datasets[dataset_ID], 'cases_df_unfeasible', list_of_var, ['p_sg', 'p_cig', 'p_gfor','p_gfol'])
    dimensions_caseid_feasible_sampled_DS[dataset_ID] = create_dimensions_caseid_df(results_dataframes_datasets[dataset_ID], 'cases_df_unfeasible_1', list_of_var, ['p_sg', 'p_cig', 'p_gfor','p_gfol'])
    dimensions_caseid_feasible_sampled_DS[dataset_ID] = create_dimensions_caseid_df(results_dataframes_datasets[dataset_ID], 'cases_df_unfeasible_2', list_of_var, ['p_sg', 'p_cig', 'p_gfor','p_gfol'])

    
# In[15]:


for dataset_ID in dataset_ID_list:

    fig, ax = plt.subplots()
    ax.scatter(dimensions_caseid_unfeasible1_DS[dataset_ID]['p_cig'], dimensions_caseid_unfeasible1_DS[dataset_ID]['p_sg'],color='silver', label='Unfeasable OP (-1)')
    ax.scatter(dimensions_caseid_unfeasible2_DS[dataset_ID]['p_cig'], dimensions_caseid_unfeasible2_DS[dataset_ID]['p_sg'],color='k', label='Unfeasable OP (-2)')
    ax.scatter(dimensions_caseid_feasible_sampled_DS[dataset_ID]['p_cig'], dimensions_caseid_feasible_sampled_DS[dataset_ID]['p_sg'], label='Feasable OP')
    ax.scatter(dimensions_caseid_feasible_DS[dataset_ID]['p_cig'], dimensions_caseid_feasible_DS[dataset_ID]['p_sg'], label='Feasable PF')
    ax.set_xlabel('$P_{CIG}$ [MW]')
    ax.set_ylabel('$P_{SG}$ [MW]')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title('Data Set'+dataset_ID)
    fig.tight_layout()


# In[16]:


for dataset_ID in dataset_ID_list:

    fig, ax = plt.subplots()
    ax.scatter(dimensions_caseid_unfeasible_DS[dataset_ID]['p_cig'], dimensions_caseid_unfeasible_DS[dataset_ID]['p_sg'],color='silver', label='Unfeasable OP')
    ax.scatter(dimensions_caseid_feasible_DS[dataset_ID].query('Stability ==0')['p_cig'], dimensions_caseid_feasible_DS[dataset_ID].query('Stability ==0')['p_sg'], color='r',label='Unstable PF')
    ax.scatter(dimensions_caseid_feasible_DS[dataset_ID].query('Stability ==1')['p_cig'], dimensions_caseid_feasible_DS[dataset_ID].query('Stability ==1')['p_sg'], color='g', label='Stable PF')
    ax.set_xlabel('$P_{CIG}$ [MW]')
    ax.set_ylabel('$P_{SG}$ [MW]')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title('Data Set'+dataset_ID)
    fig.tight_layout()

#%%

from mpl_toolkits.mplot3d import Axes3D

for dataset_ID in dataset_ID_list:
    # Create 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dimensions_caseid_feasible_DS[dataset_ID].query('Stability ==0')['p_sg'], 
               dimensions_caseid_feasible_DS[dataset_ID].query('Stability ==0')['p_gfor'],
               dimensions_caseid_feasible_DS[dataset_ID].query('Stability ==0')['p_gfol'], color='r',label='Unstable OP', marker='o')
    
    ax.scatter(dimensions_caseid_feasible_DS[dataset_ID].query('Stability ==1')['p_sg'], 
               dimensions_caseid_feasible_DS[dataset_ID].query('Stability ==1')['p_gfor'],
               dimensions_caseid_feasible_DS[dataset_ID].query('Stability ==1')['p_gfol'], color='g',label='Stable OP', marker='o')
    
    # Labels
    ax.set_xlabel('$P_{SG}$ [MW]', labelpad =10)
    ax.set_ylabel('$P_{GFOR}$ [MW]', labelpad =10)
    ax.set_zlabel('$P_{GFOL}$ [MW]', labelpad =10)
    
    ax.view_init(elev=10,azim=50)
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    fig.subplots_adjust(left=0.0, right=0.8, top=1, bottom=0.1)
    
# Mesh obtained from parsing the logs file of the data generator process: the mesh shows the cell splitting process. It is obtained from the parsing_dimensions.py code.
# 
# Plot the mesh on top of the OPs scatter plot.

# In[17]:


mesh_df_DS= dict()
for dataset_ID in dataset_ID_list:

    mesh_df_DS[dataset_ID]= pd.read_excel('mesh'+dataset_ID+'.xlsx')


# In[18]:


for dataset_ID in dataset_ID_list:

    fig, ax = plt.subplots()
    ax.scatter(dimensions_caseid_unfeasible_DS[dataset_ID]['p_cig'], dimensions_caseid_unfeasible_DS[dataset_ID]['p_sg'],color='silver', label='Unfeasable OP')
    ax.scatter(dimensions_caseid_feasible_DS[dataset_ID].query('Stability ==0')['p_cig'], dimensions_caseid_feasible_DS[dataset_ID].query('Stability ==0')['p_sg'], color='r',label='Unstable PF')
    ax.scatter(dimensions_caseid_feasible_DS[dataset_ID].query('Stability ==1')['p_cig'], dimensions_caseid_feasible_DS[dataset_ID].query('Stability ==1')['p_sg'], color='g', label='Stable PF')
    ax.set_xlabel('$P_{CIG}$ [MW]')
    ax.set_ylabel('$P_{SG}$ [MW]')
    plot_mesh(mesh_df_DS[dataset_ID], ax)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title('Data Set'+dataset_ID)
    fig.tight_layout()


# In[19]:


for dataset_ID in dataset_ID_list:

    fig, ax = plt.subplots()
    ax.scatter(dimensions_caseid_unfeasible_DS[dataset_ID]['p_cig'], dimensions_caseid_unfeasible_DS[dataset_ID]['p_sg'],color='silver', label='Unfeasable OP')
    ax.scatter(dimensions_caseid_feasible_sampled_DS[dataset_ID].query('Stability ==0')['p_cig'], dimensions_caseid_feasible_sampled_DS[dataset_ID].query('Stability ==0')['p_sg'], color='r',label='Unstable Sampled OP')
    ax.scatter(dimensions_caseid_feasible_sampled_DS[dataset_ID].query('Stability ==1')['p_cig'], dimensions_caseid_feasible_sampled_DS[dataset_ID].query('Stability ==1')['p_sg'], color='g', label='Stable Sampled OP')
    ax.set_xlabel('$P_{CIG}$ [MW]')
    ax.set_ylabel('$P_{SG}$ [MW]')
    plot_mesh(mesh_df_DS[dataset_ID], ax)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title('Data Set'+dataset_ID)
    fig.tight_layout()


# Dataframes with the cases_id, the exploration depth at which they have been evaluated and the corresponding cell name (as in the cell_info.csv file).
# It is obtained from the parsing_dimensions.py code.

# In[18]:


df_depth_DS = dict()
for dataset_ID in dataset_ID_list:
    
    df_depth_DS[dataset_ID]= pd.read_excel('cases_id_depth'+dataset_ID+'.xlsx')


# In[20]:


df_depth_DS[dataset_ID_list[0]].query('Depth == 0')['case_id']


# In[23]:


df_feasibility_balancing_DS=dict()

for dataset_ID in dataset_ID_list:

    df_feasibility_balancing_DS[dataset_ID]= pd.DataFrame(columns=['depth','feasibility','cumulative_feasibility','feasiblity_no_2','balance','cumulative_balancing'])
    cum_case_id_depth=[]
    for idx, depth in enumerate(np.sort(df_depth_DS[dataset_ID]['Depth'].unique())):
        df_feasibility_balancing_DS[dataset_ID].loc[idx, 'depth']=depth
        case_id_depth = df_depth_DS[dataset_ID].query('Depth == @depth')['case_id']
        cum_case_id_depth.extend(case_id_depth)
        feas_case_id_depth = list(set(case_id_depth) & set(case_id_feasible_DS[dataset_ID]))
        cum_feas_case_id_depth = list(set(cum_case_id_depth) & set(case_id_feasible_DS[dataset_ID]))
        
        df_feasibility_balancing_DS[dataset_ID].loc[idx, 'feasibility']= len(feas_case_id_depth)/len(case_id_depth) 
        df_feasibility_balancing_DS[dataset_ID].loc[idx, 'cumulative_feasibility']= len(cum_feas_case_id_depth)/len(cum_case_id_depth) 
    
        feas_stab_depth = len(results_dataframes_datasets[dataset_ID]['cases_df_feasible'].query('case_id == @feas_case_id_depth and Stability ==1'))
        cum_feas_stab_case_id_depth = len(results_dataframes_datasets[dataset_ID]['cases_df_feasible'].query('case_id == @cum_feas_case_id_depth and Stability ==1'))

        if len(feas_case_id_depth) !=0:
            df_feasibility_balancing_DS[dataset_ID].loc[idx, 'balance']= feas_stab_depth/len(feas_case_id_depth) 
        else:
             df_feasibility_balancing_DS[dataset_ID].loc[idx, 'balance'] = 0
        
        if len(cum_feas_case_id_depth) !=0:
            df_feasibility_balancing_DS[dataset_ID].loc[idx, 'cumulative_balancing']= cum_feas_stab_case_id_depth/len(cum_feas_case_id_depth) 
        else:
            df_feasibility_balancing_DS[dataset_ID].loc[idx, 'cumulative_balancing']= 0
    


# In[24]:


df_feasibility_balancing_DS[dataset_ID_list[0]]


# In[25]:


import matplotlib.pyplot as plt

# Assuming your DataFrame is called df
# Plot 1: Feasibility
cls=['b','r']
fig, ax = plt.subplots()

for idx, dataset_ID in enumerate(dataset_ID_list):
    ax.plot(df_feasibility_balancing_DS[dataset_ID]['depth'], df_feasibility_balancing_DS[dataset_ID]['feasibility'], marker='o',color=cls[idx],linestyle='--', label='Feasibility'+dataset_ID)
    ax.plot(df_feasibility_balancing_DS[dataset_ID]['depth'], df_feasibility_balancing_DS[dataset_ID]['cumulative_feasibility'], marker='s',color=cls[idx], label='Cumulative Feasibility'+dataset_ID)
ax.set_xlabel('Depth')
ax.set_ylabel('Feasibility')
ax.set_title('Feasibility vs Depth')
ax.grid(True)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()


# very large difference in feasibility ratio when exploring at depth >=4 

# In[26]:


# Plot 2: Balancing

cls=['b','r']
fig, ax = plt.subplots()

for idx, dataset_ID in enumerate(dataset_ID_list):
    ax.plot(df_feasibility_balancing_DS[dataset_ID]['depth'], df_feasibility_balancing_DS[dataset_ID]['balance'], marker='o',color=cls[idx],linestyle='--', label='Balance'+dataset_ID)
    ax.plot(df_feasibility_balancing_DS[dataset_ID]['depth'], df_feasibility_balancing_DS[dataset_ID]['cumulative_balancing'], marker='s',color=cls[idx], label='Cumulative Balance'+dataset_ID)
ax.set_xlabel('Depth')
ax.set_ylabel('Balance')
ax.set_title('Balance vs Depth')
ax.grid(True)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()


# very large difference in classes balance ratio since the beginning of the exporation, then both tend to 0.5 more or less (that is ok)

# In[55]:


max_depth= df_depth_DS[dataset_ID_list[0]]['Depth'].max()#max(df_depth_DS[dataset_ID_list[0]]['Depth'].max(),df_depth_DS[dataset_ID_list[1]]['Depth'].max())
fig, axes = plt.subplots(nrows=max_depth+1,ncols=len(dataset_ID_list), figsize=(10,15))
for idx_DS, dataset_ID in enumerate(dataset_ID_list):
    
    for idx, depth in enumerate(np.sort(df_depth_DS[dataset_ID]['Depth'].unique())):
        case_id_depth = df_depth_DS[dataset_ID].query('Depth == @depth')['case_id']
        feas_case_id_depth = list(set(case_id_depth) & set(case_id_feasible_DS[dataset_ID]))
        unfeas_case_id_depth = list(set(case_id_depth) & set(case_id_Unfeasible_DS[dataset_ID]))

        try:
            ax = axes[idx,idx_DS]
        except:
            ax = axes[idx]
        ax.scatter(dimensions_caseid_unfeasible1_DS[dataset_ID].query('case_id ==@unfeas_case_id_depth')['p_cig'], dimensions_caseid_unfeasible1_DS[dataset_ID].query('case_id ==@unfeas_case_id_depth')['p_sg'],color='silver', label='Unfeasable OP')
        ax.scatter(dimensions_caseid_unfeasible2_DS[dataset_ID].query('case_id ==@unfeas_case_id_depth')['p_cig'], dimensions_caseid_unfeasible2_DS[dataset_ID].query('case_id ==@unfeas_case_id_depth')['p_sg'],color='blue', label='Unfeasable OP')
        ax.scatter(dimensions_caseid_feasible_DS[dataset_ID].query('Stability ==0 and case_id ==@feas_case_id_depth')['p_cig'], dimensions_caseid_feasible_DS[dataset_ID].query('Stability ==0 and case_id ==@feas_case_id_depth')['p_sg'], color='r',label='Unstable OP')
        ax.scatter(dimensions_caseid_feasible_DS[dataset_ID].query('Stability ==1 and case_id ==@feas_case_id_depth')['p_cig'], dimensions_caseid_feasible_DS[dataset_ID].query('Stability ==1 and case_id ==@feas_case_id_depth')['p_sg'], color='g',label='Stable OP')

        ax.set_xlabel('$P_{CIG}$ [MW]')
        ax.set_ylabel('$P_{SG}$ [MW]')

        plot_mesh(mesh_df_DS[dataset_ID], ax)
        # try:
        #     plot_mesh(mesh_df_DS[dataset_ID].query('depth == @depth'), ax)
        # except:
        #     continue
        #plt.legend()
        ax.set_title('Data Set'+dataset_ID+' depth ='+str(depth))
        ax.set_xlim([100,620])
        ax.set_ylim([0,300])


# In[60]:


max_depth= df_depth_DS[dataset_ID_list[0]]['Depth'].max()#max(df_depth_DS[dataset_ID_list[0]]['Depth'].max(),df_depth_DS[dataset_ID_list[1]]['Depth'].max())
fig, axes = plt.subplots(nrows=max_depth+1,ncols=len(dataset_ID_list), figsize=(10,15))
for idx_DS, dataset_ID in enumerate(dataset_ID_list):
    
    for idx, depth in enumerate(np.sort(df_depth_DS[dataset_ID]['Depth'].unique())):
        case_id_depth = df_depth_DS[dataset_ID].query('Depth == @depth')['case_id']
        feas_case_id_depth = list(set(case_id_depth) & set(case_id_feasible_DS[dataset_ID]))
        unfeas_case_id_depth = list(set(case_id_depth) & set(case_id_Unfeasible_DS[dataset_ID]))

        try:
            ax = axes[idx,idx_DS]
        except:
            ax = axes[idx]
        ax.scatter(dimensions_caseid_unfeasible1_DS[dataset_ID].query('case_id ==@unfeas_case_id_depth')['p_cig'], dimensions_caseid_unfeasible1_DS[dataset_ID].query('case_id ==@unfeas_case_id_depth')['p_sg'],color='silver', label='Unfeasable OP')
        ax.scatter(dimensions_caseid_unfeasible2_DS[dataset_ID].query('case_id ==@unfeas_case_id_depth')['p_cig'], dimensions_caseid_unfeasible2_DS[dataset_ID].query('case_id ==@unfeas_case_id_depth')['p_sg'],color='blue', label='Unfeasable OP')
        ax.scatter(dimensions_caseid_feasible_sampled_DS[dataset_ID].query('Stability ==0 and case_id ==@feas_case_id_depth')['p_cig'], dimensions_caseid_feasible_sampled_DS[dataset_ID].query('Stability ==0 and case_id ==@feas_case_id_depth')['p_sg'], color='r',label='Unstable OP')
        ax.scatter(dimensions_caseid_feasible_sampled_DS[dataset_ID].query('Stability ==1 and case_id ==@feas_case_id_depth')['p_cig'], dimensions_caseid_feasible_sampled_DS[dataset_ID].query('Stability ==1 and case_id ==@feas_case_id_depth')['p_sg'], color='g',label='Stable OP')

        ax.set_xlabel('$P_{CIG}$ [MW]')
        ax.set_ylabel('$P_{SG}$ [MW]')

        plot_mesh(mesh_df_DS[dataset_ID], ax)
        # try:
        #     plot_mesh(mesh_df_DS[dataset_ID].query('depth <= @depth'), ax)
        # except:
        #     continue
        #plt.legend()
        ax.set_title('Data Set'+dataset_ID+' depth ='+str(depth))
        ax.set_xlim([100,620])        
        ax.set_ylim([0,300])

#%%

import networkx as nx

leaf_name = sorted(df_depth_DS[dataset_ID]['CellName'].unique(), key=len)
parent = [leaf[:-2] for leaf in leaf_name]
parent = [None if p=='' else p for p in parent]
depth = [len(leaf.replace('.',''))-1 for leaf in leaf_name]

values=[]
for leaf in leaf_name:
    case_id_leaf = df_depth_DS[dataset_ID].query('CellName == @leaf')['case_id']
    feas_case_id_leaf = list(set(case_id_leaf) & set(case_id_feasible_DS[dataset_ID]))
    if len(feas_case_id_leaf)>0:
        feas_stab_leaf = len(results_dataframes_datasets[dataset_ID]['cases_df_feasible'].query('case_id == @feas_case_id_leaf and Stability ==1'))
        values.append(np.round(feas_stab_leaf/len(feas_case_id_leaf),2)) 
    else:
        values.append(0)
# Example dataframe with parent column
df = pd.DataFrame({
    "leaf_name": leaf_name,
    "parent":    parent,
    "depth":     depth,
    "value":     values
})


# Replace missing parents with "root" itself (or leave out edges for root)
# Here: drop edges where parent is None
edges = df.dropna(subset=["parent"])

# Create directed graph
G = nx.DiGraph()

# Add nodes and edges
for _, row in df.iterrows():
    G.add_node(str(row["leaf_name"]), value=row["value"])

for _, row in edges.iterrows():
    G.add_edge(str(row["parent"]), str(row["leaf_name"]))

# Layout with graphviz
pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

# Draw
plt.figure(figsize=(8,6))
nx.draw(G, pos, with_labels=False, node_size=1500, node_color="lightblue", arrows=False)

# Labels with value
labels = {n: f"{n}\n{d['value']}" for n, d in G.nodes(data=True)}
nx.draw_networkx_labels(G, pos, labels=labels)

plt.show()

# In[52]:


plot_mesh(mesh_df_DS[dataset_ID].query('depth == 3'))


# In[ ]:


From depth >=4 **Data Set 7665** focuses on regions with low PSG and PIBR that **Data Set 7664** discarded because totally unfeasible


# ## Comparison of Models Accuracy
# Accuracy is obtained by performing:
# - k-fold cross validation on data sets subsets (adding samples generated at each exploration depth)
# - using XGBoosting (no hyperparameters tuning)
# - after data cleaning and correlated variables removal (by hierarchical clustering approach)
#   
# [see Post_processing notebook]

# In[63]:


scores_df_uncorr_HierCl_DS=dict()
for idx_DS, dataset_ID in enumerate(dataset_ID_list):
    scores_df_uncorr_HierCl_DS[dataset_ID]= pd.read_excel('scores_df_uncorr_HierCl_xgb'+dataset_ID+'.xlsx')


# In[65]:


#%%
fig, ax = plt.subplots()
cls=['b','r']
for idx_DS, dataset_ID in enumerate(dataset_ID_list):
    ax.errorbar(scores_df_uncorr_HierCl_DS[dataset_ID]['Depth'], scores_df_uncorr_HierCl_DS[dataset_ID]['score_mean'], yerr=scores_df_uncorr_HierCl_DS[dataset_ID]['score_std'], fmt='-o', capsize=5, color=cls[idx_DS], ecolor='black', elinewidth=1.5, label = 'Data Set'+dataset_ID)

ax.set_xlabel('Depth')
ax.set_ylabel('Mean accuracy $\pm$ std')
ax.grid()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()
#plt.savefig('scores_vs_depth__df_uncorr_var_HierCl_xgb.pdf')#, format='pdf')
#plt.savefig('scores_vs_depth__df_uncorr_var_HierCl_xgb.png')#, format='png')


# In[68]:


scores_df_uncorr_HierCl_DS[dataset_ID_list[1]]


# In[ ]:




