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
                     "xtick.labelsize": 20,
                     "ytick.labelsize": 20,
                     "savefig.dpi": 130,
                    'legend.fontsize': 20,
                     'legend.handlelength': 2,
                     'legend.loc': 'upper right'})


# ## Data description

# In[7]:


#path = '../results/'
path = 'D:/'
dir_names=[dir_name for dir_name in os.listdir(path) if '_2732' in dir_name   and 'zip' not in dir_name or '_2862' in dir_name  and 'zip' not in dir_name]# if dir_name.startswith('datagen') and 'zip' not in dir_name]#
# '_5933' in dir_name and 'zip' not in dir_name
#dir_names=[dir_names[0],dir_names[2]]
print(dir_names)
#dir_names=[dir_name for dir_name in os.listdir(path) if '_4909' in dir_name]# or '_3518' in dir_name]# if dir_name.startswith('datagen') and 'zip' not in dir_name]#
#%%
results_dataframes_datasets=dict()
dataset_ID_list=[]#'complete_sensitivity','partial_sensitivity']

df_op = 'df_op'#'case_df_op' #df_op
for idx, dir_name in enumerate(dir_names):
    path_results = os.path.join(path, dir_name)

    dataset_ID = dir_name[-5:] #dataset_ID_list[idx]#
    dataset_ID_list.append(dataset_ID)
    results_dataframes_datasets[dataset_ID], csv_files = open_csv(
        path_results, ['cases_df.csv', df_op+'.csv','dims_df.csv', 'cell_info.csv'])

    perc_stability(results_dataframes_datasets[dataset_ID]['cases_df'], dir_name)
      
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


# %% ---- FILL NAN VALUES WITH NULL ---

for dataset_ID in dataset_ID_list:

    results_dataframes_datasets[dataset_ID][df_op] = results_dataframes_datasets[dataset_ID][df_op].fillna(0)
    
    # ---- FIX VALUES ----
    
    Sn_cols = [col for col in results_dataframes_datasets[dataset_ID][df_op]
               if col.startswith('Sn')]
    results_dataframes_datasets[dataset_ID][df_op][Sn_cols] = results_dataframes_datasets[dataset_ID][df_op][Sn_cols]/100 #p.u. system base 100 MVA
    
    # theta_cols = [col for col in results_dataframes_datasets[dataset_ID][df_op]
    #               if col.startswith('theta')]
    # # Adjust angles greater than 180°
    # results_dataframes_datasets[dataset_ID][df_op][theta_cols] = results_dataframes_datasets[dataset_ID][df_op][theta_cols] - \
    #     (results_dataframes_datasets[dataset_ID][df_op][theta_cols] > 180) * 360
    
    # results_dataframes_datasets[dataset_ID][df_op][theta_cols] = results_dataframes_datasets[dataset_ID][df_op][theta_cols] * np.pi/180
    
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
    
    case_id_feasible_DS[dataset_ID] = list(results_dataframes_datasets[dataset_ID]['cases_df_feasible']['case_id'])
    
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
        if name_dim in ['p_sg','p_cig']:
            dimensions_caseid[name_dim]=dimensions_caseid[name_dim]/1000
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
    taus_var = [var for var in  results_dataframes_datasets[dataset_ID]['dims_df'].columns if var.startswith('tau')]

    list_of_var = dict()
    list_of_var['p_sg'] =  p_sg_var
    list_of_var['p_cig'] =  p_cig_var
    list_of_var['p_gfor'] =  p_gfor_var
    list_of_var['p_gfol'] =  p_gfol_var
    
    dimensions_caseid_feasible_DS[dataset_ID] = create_dimensions_caseid_df(results_dataframes_datasets[dataset_ID], 'df_op_feasible', list_of_var, ['p_sg', 'p_cig', 'p_gfor','p_gfol'], Sbase=100)
    case_id_feasible = case_id_feasible_DS[dataset_ID]
    dimensions_caseid_feasible_DS[dataset_ID][taus_var] = results_dataframes_datasets[dataset_ID]['dims_df'].query('case_id == @case_id_feasible')[taus_var]
    dimensions_caseid_feasible_DS[dataset_ID]['perc_g_for'] = dimensions_caseid_feasible_DS[dataset_ID]['p_gfor']/dimensions_caseid_feasible_DS[dataset_ID]['p_cig']

    p_sg_var=[var for var in results_dataframes_datasets[dataset_ID]['cases_df_unfeasible'].columns if var.startswith('p_sg')]
    p_cig_var=[var for var in results_dataframes_datasets[dataset_ID]['cases_df_unfeasible'].columns if var.startswith('p_cig')]
    p_gfor_var=[var for var in results_dataframes_datasets[dataset_ID]['cases_df_unfeasible'].columns if var.startswith('p_g_for')]
    p_gfol_var=[var for var in results_dataframes_datasets[dataset_ID]['cases_df_unfeasible'].columns if var.startswith('p_g_fol')]
    
    list_of_var = dict()
    list_of_var['p_sg'] =  p_sg_var
    list_of_var['p_cig'] =  p_cig_var
    list_of_var['p_gfor'] =  p_gfor_var
    list_of_var['p_gfol'] =  p_gfol_var
    
    dimensions_caseid_feasible_sampled_DS[dataset_ID] = create_dimensions_caseid_df(results_dataframes_datasets[dataset_ID], 'cases_df_feasible', list_of_var, ['p_sg', 'p_cig', 'p_gfor','p_gfol'])
    dimensions_caseid_unfeasible_DS[dataset_ID] = create_dimensions_caseid_df(results_dataframes_datasets[dataset_ID], 'cases_df_unfeasible', list_of_var, ['p_sg', 'p_cig', 'p_gfor','p_gfol'])
    dimensions_caseid_unfeasible1_DS[dataset_ID] = create_dimensions_caseid_df(results_dataframes_datasets[dataset_ID], 'cases_df_unfeasible_1', list_of_var, ['p_sg', 'p_cig', 'p_gfor','p_gfol'])
    dimensions_caseid_unfeasible2_DS[dataset_ID] = create_dimensions_caseid_df(results_dataframes_datasets[dataset_ID], 'cases_df_unfeasible_2', list_of_var, ['p_sg', 'p_cig', 'p_gfor','p_gfol'])
    case_id_feasible = case_id_Unfeasible_DS[dataset_ID]
    dimensions_caseid_unfeasible_DS[dataset_ID][taus_var] = results_dataframes_datasets[dataset_ID]['dims_df'].query('case_id == @case_id_feasible')[taus_var]
    dimensions_caseid_unfeasible_DS[dataset_ID]['perc_g_for'] = dimensions_caseid_unfeasible_DS[dataset_ID]['p_gfor']/dimensions_caseid_unfeasible_DS[dataset_ID]['p_cig']

#%%
import matplotlib.pyplot as plt

# Create a horizontal grid of subplots: 1 row, N columns
num_datasets = len(dataset_ID_list)
fig, axes = plt.subplots(1, num_datasets, figsize=(4*num_datasets, 5), sharey=True)

# If only one dataset, axes may not be iterable
if num_datasets == 1:
    axes = [axes]
dataset_ID_title_list = ['TEST \#1','TEST \#2']
for i, dataset_ID in enumerate(dataset_ID_list):
    ax = axes[i]
    dataset_ID_title = dataset_ID_title_list[i]
    
    ax.scatter(dimensions_caseid_unfeasible1_DS[dataset_ID]['p_cig'], dimensions_caseid_unfeasible1_DS[dataset_ID]['p_sg'], color='silver', label='Infeasible OP')
    ax.scatter(dimensions_caseid_unfeasible2_DS[dataset_ID]['p_cig'], dimensions_caseid_unfeasible2_DS[dataset_ID]['p_sg'], color='k', label='Feasible\ndiscarded OP')
    ax.scatter(dimensions_caseid_feasible_sampled_DS[dataset_ID]['p_cig'], dimensions_caseid_feasible_sampled_DS[dataset_ID]['p_sg'], label='Feasible OP')
    ax.scatter(dimensions_caseid_feasible_DS[dataset_ID]['p_cig'], dimensions_caseid_feasible_DS[dataset_ID]['p_sg'], color='#B0E0E6', label='Feasible PF')
    
    ax.set_xlabel('$P_{IBR}$ [GW]')
    if i == 0:
        ax.set_ylabel('$P_{SG}$ [GW]')
    ax.set_title(dataset_ID_title) #'Data Set ' + 

# Create a shared legend below the plots
# Use unique labels from the last plotted axis
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=4, columnspacing=0.1,handletextpad=-0.2)

fig.tight_layout(rect=[0, 0.15, 1, 1])  # Leave space at bottom for legend

plt.show()
plt.savefig('figures_paper/feasible_infeasible_points'+dataset_ID_list[0]+dataset_ID_list[1]+'.pdf', format='pdf')
plt.savefig('figures_paper/feasible_infeasible_points'+dataset_ID_list[0]+dataset_ID_list[1]+'.png',dpi=320)#, format='pdf')


# In[17]:
from pathlib import Path

mesh_df_DS= dict()
path2dataset = path +dir_names[0]
mesh_df_DS[dataset_ID_list[0]]= pd.read_excel(path2dataset+'/mesh'+dataset_ID_list[0]+'.xlsx')

path2dataset = Path(path +dir_names[1])
mesh_df_files = [f for f in path2dataset.iterdir()
           if f.is_file() and f.name.startswith('mesh') and f.suffix in ['.xlsx', '.xls'] and 'p_sg' in f.name and 'p_cig' in f.name]

mesh_df_DS[dataset_ID_list[1]]= pd.read_excel(mesh_df_files[0])

for dataset_ID in dataset_ID_list:
    idxs = mesh_df_DS[dataset_ID]['dimension'] == 'p_sg'
    mesh_df_DS[dataset_ID].loc[idxs, 'lower'] /= 1000
    mesh_df_DS[dataset_ID].loc[idxs, 'upper'] /= 1000

    idxs = mesh_df_DS[dataset_ID]['dimension'] == 'p_cig'
    mesh_df_DS[dataset_ID].loc[idxs, 'lower'] /= 1000
    mesh_df_DS[dataset_ID].loc[idxs, 'upper'] /= 1000

df_sensitivity = pd.read_excel(path+dir_names[1]+'/sensitivity_log'+dataset_ID_list[1]+'.xlsx') 
df_sensitivity['pair'] = df_sensitivity.apply(
    lambda row: tuple(sorted([row['dim1'], row['dim2']])),
    axis=1
)
# Find matching pairs and children cells
coppia = ('p_cig', 'p_sg')
parents = df_sensitivity.query('pair == @coppia')['cell']
coppia_T=('p_sg','p_cig')
parents_T = df_sensitivity.query('pair == @coppia_T')['cell']
parents=pd.concat([parents,parents_T],axis=0)

childs = list(set(
    [cell + suffix for cell in parents for suffix in ['.1', '.2', '.3', '.4']]
)) + ['0']
mesh_df_DS[dataset_ID_list[1]] = mesh_df_DS[dataset_ID_list[1]][mesh_df_DS[dataset_ID_list[1]]['block_id'].isin(childs)]

# In[18]:

import matplotlib.pyplot as plt

# Create a horizontal grid of subplots: 1 row, N columns
num_datasets = len(dataset_ID_list)
fig, axes = plt.subplots(1, num_datasets, figsize=(4*num_datasets, 5), sharey=True)

# If only one dataset, axes may not be iterable
if num_datasets == 1:
    axes = [axes]
dataset_ID_title_list = ['TEST \#1','TEST \#2']
for i, dataset_ID in enumerate(dataset_ID_list):
    ax = axes[i]
    dataset_ID_title = dataset_ID_title_list[i]
    
    ax.scatter(dimensions_caseid_unfeasible_DS[dataset_ID]['p_cig'], dimensions_caseid_unfeasible_DS[dataset_ID]['p_sg'],color='silver', label='Infeasable OP')
    ax.scatter(dimensions_caseid_feasible_DS[dataset_ID].query('Stability ==0')['p_cig'], dimensions_caseid_feasible_DS[dataset_ID].query('Stability ==0')['p_sg'], color='r',label='Unstable PF')
    ax.scatter(dimensions_caseid_feasible_DS[dataset_ID].query('Stability ==1')['p_cig'], dimensions_caseid_feasible_DS[dataset_ID].query('Stability ==1')['p_sg'], color='g', label='Stable PF')
    
    plot_mesh(mesh_df_DS[dataset_ID],  'p_cig', 'p_sg', 'p_cig', 'p_sg', ax)

    ax.set_xlabel('$P_{IBR}$ [MW]')
    if i == 0:
        ax.set_ylabel('$P_{SG}$ [MW]')
    ax.set_title(dataset_ID_title) #'Data Set ' + 

# Create a shared legend below the plots
# Use unique labels from the last plotted axis
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=4, columnspacing=0.1,handletextpad=-0.2)

fig.tight_layout(rect=[0, 0.15, 1, 1])  # Leave space at bottom for legend

plt.show()
plt.savefig('figures_paper/stable_unstable_infeasible_points_and_mesh'+dataset_ID_list[0]+dataset_ID_list[1]+'.pdf', format='pdf')
plt.savefig('figures_paper/stable_unstable_infeasible_points_and_mesh'+dataset_ID_list[0]+dataset_ID_list[1]+'.png',dpi=320)#, format='pdf')


# In[19]:

# Dataframes with the cases_id, the exploration depth at which they have been evaluated and the corresponding cell name (as in the cell_info.csv file).
# It is obtained from the parsing_dimensions.py code.

# In[18]:

df_depth_DS = dict()
for path, dataset_ID in zip(dir_names,dataset_ID_list):
    
    df_depth_DS[dataset_ID]= pd.read_excel('D:/'+path+'/cases_id_depth'+dataset_ID+'.xlsx')
    #df_depth_DS[dataset_ID]= pd.read_excel('D:/'+path+'/cases_id_depthSensitivity.xlsx')

#%%    
for dataset_ID in dataset_ID_list:
   
    df_depth_DS[dataset_ID]= df_depth_DS[dataset_ID].set_index('case_id')
    
    df_depth_DS[dataset_ID] = df_depth_DS[dataset_ID].merge(
    results_dataframes_datasets[dataset_ID][df_op].set_index('case_id')[['Stability']],
    left_index=True,
    right_index=True,
    how='outer')
    df_depth_DS[dataset_ID] = df_depth_DS[dataset_ID].reset_index()

# In[20]:


df_depth_DS[dataset_ID_list[0]].query('Depth == 0')['case_id']

#%%
def calculate_entropy(freqs):
    """Obtain cell entropy from stability and non-stability frequencies.

    :param freqs: two-element list with the frequency (1-based) of stable and
    non-stable cases, respectively
    :return: Entropy
    """
    cell_entropy = 0
    for i in range(len(freqs)):
        if freqs[i] != 0:
            cell_entropy = cell_entropy - freqs[i] * np.log(freqs[i])
    return cell_entropy

# In[23]:


df_feasibility_balancing_DS=dict()

for dataset_ID in dataset_ID_list:

    df_feasibility_balancing_DS[dataset_ID]= pd.DataFrame(columns=['depth','feasibility','cumulative_feasibility','mean_feasiblity_cell','std_feasiblity_cell','balance','cumulative_balancing', 'mean_balance_cell','std_balance_cell', 'mean_entropy_cell', 'std_entropy_cell'
                                                                   'mean_unfeas_1','std_unfeas_1','mean_unfeas_2','std_unfeas_2'])
    
    cases_x_stab_cum = pd.DataFrame()
    tot_cases_cum = pd.DataFrame()
    
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

        tot_cases = df_depth_DS[dataset_ID].query('Depth == @depth').groupby('CellName')[['Stability']].count()
        cases_x_stab = df_depth_DS[dataset_ID].query('Depth == @depth').groupby(['CellName', 'Stability'])[['Stability']].count()
                
        feas_cell = [cases_x_stab.loc[(cell,[0,1]),'Stability'].sum()/tot_cases.loc[cell,'Stability'] if 0 in cases_x_stab.loc[cell].index or 1 in cases_x_stab.loc[cell].index else 0 for cell in tot_cases.index]
        df_feasibility_balancing_DS[dataset_ID].loc[idx, 'mean_feasiblity_cell'] = np.mean(feas_cell)
        df_feasibility_balancing_DS[dataset_ID].loc[idx, 'std_feasiblity_cell'] = np.std(feas_cell)
        
        unfeas1_cell = [cases_x_stab.loc[(cell,[-1]),'Stability'].sum()/tot_cases.loc[cell,'Stability'] if -1 in cases_x_stab.loc[cell].index else 0 for cell in tot_cases.index]
        df_feasibility_balancing_DS[dataset_ID].loc[idx, 'mean_unfeas_1'] = np.mean(unfeas1_cell)
        df_feasibility_balancing_DS[dataset_ID].loc[idx, 'std_unfeas_1'] = np.std(unfeas1_cell)
   
        unfeas2_cell = [cases_x_stab.loc[(cell,[-2]),'Stability'].sum()/tot_cases.loc[cell,'Stability'] if -2 in cases_x_stab.loc[cell].index else 0 for cell in tot_cases.index]
        df_feasibility_balancing_DS[dataset_ID].loc[idx, 'mean_unfeas_2'] = np.mean(unfeas2_cell)
        df_feasibility_balancing_DS[dataset_ID].loc[idx, 'std_unfeas_2'] = np.std(unfeas2_cell)
   
        balance_cell=[]
        for cell in tot_cases.index:
            if 0 in cases_x_stab.loc[cell].index and 1 in cases_x_stab.loc[cell].index:
                balance_cell.append(cases_x_stab.loc[(cell,[1]),'Stability'].sum()/cases_x_stab.loc[(cell,[0,1]),'Stability'].sum())
            else:
                if 0 not in cases_x_stab.loc[cell].index:
                    balance_cell.append(1)
                else:
                    balance_cell.append(0)
                    
        # balance_cell = [cases_x_stab.loc[(cell,[1]),'Stability'].sum()/cases_x_stab.loc[(cell,[0,1]),'Stability'].sum() 
        #                 if 0 in cases_x_stab.loc[cell].index or 1 in cases_x_stab.loc[cell].index else 0 for cell in tot_cases.index]
        
        
        df_feasibility_balancing_DS[dataset_ID].loc[idx, 'mean_balance_cell'] = np.mean(balance_cell)
        df_feasibility_balancing_DS[dataset_ID].loc[idx, 'std_balance_cell'] = np.std(balance_cell)


        entropy_cell = [-cases_x_stab.loc[(cell,[1]),'Stability'].sum()/cases_x_stab.loc[(cell,[0,1]),'Stability'].sum()*np.log(cases_x_stab.loc[(cell,[1]),'Stability'].sum()/cases_x_stab.loc[(cell,[0,1]),'Stability'].sum())\
                        -cases_x_stab.loc[(cell,[0]),'Stability'].sum()/cases_x_stab.loc[(cell,[0,1]),'Stability'].sum()*np.log(cases_x_stab.loc[(cell,[0]),'Stability'].sum()/cases_x_stab.loc[(cell,[0,1]),'Stability'].sum())
                        if 0 in cases_x_stab.loc[cell].index and 1 in cases_x_stab.loc[cell].index else 0
                        for cell in tot_cases.index]
        df_feasibility_balancing_DS[dataset_ID].loc[idx, 'mean_entropy_cell'] = np.mean([x for x in entropy_cell if x != 0])
        df_feasibility_balancing_DS[dataset_ID].loc[idx, 'std_entropy_cell'] = np.std([x for x in entropy_cell if x != 0])

        if len(feas_case_id_depth) !=0:
            df_feasibility_balancing_DS[dataset_ID].loc[idx, 'balance']= feas_stab_depth/len(feas_case_id_depth) 
            
        else:
             df_feasibility_balancing_DS[dataset_ID].loc[idx, 'balance'] = 0
        
        if len(cum_feas_case_id_depth) !=0:
            df_feasibility_balancing_DS[dataset_ID].loc[idx, 'cumulative_balancing']= cum_feas_stab_case_id_depth/len(cum_feas_case_id_depth) 
            cases_x_stab_cum = df_depth_DS[dataset_ID].query('Depth <= @depth').groupby('Stability')[['case_id']].count()
            df_feasibility_balancing_DS[dataset_ID].loc[idx, 'cumulative_entropy']= calculate_entropy([cases_x_stab_cum.loc[0,'case_id']/cases_x_stab_cum.loc[[0,1],'case_id'].sum(axis=0),cases_x_stab_cum.loc[1,'case_id']/cases_x_stab_cum.loc[[0,1],'case_id'].sum(axis=0)])

        
        else:
            df_feasibility_balancing_DS[dataset_ID].loc[idx, 'cumulative_balancing']= 0
            df_feasibility_balancing_DS[dataset_ID].loc[idx, 'cumulative_entropy'] = 0
    df_feasibility_balancing_DS[dataset_ID]=df_feasibility_balancing_DS[dataset_ID].fillna(0)


# In[24]:


df_feasibility_balancing_DS[dataset_ID_list[0]]


#%%

import matplotlib.pyplot as plt

cls = ['gray', 'black', 'blue']

# Create 1 row, 2 columns of subplots, sharing the Y axis
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(4*num_datasets, 5))
dataset_ID_title_list = ['TEST \#1','TEST \#2']

for idx, dataset_ID in enumerate(dataset_ID_list):
    ax = axes[idx]  # Select subplot

    ax.errorbar(
        df_feasibility_balancing_DS[dataset_ID]['depth'],
        df_feasibility_balancing_DS[dataset_ID]['mean_unfeas_1']*100,
        df_feasibility_balancing_DS[dataset_ID]['std_unfeas_1']*100,
        fmt='o-',
        ecolor=cls[0],
        elinewidth=1.5,
        capsize=5,
        capthick=1.5,
        markersize=8,
        color=cls[0],
        label='Infeasible'
    )

    ax.errorbar(
        df_feasibility_balancing_DS[dataset_ID]['depth'],
        df_feasibility_balancing_DS[dataset_ID]['mean_feasiblity_cell']*100,
        df_feasibility_balancing_DS[dataset_ID]['std_feasiblity_cell']*100,
        fmt='o-',
        ecolor=cls[2],
        elinewidth=1.5,
        capsize=5,
        capthick=1.5,
        markersize=8,
        color=cls[2],
        label='Feasible'
    )
    
    ax.errorbar(
        df_feasibility_balancing_DS[dataset_ID]['depth'],
        df_feasibility_balancing_DS[dataset_ID]['mean_unfeas_2']*100,
        df_feasibility_balancing_DS[dataset_ID]['std_unfeas_2']*100,
        fmt='o-',
        ecolor=cls[1],
        elinewidth=1.5,
        capsize=5,
        capthick=1.5,
        markersize=8,
        color=cls[1],
        label='Feasible discarded'
    )
    
    ax.plot(np.arange(-1,12), np.ones(len(np.arange(-1,12)))*5, color='red')#, linewidth=0.5)
    ax.set_xlim(-0.5, df_feasibility_balancing_DS[dataset_ID]['depth'].max()+0.05)
    ax.set_xlabel('Depth')
    ax.set_title(dataset_ID_title_list[idx])#['Sensitivity' if dataset_ID == 'ivity' else dataset_ID][0]
    ax.grid(True)

# Shared Y label
axes[0].set_ylabel('Rate [\%]')

# Common legend at the bottom (outside)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='lower center',
    ncol=3,
    frameon=True,
    bbox_to_anchor=(0.5, 0)
)

fig.tight_layout(rect=[0, 0.1, 1, 1])  # Leave space for legend

plt.show()
plt.savefig('figures_paper/feasiblility_rate'+dataset_ID_list[0]+dataset_ID_list[1]+'.pdf', format='pdf')
plt.savefig('figures_paper/feasiblility_rate'+dataset_ID_list[0]+dataset_ID_list[1]+'.png',dpi=320)#, format='pdf')

#%%
cls = ['green']
import matplotlib.colors as mcolors

# Create 1 row, 2 columns of subplots, sharing the Y axis
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(4*num_datasets, 5))

for idx, dataset_ID in enumerate(dataset_ID_list):
    ax = axes[idx]  # Select subplot

    ax.errorbar(
        df_feasibility_balancing_DS[dataset_ID]['depth'],
        df_feasibility_balancing_DS[dataset_ID]['mean_entropy_cell'],
        df_feasibility_balancing_DS[dataset_ID]['std_entropy_cell'],
        fmt='o-',
        ecolor=cls[0],
        elinewidth=1.5,
        capsize=5,
        capthick=1.5,
        markersize=8,
        color=cls[0],
        label='Subregions mean $\pm$ std'
    )
    
    ax.plot(df_feasibility_balancing_DS[dataset_ID]['depth'],df_feasibility_balancing_DS[dataset_ID]['cumulative_entropy'],
            color = 'lightgreen', label='Cumulative'
)
    
    ax.set_xlabel('Depth')
    ax.set_title(dataset_ID_title_list[idx])#['Sensitivity' if dataset_ID == 'ivity' else dataset_ID][0])
    ax.grid(True)

# Shared Y label
axes[0].set_ylabel('Entropy')

# Common legend at the bottom (outside)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='lower center',
    ncol=2,
    frameon=True,
    bbox_to_anchor=(0.5, 0)
)

fig.tight_layout(rect=[0, 0.1, 1, 1])

plt.savefig('figures_paper/entropy'+dataset_ID_list[0]+dataset_ID_list[1]+'_with_cum_entropy.pdf', format='pdf')
plt.savefig('figures_paper/entropy'+dataset_ID_list[0]+dataset_ID_list[1]+'_with_cum_entropy.png',dpi=320)#, format='pdf')

#%%

scores_df=dict()
for dataset_ID, dir_name in zip(dataset_ID_list,dir_names):
    scores_df[dataset_ID]=pd.read_excel('D:/'+dir_name+'/scores_depth_PFI_xgb.xlsx')

#%%

cls=[ "#EBAA55", "#B400C8"]
mrk = ['o-','s-']
fig, ax = plt.subplots()

for idx, dataset_ID in enumerate(dataset_ID_list):   
    ax.errorbar(
        scores_df[dataset_ID]['Depth'],
        scores_df[dataset_ID]['score_mean'],
        scores_df[dataset_ID]['score_std'],
        fmt=mrk[idx],                    # 'o' for circular markers, '-' for connecting line
        ecolor=cls[idx],              # color of error bars
        elinewidth=1.5,              # thickness of error bar lines
        capsize=5,                   # length of error bar caps
        capthick=1.5,                # thickness of the cap lines
        markersize=8,                # size of markers
        color=cls[idx],                # color of line and markers
        label='TEST \#'+str(idx+1)
    )    
ax.set_xlabel('Depth')
ax.set_ylabel('Accuracy')
ax.grid(True)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()

plt.savefig('figures_paper/accuracy'+dataset_ID_list[0]+dataset_ID_list[1]+'.pdf', format='pdf')
plt.savefig('figures_paper/accuracy'+dataset_ID_list[0]+dataset_ID_list[1]+'.png',dpi=320)#, format='pdf')

#%%
cls=['b','r']
mrk = ['o-','s-']
fig, ax = plt.subplots()

for idx, dataset_ID in enumerate(dataset_ID_list):
    ax.errorbar(
        df_feasibility_balancing_DS[dataset_ID]['depth'],
        df_feasibility_balancing_DS[dataset_ID]['mean_feasiblity_cell'],
        df_feasibility_balancing_DS[dataset_ID]['std_feasiblity_cell'],
        fmt=mrk[idx],                    # 'o' for circular markers, '-' for connecting line
        ecolor='black',              # color of error bars
        elinewidth=1.5,              # thickness of error bar lines
        capsize=5,                   # length of error bar caps
        capthick=1.5,                # thickness of the cap lines
        markersize=8,                # size of markers
        color='blue',                # color of line and markers
        label='Data with error bars'
    )    
    ax.plot(df_feasibility_balancing_DS[dataset_ID]['depth'], df_feasibility_balancing_DS[dataset_ID]['cumulative_feasibility'], marker='s',color=cls[idx], label='Cumulative Feasibility'+dataset_ID)
ax.set_xlabel('Depth')
ax.set_ylabel('Feasibility')
ax.set_title('Feasibility vs Depth')
ax.grid(True)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()


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

#%%
cls=['b','r']
mrk = ['o-','s-']

fig, ax = plt.subplots()

for idx, dataset_ID in enumerate(dataset_ID_list):
    ax.errorbar(
        df_feasibility_balancing_DS[dataset_ID]['depth'],
        df_feasibility_balancing_DS[dataset_ID]['mean_entropy_cell'],
        df_feasibility_balancing_DS[dataset_ID]['std_entropy_cell'],
        fmt=mrk[idx],                    # 'o' for circular markers, '-' for connecting line
        ecolor=cls[idx],              # color of error bars
        elinewidth=1.5,              # thickness of error bar lines
        capsize=5,                   # length of error bar caps
        capthick=1.5,                # thickness of the cap lines
        markersize=8,                # size of markers
        color=cls[idx],                # color of line and markers
        label= dataset_ID[1:]
    )        
    #ax.plot(df_feasibility_balancing_DS[dataset_ID]['depth'], df_feasibility_balancing_DS[dataset_ID]['cumulative_balancing'], marker='s',color=cls[idx], label='Cumulative Balance'+dataset_ID)
ax.set_xlabel('Depth')
ax.set_ylabel('Entropy')
#ax.set_title('Balance vs Depth')
ax.grid(True)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()


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
        #ax.set_xlim([100,620])        
        #ax.set_ylim([0,300])
        
#%%
df_entropy_cell_DS = dict()
for dataset_ID in dataset_ID_list:
    
    df_entropy_cell_DS[dataset_ID]= pd.read_excel('df_entropy_cell'+dataset_ID+'.xlsx')
    #df_entropy_cell_DS[dataset_ID]['CellName'] = [str(c) for c in df_entropy_cell_DS[dataset_ID]['CellName']]
#%%
df_entropy_cell_comparison=df_entropy_cell_DS[dataset_ID_list[0]].set_index('CellName')

df_entropy_cell_comparison = df_entropy_cell_comparison.merge(
    df_entropy_cell_DS[dataset_ID_list[1]].set_index('CellName'),
    right_index=True,
    left_on='CellName'
)

#%%
stability_cell_DS=dict()
for dataset_ID in dataset_ID_list:
    stability_cell_DS[dataset_ID]=results_dataframes_datasets[dataset_ID]['cases_df'].query('cell_name == "0.1.1.3.1"')['Stability']

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