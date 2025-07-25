import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils_pp_standalone import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# %%

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

# %%
# ncig_list=['10','11','12']
path = '../results/MareNostrum'

# dir_names=[dir_name for dir_name in os.listdir(path) if dir_name.startswith('datagen') and 'zip' not in dir_name]#

dir_names = [
    'datagen_ACOPF_slurm23172357_cu10_nodes32_LF09_seed3_nc3_ns500_d7_20250627_214226_7664-20250630T085420Z-1-005']

for dir_name in dir_names:
    path_results = os.path.join(path, dir_name)

    results_dataframes, csv_files = open_csv(
        path_results, ['cases_df.csv', 'case_df_op.csv'])

    perc_stability(results_dataframes['case_df_op'], dir_name)

# %% ---- FILL NAN VALUES WITH NULL ---

results_dataframes['case_df_op'] = results_dataframes['case_df_op'].fillna(0)

# %% ---- SELECT ONLY FEASIBLE CASES ----

results_dataframes['case_df_op_feasible'] = results_dataframes['case_df_op'].query(
    'Stability >= 0')

case_id_feasible = list(results_dataframes['case_df_op_feasible']['case_id'])

# case_id=case_id_feasible[0]
# results_dataframes['case_df_op_feasible'].query('case_id == @case_id')['P_SG12'] <--- quantities calculated by power flow
# results_dataframes['cases_df'].query('case_id == @case_id')['p_sg_Var10'] <-- quantities sampled

results_dataframes['cases_df_feasible'] = results_dataframes['cases_df'].query(
    'case_id == @case_id_feasible')  # <-- quantities sampled

n_feas_cases = len(case_id_feasible)

results_dataframes['case_df_op_feasible_X'] = results_dataframes['case_df_op_feasible'].drop(['case_id', 'Stability'], axis=1)                        

       
# %% ---- SELECT ONLY UNFEASIBLE CASES ----

results_dataframes['case_df_op_unfeasible'] = results_dataframes['case_df_op'].query('Stability < 0')
results_dataframes['case_df_op_unfeasible_1'] = results_dataframes['case_df_op'].query('Stability == -1')
results_dataframes['case_df_op_unfeasible_2'] = results_dataframes['case_df_op'].query('Stability == -2')

case_id_Unfeasible = list(results_dataframes['case_df_op_unfeasible']['case_id'])
case_id_Unfeasible1 = list(results_dataframes['case_df_op_unfeasible_1']['case_id'])
case_id_Unfeasible2 = list(results_dataframes['case_df_op_unfeasible_2']['case_id'])

results_dataframes['cases_df_unfeasible'] = results_dataframes['cases_df'].query(
    'case_id == @case_id_Unfeasible')  # <-- quantities sampled
results_dataframes['cases_df_unfeasible_1'] = results_dataframes['cases_df'].query(
    'case_id == @case_id_Unfeasible1')  # <-- quantities sampled
results_dataframes['cases_df_unfeasible_2'] = results_dataframes['cases_df'].query(
    'case_id == @case_id_Unfeasible2')  # <-- quantities sampled


#%%
p_sg_var=[var for var in results_dataframes['cases_df_feasible'].columns if var.startswith('p_sg_')]
p_cig_var=[var for var in results_dataframes['cases_df_feasible'].columns if var.startswith('p_cig_')]

dimensions_caseid_stability = pd.DataFrame(columns = ['p_sg','p_cig','case_id','Stability'])
dimensions_caseid_stability['p_sg'] =  results_dataframes['cases_df_feasible'][p_sg_var].sum(axis=1)
dimensions_caseid_stability['p_cig'] =  results_dataframes['cases_df_feasible'][p_cig_var].sum(axis=1)
dimensions_caseid_stability['case_id'] =  results_dataframes['cases_df_feasible']['case_id']

grup_by_case = results_dataframes['case_df_op_feasible'].groupby('case_id').mean()
dimensions_caseid_stability['Stability'] = list(grup_by_case['Stability'])

# fig, ax = plt.subplots()
# ax.scatter(dimensions_caseid_stability['p_cig'], dimensions_caseid_stability['p_sg'])

#%%
dimensions_caseid_unfeasible = pd.DataFrame(columns = ['p_sg','p_cig','case_id'])
dimensions_caseid_unfeasible['p_sg'] =  results_dataframes['cases_df_unfeasible'][p_sg_var].sum(axis=1)
dimensions_caseid_unfeasible['p_cig'] =  results_dataframes['cases_df_unfeasible'][p_cig_var].sum(axis=1)
dimensions_caseid_unfeasible['case_id'] =  results_dataframes['cases_df_unfeasible']['case_id']

dimensions_caseid_unfeasible1 = pd.DataFrame(columns = ['p_sg','p_cig','case_id'])
dimensions_caseid_unfeasible1['p_sg'] =  results_dataframes['cases_df_unfeasible_1'][p_sg_var].sum(axis=1)
dimensions_caseid_unfeasible1['p_cig'] =  results_dataframes['cases_df_unfeasible_1'][p_cig_var].sum(axis=1)
dimensions_caseid_unfeasible1['case_id'] =  results_dataframes['cases_df_unfeasible_1']['case_id']

dimensions_caseid_unfeasible2 = pd.DataFrame(columns = ['p_sg','p_cig','case_id'])
dimensions_caseid_unfeasible2['p_sg'] =  results_dataframes['cases_df_unfeasible_2'][p_sg_var].sum(axis=1)
dimensions_caseid_unfeasible2['p_cig'] =  results_dataframes['cases_df_unfeasible_2'][p_cig_var].sum(axis=1)
dimensions_caseid_unfeasible2['case_id'] =  results_dataframes['cases_df_unfeasible_2']['case_id']

#%%
# fig, ax = plt.subplots()
# ax.scatter(dimensions_caseid_unfeasible1['p_cig'], dimensions_caseid_unfeasible1['p_sg'],color='silver', label='Unfeasable OP')
# ax.scatter(dimensions_caseid_unfeasible2['p_cig'], dimensions_caseid_unfeasible2['p_sg'],color='k', label='Unfeasable OP')
# ax.scatter(dimensions_caseid_stability['p_cig'], dimensions_caseid_stability['p_sg'], label='Feasable OP')
# plt.legend()

#%%

# fig, ax = plt.subplots()
# ax.scatter(dimensions_caseid_unfeasible['p_cig'], dimensions_caseid_unfeasible['p_sg'],color='silver', label='Unfeasable OP')
# ax.scatter(dimensions_caseid_stability.query('Stability ==0')['p_cig'], dimensions_caseid_stability.query('Stability ==0')['p_sg'], color='r',label='Unstable OP')
# ax.scatter(dimensions_caseid_stability.query('Stability ==1')['p_cig'], dimensions_caseid_stability.query('Stability ==1')['p_sg'], color='g', label='Stable OP')
# plt.legend()

#%%

import re
# Load your full file
with open(path+'/'+dir_names[0]+"/execution_logs.txt") as f:
    text = f.read()

# Split the file into blocks starting with "Dimensions:"
blocks = re.split(r"\bDimensions:\s*", text)
blocks = [b for b in blocks if "Dimension" in b]

all_data = []

for block_id, block in enumerate(blocks):
    # Extract entropy, delta entropy, depth (if available in the block)
    entropy_match = re.search(r"Entropy:\s*([-\d.eE]+)", block)
    delta_entropy_match = re.search(r"Delta Entropy:\s*([-\d.eE]+)", block)
    depth_match = re.search(r"Depth:\s*(\d+)", block)

    entropy = float(entropy_match.group(1)) if entropy_match else None
    delta_entropy = float(delta_entropy_match.group(1)) if delta_entropy_match else None
    depth = int(depth_match.group(1)) if depth_match else None

    # Find Dimension(...) lines and extract relevant data
    for match in re.finditer(r'Dimension\("(?P<name>[^"]+)", borders=(?P<borders>\([^)]+\)|None)\)', block):
        name = match.group("name")
        if name.startswith("tau_"):
            continue  # skip taus

        borders = match.group("borders")
        if borders == "None":
            lower, upper = None, None
        else:
            lower, upper = map(float, borders.strip("()").split(","))

        all_data.append({
            "block_id": block_id,
            "dimension": name,
            "lower": lower,
            "upper": upper,
            "entropy": entropy,
            "delta_entropy": delta_entropy,
            "depth": depth
        })

# Create DataFrame
df = pd.DataFrame(all_data)

# Sort or reset index if needed
df.reset_index(drop=True, inplace=True)

print(df.head())

#%%

# Filter for only p_cig and p_sg
mesh_df = df[df["dimension"].isin(["p_cig", "p_sg"])]

#pd.DataFrame.to_excel(mesh_df,'mesh.xlsx', index=False)

#plot_mesh(mesh_df, ax)

#%%
df_cell_info = pd.read_csv(path+'/'+dir_names[0]+'/cell_info.csv')
df_cell_info.columns = [col.replace(' ','') for col in df_cell_info.columns]
#%%
# Start with your original DataFrame: mesh_df

# Pivot dimension rows into columns
pivot_df = mesh_df.pivot(index="block_id", columns="dimension", values=["lower", "upper"])

# Flatten the MultiIndex columns
pivot_df.columns = [f"{dim}_{bound}" for bound, dim in pivot_df.columns]

# Reset index to get block_id back as a column
pivot_df = pivot_df.reset_index()

# Select representative values for the other metadata (entropy, delta_entropy, depth)
meta_cols = mesh_df.drop(columns=["dimension", "lower", "upper"]).drop_duplicates(subset=["block_id"])

# Merge metadata back in
final_df = pivot_df.merge(meta_cols, on="block_id", how="left")

# Optional: reorder columns
final_df = final_df[[
    "block_id",
    "p_sg_lower", "p_sg_upper",
    "p_cig_lower", "p_cig_upper",
    "entropy", "delta_entropy", "depth"
]]

# Show result
print(final_df)


#%%

results_dataframes['cases_df']['p_sg'] =  results_dataframes['cases_df'][p_sg_var].sum(axis=1)
results_dataframes['cases_df']['p_cig'] =  results_dataframes['cases_df'][p_cig_var].sum(axis=1)
#%%
cell_case_id=dict()
df_cell_info_alive = df_cell_info.query('Status == 1')

for idx, row  in final_df.iterrows():
    p_sg_lower, p_sg_upper, p_cig_lower, p_cig_upper = row['p_sg_lower'], row['p_sg_upper'], row['p_cig_lower'], row['p_cig_upper']

    points = results_dataframes['cases_df'].query('p_sg >= @p_sg_lower and p_sg <= @p_sg_upper and p_cig >= @p_cig_lower and p_cig<= @p_cig_upper')['case_id']
    
    entropy,deltaentropy, depth = row['entropy'], row['delta_entropy'], row['depth']
    
    closest_cell, _ , df_cell_info_alive= find_closest_row(df_cell_info_alive, ['Entropy', 'DeltaEntropy', 'Depth'], [entropy,deltaentropy, depth ] )

    final_df.loc[idx,'CellName'] = closest_cell['CellName']
    try:
        cell_case_id[closest_cell['CellName']].extend(points)
    except:
        cell_case_id[closest_cell['CellName']] =points

#%%
# ax = plot_mesh(mesh_df)
# for key, item in cell_case_id.items():
#     print(key)
#     case_id_list= list(item)
#     ax.scatter(results_dataframes['cases_df'].query('case_id == @case_id_list')['p_cig'],results_dataframes['cases_df'].query('case_id == @case_id_list')['p_sg'])

#     plt.pause(1) 
 
#%% 
internal_leaves=list(df_cell_info_alive['CellName'])
internal_leaves = sorted(internal_leaves, key=len, reverse=True)
final_leaves = list(cell_case_id.keys())

df = df_cell_info.query('Status == 1')
import pandas as pd
from collections import defaultdict
rng = np.random.default_rng(seed=42)  # Create a random generator with fixed seed

# Assuming your data is loaded in a DataFrame called `df`
# with columns: CellName, Depth, Entropy, DeltaEntropy, FeasibleRatio, Status

# Step 1: Sort and clean the data
df['Depth'] = df['CellName'].apply(lambda x: x.count('.') if isinstance(x, str) else 0)
df['Parent'] = df['CellName'].apply(lambda x: '.'.join(x.split('.')[:-1]) if isinstance(x, str) and '.' in x else None)

# Step 2: Create mapping from parent to children
parent_to_children = defaultdict(list)
for idx, row in df.iterrows():
    if row['Parent'] is not None:
        parent_to_children[row['Parent']].append(row['CellName'])

#ax = plot_mesh(mesh_df)

for internal_leaf in internal_leaves:
    childs=parent_to_children[internal_leaf]
    try:
        cell_case_id[internal_leaf]
        print(internal_leaf+' already exists!')
    except:
        cell_case_id[internal_leaf] = []
        for child in childs:
            cell_case_id[internal_leaf].extend(list(cell_case_id[child])[0:len(cell_case_id[child])-1500])
            cell_case_id[child] = list(cell_case_id[child])[-1500:]  
        
        cell_case_id[internal_leaf] = np.random.permutation(cell_case_id[internal_leaf])  
        
        # case_id_list= cell_case_id[internal_leaf]
        # ax.scatter(results_dataframes['cases_df'].query('case_id == @case_id_list')['p_cig'],results_dataframes['cases_df'].query('case_id == @case_id_list')['p_sg'])

        # plt.pause()

        # for child in childs:
        #     case_id_list= cell_case_id[child]
        #     ax.scatter(results_dataframes['cases_df'].query('case_id == @case_id_list')['p_cig'],results_dataframes['cases_df'].query('case_id == @case_id_list')['p_sg'])

        #     plt.pause(3)

        
    
#%%
cell_case_id = dict(sorted(cell_case_id.items(), key=lambda item: len(item[0])))

#%%
# ax = plot_mesh(mesh_df)
# for key, item in cell_case_id.items():
#     #print(key)
#     case_id_list= list(item)
#     ax.scatter(results_dataframes['cases_df'].query('case_id == @case_id_list')['p_cig'],results_dataframes['cases_df'].query('case_id == @case_id_list')['p_sg'])

#     #plt.pause(1)             

#%%

from collections import defaultdict

# Group keys by their length
grouped = defaultdict(list)

for key in cell_case_id:
    grouped[len(key.replace('.',''))-1].append(key)

# Convert to regular dict if needed
grouped_by_length = dict(grouped)

print(grouped_by_length)

df_depth = pd.DataFrame(columns=['Depth','case_id','CellName'])

for depth, items in grouped_by_length.items():
    for item in items:
        df_cell = pd.DataFrame(columns=['Depth','case_id','CellName'])
        df_cell['case_id'] = cell_case_id[item]
        df_cell['Depth'] = depth
        df_cell['CellName'] = item
        
        df_depth = pd.concat([df_depth, df_cell],axis=0)

df_depth = df_depth.reset_index(drop=True)

pd.DataFrame.to_excel(df_depth, 'cases_id_depth.xlsx')

#%%
# ax = plot_mesh(mesh_df)
# for key, item in cell_case_id.items():
#     #print(key)
#     case_id_list= list(set(item)-set(case_id_Unfeasible))
#     ax.scatter(results_dataframes['cases_df'].query('case_id == @case_id_list')['p_cig'],results_dataframes['cases_df'].query('case_id == @case_id_list')['p_sg'])

#     plt.pause(1)    

ax = plot_mesh(mesh_df)
for depth in df_depth['Depth'].unique():
    #print(key)
    case_id_list= list(set(df_depth.query('Depth == @depth')['case_id'])-set(case_id_Unfeasible))
    ax.scatter(results_dataframes['cases_df'].query('case_id == @case_id_list')['p_cig'],results_dataframes['cases_df'].query('case_id == @case_id_list')['p_sg'], alpha =0.1)

    plt.pause(1)             


#%%
# df_cell_info_copy=df_cell_info.query('Status == 1').copy(deep=True)
# final_df_copy=final_df.copy(deep=True)
# cell_case_id=dict()

# for i in range(0,len(results_dataframes['cases_df']),500):
               
#     min_psg, min_pcig = results_dataframes['cases_df'].loc[i:i+499,['p_sg','p_cig']].min()
#     max_psg, max_pcig = results_dataframes['cases_df'].loc[i:i+499,['p_sg','p_cig']].max()
    
#     case_id_list=list(results_dataframes['cases_df'].loc[i:i+499,'case_id'])
    
#     # print(min_psg, min_pcig)
#     # print(max_psg, max_pcig)
    
#     closest_row , idx, final_df_copy = find_closest_row(final_df_copy, ['p_sg_lower', 'p_sg_upper', 'p_cig_lower', 'p_cig_upper'], [min_psg, max_psg, min_pcig, max_pcig])
    
#     closest_entropy, closest_deltaentropy, closest_depth = closest_row['entropy'], closest_row['delta_entropy'], closest_row['depth']
    
#     closest_cell, _ , df_cell_info_copy= find_closest_row(df_cell_info_copy, ['Entropy', 'DeltaEntropy', 'Depth'], [closest_entropy,closest_deltaentropy, closest_depth ] )

#     final_df.loc[idx,'CellName'] = closest_cell['CellName']
#     try:
#         cell_case_id[closest_cell['CellName']].extend(case_id_list)
#     except:
#         cell_case_id[closest_cell['CellName']] =case_id_list

#%%

# import matplotlib.pyplot as plt
# import time

# ax = plot_mesh(mesh_df)
# df_reversed = results_dataframes['cases_df'][::-1].reset_index(drop=True)

# for i in range(0,len(results_dataframes['cases_df']),500):
        
#     #ax.scatter(results_dataframes['cases_df'].loc[i:i+500,'p_cig'], results_dataframes['cases_df'].loc[i:i+500,'p_sg'])
#     ax.scatter(df_reversed.loc[i:i+500,'p_cig'], df_reversed.loc[i:i+500,'p_sg'])

#     # plt.xlabel("p_cig")
#     # plt.ylabel("p_sg")
#     # plt.grid(True)

#     plt.pause(1)  # pause for 0.5 seconds before next rectangle
