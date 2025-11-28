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
path = '../results/'

dir_name=[dir_name for dir_name in os.listdir(path) if '_2732' in dir_name and 'zip' not in dir_name][0]# if dir_name.startswith('datagen') and 'zip' not in dir_name]#
print(dir_name)
# dir_names = [
#     #'datagen_ACOPF_slurm23172357_cu10_nodes32_LF09_seed3_nc3_ns500_d7_20250627_214226_7664']
#     'datagen_ACOPF_slurm25105245_cu8_nodes32_LF09_seed3_nc3_ns500_d7_20250731_132256_7665']

#%%
path_results = os.path.join(path, dir_name)

results_dataframes, csv_files = open_csv(
    path_results, ['cases_df.csv', 'case_df_op.csv'])

perc_stability(results_dataframes['case_df_op'], dir_name)

dataset_ID = dir_name[-5:]


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
def create_dimensions_caseid_df(df_dict, df_name, vars_dim1, vars_dim2, name_dim1, name_dim2):
    dimensions_caseid = pd.DataFrame(columns = [name_dim1,name_dim2,'case_id','Stability'])
    dimensions_caseid[name_dim1] =  df_dict[df_name][vars_dim1].sum(axis=1)
    dimensions_caseid[name_dim2] =  df_dict[df_name][vars_dim2].sum(axis=1)
    dimensions_caseid['case_id'] =  df_dict[df_name]['case_id']
    dimensions_caseid['Stability'] = list(df_dict[df_name]['Stability'])

    return dimensions_caseid

p_sg_var=[var for var in results_dataframes['case_df_op_feasible'].columns if var.startswith('P_SG')]
p_cig_var=[var for var in results_dataframes['case_df_op_feasible'].columns if var.startswith('P_GFOR') or var.startswith('P_GFOL')]

dimensions_caseid_feasible = create_dimensions_caseid_df(results_dataframes, 'case_df_op_feasible', p_sg_var, p_cig_var, 'p_sg', 'p_cig')
dimensions_caseid_feasible['p_sg'] = dimensions_caseid_feasible['p_sg']*100
dimensions_caseid_feasible['p_cig'] = dimensions_caseid_feasible['p_cig']*100

p_sg_var=[var for var in results_dataframes['cases_df_unfeasible'].columns if var.startswith('p_sg')]
p_cig_var=[var for var in results_dataframes['cases_df_unfeasible'].columns if var.startswith('p_cig')]

dimensions_caseid_feasible_sampled = create_dimensions_caseid_df(results_dataframes, 'cases_df_feasible', p_sg_var, p_cig_var, 'p_sg', 'p_cig')
dimensions_caseid_unfeasible = create_dimensions_caseid_df(results_dataframes, 'cases_df_unfeasible', p_sg_var, p_cig_var, 'p_sg', 'p_cig')
dimensions_caseid_unfeasible1 = create_dimensions_caseid_df(results_dataframes, 'cases_df_unfeasible_1', p_sg_var, p_cig_var, 'p_sg', 'p_cig')
dimensions_caseid_unfeasible2 = create_dimensions_caseid_df(results_dataframes, 'cases_df_unfeasible_2', p_sg_var, p_cig_var, 'p_sg', 'p_cig')


#%%
fig, ax = plt.subplots()
ax.scatter(dimensions_caseid_unfeasible1['p_cig'], dimensions_caseid_unfeasible1['p_sg'],color='silver', label='Unfeasable OP (-1)')
ax.scatter(dimensions_caseid_unfeasible2['p_cig'], dimensions_caseid_unfeasible2['p_sg'],color='k', label='Unfeasable OP (-2)')
ax.scatter(dimensions_caseid_feasible_sampled['p_cig'], dimensions_caseid_feasible_sampled['p_sg'], label='Feasable OP')
ax.scatter(dimensions_caseid_feasible['p_cig'], dimensions_caseid_feasible['p_sg'], label='Feasable PF')
ax.set_xlabel('$P_{CIG}$ [MW]')
ax.set_ylabel('$P_{SG}$ [MW]')
plt.legend()


# In[11]:


fig, ax = plt.subplots()
ax.scatter(dimensions_caseid_unfeasible['p_cig'], dimensions_caseid_unfeasible['p_sg'],color='silver', label='Unfeasable OP')
ax.scatter(dimensions_caseid_feasible.query('Stability ==0')['p_cig'], dimensions_caseid_feasible.query('Stability ==0')['p_sg'], color='r',label='Unstable PF')
ax.scatter(dimensions_caseid_feasible.query('Stability ==1')['p_cig'], dimensions_caseid_feasible.query('Stability ==1')['p_sg'], color='g', label='Stable PF')
ax.set_xlabel('$P_{CIG}$ [MW]')
ax.set_ylabel('$P_{SG}$ [MW]')
plt.legend()
#%%

import re
# Load your full file
with open(path+'/'+dir_name+"/execution_logs.txt") as f:
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

pd.DataFrame.to_excel(mesh_df,'mesh'+dataset_ID+'.xlsx', index=False)

plot_mesh(mesh_df, ax)

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

df_depth = pd.DataFrame(columns=['Depth','case_id','CellName'])

df_depth['case_id'] = results_dataframes['cases_df']['case_id']
df_depth['CellName'] = results_dataframes['cases_df']['cell_name']
df_depth['Depth'] = [len(x.split('.'))-1 if '.' in x else 0 for x in results_dataframes['cases_df']['cell_name']]

pd.DataFrame.to_excel(df_depth, 'cases_id_depth'+dataset_ID+'.xlsx')
#%%

results_dataframes['cases_df']['p_sg'] =  results_dataframes['cases_df'][p_sg_var].sum(axis=1)
results_dataframes['cases_df']['p_cig'] =  results_dataframes['cases_df'][p_cig_var].sum(axis=1)

#%%

ax = plot_mesh(mesh_df)
for depth in np.sort(df_depth['Depth'].unique()):
    #print(key)
#    case_id_list= list(set(df_depth.query('Depth == @depth')['case_id']) & set(case_id_feasible))
    case_id_list= list(df_depth.query('Depth == @depth')['case_id'])
    ax.scatter(results_dataframes['cases_df'].query('case_id == @case_id_list')['p_cig'],results_dataframes['cases_df'].query('case_id == @case_id_list')['p_sg'], label='Depth '+str(depth))

    plt.pause(1)             

ax.legend(loc='center left')#, bbox_to_anchor=(1, 0.5))

#%%
ax = plot_mesh(mesh_df)
for depth in np.sort(df_depth['Depth'].unique()):
    #print(key)
#    case_id_list= list(set(df_depth.query('Depth == @depth')['case_id']) & set(case_id_feasible))
    case_id_list= list(df_depth.query('Depth == @depth')['case_id'])
    ax.scatter(results_dataframes['cases_df'].query('case_id == @case_id_list and Stability <0')['p_cig'],results_dataframes['cases_df'].query('case_id == @case_id_list and Stability <0')['p_sg'], color='silver')
    ax.scatter(results_dataframes['cases_df'].query('case_id == @case_id_list and Stability ==0')['p_cig'],results_dataframes['cases_df'].query('case_id == @case_id_list and Stability ==0')['p_sg'], color = 'r')
    ax.scatter(results_dataframes['cases_df'].query('case_id == @case_id_list and Stability ==1')['p_cig'],results_dataframes['cases_df'].query('case_id == @case_id_list and Stability ==1')['p_sg'], color = 'g')

    plt.pause(5)             

#ax.legend(loc='center left')#, bbox_to_anchor=(1, 0.5))


