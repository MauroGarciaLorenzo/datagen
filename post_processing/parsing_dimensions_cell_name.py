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
path = '../results/'
#path = 'D:/'
dir_name=[dir_name for dir_name in os.listdir(path) if '_2732' in dir_name and 'zip' not in dir_name][0]# if dir_name.startswith('datagen') and 'zip' not in dir_name]#
print(dir_name)
# dir_names = [
#     #'datagen_ACOPF_slurm23172357_cu10_nodes32_LF09_seed3_nc3_ns500_d7_20250627_214226_7664']
#     'datagen_ACOPF_slurm25105245_cu8_nodes32_LF09_seed3_nc3_ns500_d7_20250731_132256_7665']

#%%
path_results = os.path.join(path, dir_name)
df_op='df_op'#'case_df_op'
results_dataframes, csv_files = open_csv(
    path_results, ['cases_df.csv', df_op+'.csv', 'dims_df.csv', 'cell_info.csv'])

# results_dataframes, csv_files = open_csv(
#     path_results, ['dims_df.csv'], results_dataframes)

perc_stability(results_dataframes[df_op], dir_name)

dataset_ID = dir_name[-5:]

# %% ---- FILL NAN VALUES WITH NULL ---

results_dataframes[df_op] = results_dataframes[df_op].fillna(0)

# %% ---- SELECT ONLY FEASIBLE CASES ----

results_dataframes['case_df_op_feasible'] = results_dataframes[df_op].query('Stability >= 0')

case_id_feasible = list(results_dataframes['case_df_op_feasible']['case_id'])

# case_id=case_id_feasible[0]
# results_dataframes['case_df_op_feasible'].query('case_id == @case_id')['P_SG12'] <--- quantities calculated by power flow
# results_dataframes['cases_df'].query('case_id == @case_id')['p_sg_Var10'] <-- quantities sampled

results_dataframes['cases_df_feasible'] = results_dataframes['cases_df'].query('Stability >= 0')
case_id_feasible = list(results_dataframes['cases_df_feasible']['case_id'])
n_feas_cases = len(case_id_feasible)

results_dataframes['case_df_op_feasible_X'] = results_dataframes['case_df_op_feasible'].drop(['case_id', 'Stability'], axis=1)                        

       
# %% ---- SELECT ONLY UNFEASIBLE CASES ----

results_dataframes['case_df_op_unfeasible'] = results_dataframes[df_op].query('Stability < 0')
results_dataframes['case_df_op_unfeasible_1'] = results_dataframes[df_op].query('Stability == -1')
results_dataframes['case_df_op_unfeasible_2'] = results_dataframes[df_op].query('Stability == -2')

case_id_Unfeasible = list(results_dataframes['case_df_op_unfeasible']['case_id'])
case_id_Unfeasible1 = list(results_dataframes['case_df_op_unfeasible_1']['case_id'])
case_id_Unfeasible2 = list(results_dataframes['case_df_op_unfeasible_2']['case_id'])

results_dataframes['cases_df_unfeasible'] = results_dataframes['cases_df'].query('Stability < 0')
results_dataframes['cases_df_unfeasible_1'] = results_dataframes['cases_df'].query('Stability == -1')
results_dataframes['cases_df_unfeasible_2'] = results_dataframes['cases_df'].query('Stability == -2')

case_id_Unfeasible = list(results_dataframes['cases_df']['case_id'])
case_id_Unfeasible1 = list(results_dataframes['cases_df_unfeasible_1']['case_id'])
case_id_Unfeasible2 = list(results_dataframes['cases_df_unfeasible_1']['case_id'])

# results_dataframes['cases_df_unfeasible'] = results_dataframes['cases_df'].query(
#     'case_id == @case_id_Unfeasible')  # <-- quantities sampled
# results_dataframes['cases_df_unfeasible_1'] = results_dataframes['cases_df'].query(
#     'case_id == @case_id_Unfeasible1')  # <-- quantities sampled
# results_dataframes['cases_df_unfeasible_2'] = results_dataframes['cases_df'].query(
#     'case_id == @case_id_Unfeasible2')  # <-- quantities sampled


#%%
def create_dimensions_caseid_df(df_dict, df_name, vars_dim1, vars_dim2, name_dim1, name_dim2):
    dimensions_caseid = pd.DataFrame(columns = [name_dim1,name_dim2,'case_id','Stability'])
    dimensions_caseid[name_dim1] =  df_dict[df_name][vars_dim1].sum(axis=1)
    dimensions_caseid[name_dim2] =  df_dict[df_name][vars_dim2].sum(axis=1)
    dimensions_caseid['case_id'] =  df_dict[df_name]['case_id']
    dimensions_caseid['Stability'] = list(df_dict[df_name]['Stability'])

    return dimensions_caseid
#%%

p_sg_var=[var for var in results_dataframes['case_df_op_feasible'].columns if var.startswith('P_SG')]
p_cig_var=[var for var in results_dataframes['case_df_op_feasible'].columns if var.startswith('P_GFOR') or var.startswith('P_GFOL')]
p_gfor_var=[var for var in results_dataframes['case_df_op_feasible'].columns if var.startswith('P_GFOR')]
taus_var = [var for var in results_dataframes['dims_df'].columns if var.startswith('tau')]

dimensions_caseid_feasible = create_dimensions_caseid_df(results_dataframes, 'case_df_op_feasible', p_sg_var, p_cig_var, 'p_sg', 'p_cig')
dimensions_caseid_feasible['perc_g_for'] = results_dataframes['case_df_op_feasible'][p_gfor_var].sum(axis=1)/dimensions_caseid_feasible['p_cig']
dimensions_caseid_feasible['p_sg'] = dimensions_caseid_feasible['p_sg']*100
dimensions_caseid_feasible['p_cig'] = dimensions_caseid_feasible['p_cig']*100
dimensions_caseid_feasible[taus_var] = results_dataframes['dims_df'].query('case_id == @case_id_feasible')[taus_var]


#%%

p_sg_var=[var for var in results_dataframes['cases_df_unfeasible'].columns if var.startswith('p_sg')]
p_cig_var=[var for var in results_dataframes['cases_df_unfeasible'].columns if var.startswith('p_cig')]
taus_var = [var for var in results_dataframes['dims_df'].columns if var.startswith('tau')]

dimensions_caseid_feasible_sampled = create_dimensions_caseid_df(results_dataframes, 'cases_df_feasible', p_sg_var, p_cig_var, 'p_sg', 'p_cig')
dimensions_caseid_feasible_sampled['perc_g_for']=results_dataframes['dims_df'].loc[dimensions_caseid_feasible_sampled.index,'perc_g_for']
dimensions_caseid_unfeasible = create_dimensions_caseid_df(results_dataframes, 'cases_df_unfeasible', p_sg_var, p_cig_var, 'p_sg', 'p_cig')
dimensions_caseid_unfeasible['perc_g_for']=results_dataframes['dims_df'].loc[dimensions_caseid_unfeasible.index,'perc_g_for']
dimensions_caseid_unfeasible[taus_var] = results_dataframes['dims_df'].query('case_id == @case_id_Unfeasible')[taus_var]

dimensions_caseid_unfeasible1 = create_dimensions_caseid_df(results_dataframes, 'cases_df_unfeasible_1', p_sg_var, p_cig_var, 'p_sg', 'p_cig')
dimensions_caseid_unfeasible2 = create_dimensions_caseid_df(results_dataframes, 'cases_df_unfeasible_2', p_sg_var, p_cig_var, 'p_sg', 'p_cig')


#%%
fig, ax = plt.subplots()
ax.scatter(dimensions_caseid_unfeasible1['p_cig'], dimensions_caseid_unfeasible1['p_sg'],color='silver', label='Unfeasible OP (-1)')
ax.scatter(dimensions_caseid_unfeasible2['p_cig'], dimensions_caseid_unfeasible2['p_sg'],color='k', label='Unfeasible OP (-2)')
ax.scatter(dimensions_caseid_feasible_sampled['p_cig'], dimensions_caseid_feasible_sampled['p_sg'], label='Feasible OP')
ax.scatter(dimensions_caseid_feasible['p_cig'], dimensions_caseid_feasible['p_sg'], label='Feasible PF')
ax.set_xlabel('$P_{CIG}$ [MW]')
ax.set_ylabel('$P_{SG}$ [MW]')
fig.tight_layout()
plt.legend()

#%%
fig, ax = plt.subplots()
ax.scatter(dimensions_caseid_unfeasible['p_cig'], dimensions_caseid_unfeasible['p_sg'],color='silver', label='Unfeasable OP')
ax.scatter(dimensions_caseid_feasible_sampled.query('Stability ==0')['p_cig'], dimensions_caseid_feasible_sampled.query('Stability ==0')['p_sg'], color='r',label='Unstable PF')
ax.scatter(dimensions_caseid_feasible_sampled.query('Stability ==1')['p_cig'], dimensions_caseid_feasible_sampled.query('Stability ==1')['p_sg'], color='g', label='Stable PF')
ax.set_xlabel('$P_{CIG}$ [MW]')
ax.set_ylabel('$P_{SG}$ [MW]')
fig.tight_layout()
plt.legend()
#%%

fig, ax = plt.subplots()
ax.scatter(dimensions_caseid_unfeasible['perc_g_for'], dimensions_caseid_unfeasible['p_sg'],color='silver', label='Unfeasable OP')
ax.scatter(dimensions_caseid_feasible_sampled.query('Stability ==0')['perc_g_for'], dimensions_caseid_feasible_sampled.query('Stability ==0')['p_sg'], color='r',label='Unstable PF')
ax.scatter(dimensions_caseid_feasible_sampled.query('Stability ==1')['perc_g_for'], dimensions_caseid_feasible_sampled.query('Stability ==1')['p_sg'], color='g', label='Stable PF')
ax.set_xlabel('$\%_{GFOR}$')
ax.set_ylabel('$P_{SG}$ [MW]')
fig.tight_layout()
plt.legend()

# In[11]:


fig, ax = plt.subplots()
ax.scatter(dimensions_caseid_unfeasible['p_cig'], dimensions_caseid_unfeasible['p_sg'],color='silver', label='Unfeasable OP')
ax.scatter(dimensions_caseid_feasible.query('Stability ==0')['p_cig'], dimensions_caseid_feasible.query('Stability ==0')['p_sg'], color='r',label='Unstable PF')
ax.scatter(dimensions_caseid_feasible.query('Stability ==1')['p_cig'], dimensions_caseid_feasible.query('Stability ==1')['p_sg'], color='g', label='Stable PF')
ax.set_xlabel('$P_{CIG}$ [MW]')
ax.set_ylabel('$P_{SG}$ [MW]')
fig.tight_layout()
plt.legend()

#%%
fig, ax = plt.subplots()
ax.scatter(dimensions_caseid_unfeasible['perc_g_for'], dimensions_caseid_unfeasible['p_sg'],color='silver', label='Unfeasable OP')
ax.scatter(dimensions_caseid_feasible.query('Stability ==0')['perc_g_for'], dimensions_caseid_feasible.query('Stability ==0')['p_sg'], color='r',label='Unstable PF')
ax.scatter(dimensions_caseid_feasible.query('Stability ==1')['perc_g_for'], dimensions_caseid_feasible.query('Stability ==1')['p_sg'], color='g', label='Stable PF')
ax.set_xlabel('$\%_{GFOR}$')
ax.set_ylabel('$P_{SG}$ [MW]')
fig.tight_layout()
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
        # if name.startswith("tau_"):
        #     continue  # skip taus

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

#%% -- SI NO HAY EXECUTION LOG
results_dataframes['cell_info'].rename(columns={'Cell Name': 'CellName'}, inplace=True)

#%%
all_data = []

for cell in results_dataframes['cell_info']['CellName'].unique():
    case_id_cell = list(results_dataframes['cases_df'].query('cell_name == @cell')['case_id'])
    cell_describe = results_dataframes['dims_df'].query('case_id == @case_id_cell').describe()
    dimensions = list (set(results_dataframes['dims_df'].columns)-set(['case_id','p_g_fol','q_sg','q_cig','q_g_fol','p_g_for','q_g_for','q_load', 'p_load']))
    for dimension in dimensions:
        all_data.append({
                "block_id": cell,
                "dimension": dimension,
                "lower": cell_describe.loc['min',dimension],
                "upper": cell_describe.loc['max',dimension],
                "entropy": results_dataframes['cell_info'].loc[results_dataframes['cell_info'].query('CellName==@cell').index[0],'Entropy'],
                #"delta_entropy": delta_entropy,
                "depth": results_dataframes['cell_info'].loc[results_dataframes['cell_info'].query('CellName==@cell').index[0],'Depth']
            })    
    
df = pd.DataFrame(all_data)
df.reset_index(drop=True, inplace=True)
print(df.head())

df['lower'] = np.round(df['lower'],2)
df['upper'] = np.round(df['upper'],2)

#%%
# result container
df_sensitivity = pd.DataFrame(columns=['cell', 'dim1', 'dim2'])
i = 0

# start from root parent
parents = ['0']       # queue of parents to process
max_depth = 6        # depth in terms of digits in block_id (after removing '.')

while parents:
    parent = parents.pop(0)

    # depth of current parent (using your convention)
    parent_depth = len(parent.replace('.', ''))

    # if this parent is already at max depth, don't split it further
    if parent_depth >= max_depth:
        continue

    # build its 4 children IDs (0.1 -> 0.1.1..0.1.4, etc.)
    childs = [f'{parent}.{k}' for k in range(1, 5)]

    # keep only children that actually exist in df
    existing_childs = [c for c in childs if c in df['block_id'].values]
    if not existing_childs:
        continue

    # ---- your original sensitivity logic ----
    df_parent = (
        df.query('block_id == @parent')[['dimension', 'lower', 'upper']]
          .sort_values(by='dimension')
          .reset_index(drop=True)
    )

    # use the first child – all four differ only in the two split dims
    child = existing_childs[0]
    df_child = (
        df.query('block_id == @child')[['dimension', 'lower', 'upper']]
          .sort_values(by='dimension')
          .reset_index(drop=True)
    )

    # relative difference
    df_diff = (df_parent[['lower', 'upper']] - df_child[['lower', 'upper']]) / df_parent[['lower', 'upper']]

    # two dimensions with largest change (using 'upper' like you did)
    dim_idx = df_diff.sort_values(by='upper', ascending=False).index[0:2]

    df_sensitivity.loc[i, 'cell'] = parent
    df_sensitivity.loc[i, 'dim1'] = df_parent.loc[dim_idx[0], 'dimension']
    df_sensitivity.loc[i, 'dim2'] = df_parent.loc[dim_idx[1], 'dimension']
    i += 1
    # -----------------------------------------

    # enqueue children as future parents (to go deeper)
    parents.extend(existing_childs)

# df_sensitivity now has, for each parent cell, the two split dimensions
pd.DataFrame.to_excel(df_sensitivity,path+dir_name+'/sensitivity_log'+dataset_ID+'.xlsx', index=False)

#%%
# Crear una columna con el par ordenado alfabéticamente
df_sensitivity['pair'] = df_sensitivity.apply(
    lambda row: tuple(sorted([row['dim1'], row['dim2']])),
    axis=1
)

# Quitar duplicados
unique_pairs = df_sensitivity['pair'].drop_duplicates()

# Convertir a dataframe si lo quieres en tabla
df_unique_pairs = unique_pairs.apply(pd.Series)
df_unique_pairs.columns = ['dim1', 'dim2']

df_unique_pairs
#%%

# dim_divided=[]
# for dim in df['dimension'].unique():
#     if len(df.query('dimension == @dim')['lower'].unique())==1:
#         continue
#     else:
#         print(dim)
#         dim_divided.append(dim)

# #%%

# dim_combs=[(dim,'p_sg') for dim in dim_divided if dim!='p_sg']

# for dims in dim_combs:

#%%
for i in df_unique_pairs.index:
    # Filter for only p_cig and p_sg
    #mesh_df = df[df["dimension"].isin([dims[0], "p_sg"])]
    dim1=df_unique_pairs.loc[i,'dim1']
    dim2= df_unique_pairs.loc[i,'dim2']
    mesh_df = df[df["dimension"].isin([dim1,dim2])]
    
    fig, ax = plt.subplots()
    ax.scatter(dimensions_caseid_unfeasible[dim1], dimensions_caseid_unfeasible[dim2],color='silver', label='Unfeasable OP')
    ax.scatter(dimensions_caseid_feasible.query('Stability ==0')[dim1], dimensions_caseid_feasible.query('Stability ==0')[dim2], color='r',label='Unstable PF')
    ax.scatter(dimensions_caseid_feasible.query('Stability ==1')[dim1], dimensions_caseid_feasible.query('Stability ==1')[dim2], color='g', label='Stable PF')
    try:
        ax.set_xlabel('$'+dim1.replace('perc','\%').replace('p','P').replace('g_for','{GFOR}').replace('sg','{SG} [MW]').replace('cig','{IBR} [MW]')+'$')
    except:
        ax.set_xlabel(dim1)
    try:
        ax.set_ylabel('$'+dim2.replace('perc','\%').replace('p','P').replace('g_for','{GFOR}').replace('sg','{SG} [MW]').replace('cig','{IBR} [MW]')+'$')#'$P_{SG}$ [MW]')
    except:
        ax.set_ylabel(dim2)
        
    fig.tight_layout()
    plt.legend()
    
    #plot_mesh(mesh_df,dims[0], 'p_sg', dims[0], '$P_{SG}$ [MW]',ax)
    #plot_mesh(mesh_df,dim1, dim2, dim1, dim2,ax)
    #pd.DataFrame.to_excel(mesh_df,path+dir_name+'/mesh'+dataset_ID.replace('ivity','Sensitivity')+'_'+dims[0]+'_p_sg.xlsx', index=False)



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
df_depth['Depth'] = [len(str(x).split('.'))-1 if '.' in str(x) else 0 for x in results_dataframes['cases_df']['cell_name']]

pd.DataFrame.to_excel(df_depth, path+dir_name+'/cases_id_depth'+dataset_ID.replace('ivity','Sensitivity')+'.xlsx')
#%%

results_dataframes['cases_df']['p_sg'] =  results_dataframes['cases_df'][p_sg_var].sum(axis=1)
results_dataframes['cases_df']['p_cig'] =  results_dataframes['cases_df'][p_cig_var].sum(axis=1)

#%%

ax = plot_mesh(mesh_df, 'p_cig','p_sg','p_cig','p_sg')
for depth in np.sort(df_depth['Depth'].unique()):
    #print(key)
#    case_id_list= list(set(df_depth.query('Depth == @depth')['case_id']) & set(case_id_feasible))
    case_id_list= list(df_depth.query('Depth == @depth')['case_id'])
    ax.scatter(results_dataframes['cases_df'].query('case_id == @case_id_list')['p_cig'],results_dataframes['cases_df'].query('case_id == @case_id_list')['p_sg'], label='Depth '+str(depth))

    plt.pause(1)             

ax.legend(loc='center left')#, bbox_to_anchor=(1, 0.5))

#%%
mesh_df = df[df["dimension"].isin(['p_cig', "p_sg"])]

ax = plot_mesh(mesh_df, 'p_cig','p_sg','p_cig','p_sg')
for depth in np.sort(df_depth['Depth'].unique()):
    #print(key)
#    case_id_list= list(set(df_depth.query('Depth == @depth')['case_id']) & set(case_id_feasible))
    case_id_list= list(df_depth.query('Depth == @depth')['case_id'])
    ax.scatter(results_dataframes['cases_df'].query('case_id == @case_id_list and Stability <0')['p_cig'],results_dataframes['cases_df'].query('case_id == @case_id_list and Stability <0')['p_sg'], color='silver')
    ax.scatter(results_dataframes['cases_df'].query('case_id == @case_id_list and Stability ==0')['p_cig'],results_dataframes['cases_df'].query('case_id == @case_id_list and Stability ==0')['p_sg'], color = 'r')
    ax.scatter(results_dataframes['cases_df'].query('case_id == @case_id_list and Stability ==1')['p_cig'],results_dataframes['cases_df'].query('case_id == @case_id_list and Stability ==1')['p_sg'], color = 'g')

    plt.pause(5)             

#ax.legend(loc='center left')#, bbox_to_anchor=(1, 0.5))

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

def eval_entropy(stabilities, entropy_parent):
    """Calculate entropy of the cell using its list of stabilities.

    :param stabilities: List of stabilities (result of the evaluation of every
    case)
    :param entropy_parent: Parent entropy based on concrete cases (those which
    correspond to the cell)
    :return: Entropy and delta entropy
    """

    stabilities = [x for x in stabilities if x >= 0]
    
    if len(stabilities)==0:
            entropy = 0
    else:
        freqs = []
        counter = 0
        for stability in stabilities:
            if stability == 1:
                counter += 1
        freqs.append(counter / len(stabilities))
        freqs.append((len(stabilities) - counter) / len(stabilities))
        entropy = calculate_entropy(freqs)
    if entropy_parent is None:
        delta_entropy = 1
    else:
        delta_entropy = entropy - entropy_parent
    return entropy, delta_entropy

#%%
df_entropy_cell = pd.DataFrame(columns = ['CellName','Entropy'],index=np.arange(0,len(results_dataframes['cases_df']['cell_name'].unique())))

for idx, cellname in enumerate(results_dataframes['cases_df']['cell_name'].unique()):
    
    stabilities = results_dataframes['cases_df_feasible'].query('cell_name == @cellname')['Stability']
    
    entropy, delta_entropy = eval_entropy(stabilities,None)
    
    df_entropy_cell.loc[idx,'CellName']=cellname
    df_entropy_cell.loc[idx,'Entropy']=entropy

pd.DataFrame.to_excel(df_entropy_cell, path+dir_name+'/df_entropy_cell'+dataset_ID.replace('ivity','Sensitivity')+'.xlsx')
