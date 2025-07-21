import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils_pp_standalone import *

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

fig, ax = plt.subplots()
ax.scatter(dimensions_caseid_stability['p_cig'], dimensions_caseid_stability['p_sg'])

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
fig, ax = plt.subplots()
ax.scatter(dimensions_caseid_unfeasible1['p_cig'], dimensions_caseid_unfeasible1['p_sg'],color='silver', label='Unfeasable OP')
ax.scatter(dimensions_caseid_unfeasible2['p_cig'], dimensions_caseid_unfeasible2['p_sg'],color='k', label='Unfeasable OP')
ax.scatter(dimensions_caseid_stability['p_cig'], dimensions_caseid_stability['p_sg'], label='Feasable OP')
plt.legend()

#%%

fig, ax = plt.subplots()
ax.scatter(dimensions_caseid_unfeasible['p_cig'], dimensions_caseid_unfeasible['p_sg'],color='silver', label='Unfeasable OP')
ax.scatter(dimensions_caseid_stability.query('Stability ==0')['p_cig'], dimensions_caseid_stability.query('Stability ==0')['p_sg'], color='r',label='Unstable OP')
ax.scatter(dimensions_caseid_stability.query('Stability ==1')['p_cig'], dimensions_caseid_stability.query('Stability ==1')['p_sg'], color='g', label='Stable OP')
plt.legend()

#%%

import pandas as pd
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
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Filter for only p_cig and p_sg
mesh_df = df[df["dimension"].isin(["p_cig", "p_sg"])]

# Group by each block based on entropy, delta_entropy, and depth
grouped = mesh_df.groupby('block_id')

# Create the plot
#fig, ax = plt.subplots(figsize=(8, 6))

for i, group in grouped:
    #group = block_id_group[1]
    try:
        p_cig_row = group[group["dimension"] == "p_cig"].iloc[0]
        p_sg_row = group[group["dimension"] == "p_sg"].iloc[0]

        x0, x1 = p_cig_row["lower"], p_cig_row["upper"]
        y0, y1 = p_sg_row["lower"], p_sg_row["upper"]

        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                 linewidth=1, edgecolor='blue', facecolor='lightblue', alpha=0.4)
        ax.add_patch(rect)
    except IndexError:
        # Skip blocks that are missing either p_cig or p_sg
        continue

ax.set_xlabel("Total $P_{IBR}$ [MW]")
ax.set_ylabel("Total $P_{SG}$ [MW]")
#ax.set_title("2D Mesh of p_cig vs p_sg")
plt.grid(True)
plt.tight_layout()
plt.show()
ax.set_xlim(900, 4700)    # Example range for p_cig
ax.set_ylim(4000, 1.1*mesh_df.query('dimension == "p_sg"')['upper'].max())   # Example range for p_sg

#%%
df_cell_info = pd.read_csv(path+'/'+dir_names[0]+'/cell_info.csv')
