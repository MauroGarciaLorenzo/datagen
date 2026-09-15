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
#dir_name=[dir_name for dir_name in os.listdir(path) if 'test_sensitivity' in dir_name and 'zip' not in dir_name][0]# if dir_name.startswith('datagen') and 'zip' not in dir_name]#
dir_name = "run_datagen_acopf.py_LF09_seed16_nc3_ns5_d3_20260827_113414_5030"
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
dimensions_caseid_unfeasible['perc_g_for']=results_dataframes['dims_df'].loc[dimensions_caseid_unfeasible.index,'perc_g_for']*100
dimensions_caseid_unfeasible[taus_var] = results_dataframes['dims_df'].query('case_id == @case_id_Unfeasible')[taus_var]

dimensions_caseid_unfeasible1 = create_dimensions_caseid_df(results_dataframes, 'cases_df_unfeasible_1', p_sg_var, p_cig_var, 'p_sg', 'p_cig')
dimensions_caseid_unfeasible2 = create_dimensions_caseid_df(results_dataframes, 'cases_df_unfeasible_2', p_sg_var, p_cig_var, 'p_sg', 'p_cig')

#%%
bf_N_1 = list(results_dataframes['df_op']['bf_N_1'])
bt_N_1 = list(results_dataframes['df_op']['bt_N_1'])

results_dataframes['df_op']['N_1_bf_bt']=[str(int(bf))+'_'+str(int(bt)) for bf,bt in zip(bf_N_1,bt_N_1)]
results_dataframes['df_op'].loc[results_dataframes['df_op'].query('N_1_bf_bt == "-1_-1"').index,'N_1_bf_bt']='0_0'
grouped_df_op = results_dataframes['df_op'].groupby(['N_1_bf_bt','Stability'])['case_id'].count()
grouped_df_op

#%%
grouped_df_op = pd.DataFrame(grouped_df_op)
grouped_df_op = grouped_df_op.drop(grouped_df_op.query('Stability == -2').index,axis=0)


grouped_df_op['percentage'] = (
    grouped_df_op['case_id']
    / grouped_df_op.groupby(level='N_1_bf_bt')['case_id'].transform('sum')
    * 100
)

#%%

import matplotlib.pyplot as plt

# Convert MultiIndex to columns
plot_df = grouped_df_op['percentage'].unstack(fill_value=0)

x = plot_df.get(-1, 0)  # Stability = -1
y = plot_df.get(0, 0)   # Stability = 0

plt.figure(figsize=(8, 6))

plt.scatter(x, y)

for topology in plot_df.index:
    plt.annotate(
        topology,
        (x.loc[topology], y.loc[topology]),
        xytext=(5, 5),
        textcoords='offset points'
    )

plt.xlabel('Non-feasible cases (\%)')
plt.ylabel('Unstable cases (\%)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
