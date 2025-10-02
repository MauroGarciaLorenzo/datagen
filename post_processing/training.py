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
    #'datagen_ACOPF_slurm23172357_cu10_nodes32_LF09_seed3_nc3_ns500_d7_20250627_214226_7664']
    'datagen_ACOPF_slurm25105245_cu8_nodes32_LF09_seed3_nc3_ns500_d7_20250731_132256_7665']

for dir_name in dir_names:
    path_results = os.path.join(path, dir_name)

    results_dataframes, csv_files = open_csv(
        path_results, ['cases_df.csv', 'case_df_op.csv'])

    perc_stability(results_dataframes['cases_df'], dir_name)
    
    dataset_ID = dir_name[-5:]

for key, item in results_dataframes.items():
    print(key+': '+str(len(item)))
    #results_dataframes[key+'_drop_duplicates']= item.drop(['case_id'],axis=1).drop_duplicates(keep='first')
    print(key+'_drop_duplicates'+': '+str(len(item.drop_duplicates(keep='first'))))


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
cases_id_depth = pd.read_excel('cases_id_depth'+dataset_ID+'.xlsx')[['Depth','case_id','CellName']]

cases_id_depth_feas = cases_id_depth.query('case_id == @case_id_feasible')

#%%
df = pd.read_csv('DataSet_training_uncorr_var_HierCl'+dataset_ID+'.csv').drop('Unnamed: 0', axis=1).drop_duplicates(keep='first')
             #_var_HierCl
             
p_cig_cols = [col for col in results_dataframes['cases_df_feasible'].columns if col.startswith('p_cig')]
p_sg_cols = [col for col in results_dataframes['cases_df_feasible'].columns if col.startswith('p_sg')]
results_dataframes['cases_df_feasible']['p_cig'] = results_dataframes['cases_df_feasible'][p_cig_cols].sum(axis=1)
results_dataframes['cases_df_feasible']['p_sg'] = results_dataframes['cases_df_feasible'][p_sg_cols].sum(axis=1)

exclude_cases = list(results_dataframes['cases_df_feasible'].query('p_cig > 2738.010789')['case_id'])
df = df.query('case_id != @exclude_cases')

#%%
mesh_df = pd.read_excel('mesh'+dataset_ID+'.xlsx')

ax = plot_mesh(mesh_df)
for depth in range(0,7):
        add_case_id = list(cases_id_depth.query('Depth == @depth')['case_id'])
        ax.scatter(results_dataframes['cases_df_feasible'].query('case_id == @add_case_id')['p_cig'],
                   results_dataframes['cases_df_feasible'].query('case_id == @add_case_id')['p_sg'], label = 'Depth '+str(depth))
plt.legend()


#%%

from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

n_fold = 5 
# n_cases_training=0
df_training = pd.DataFrame(columns= df.columns)
cases_id_training = []

scores_df=pd.DataFrame(columns=['Depth','score_mean','score_std','n_training_cases','perc_stable'])
ax = plot_mesh(mesh_df)

for depth in range(0,7):
    #n_cases_training = n_cases_training + len(cases_id_depth_feas.query('Depth == @depth'))
    #print(n_cases_training)
    add_case_id = list(cases_id_depth_feas.query('Depth == @depth')['case_id'])
    cases_id_training.extend(add_case_id)
    df_training = df.query('case_id == @cases_id_training')
    scores_df.loc[depth,'n_training_cases']=len(df_training)
    scores_df.loc[depth,'perc_stable']=len(df_training.query('Stability == 1'))/len(df_training)

    if len(df_training)>= n_fold:
        #clf = svm.SVC(kernel='linear', C=1, random_state=42)
        #clf = MLPClassifier(random_state=1, max_iter=5000, activation='relu')
        clf = Pipeline([('scaler', StandardScaler()), ('xgb', XGBClassifier())])
        X = df_training.drop(['case_id','Stability'],axis=1).reset_index(drop=True)
        y = df_training[['Stability']].reset_index(drop=True).values.astype(int).ravel()
        scores = cross_val_score(clf, X, y, cv=n_fold, scoring='accuracy')
        
        scores_df.loc[depth,'Depth']=depth
        scores_df.loc[depth,'score_mean']=scores.mean()
        scores_df.loc[depth,'score_std']=scores.std()
        
        ax.scatter(results_dataframes['cases_df_feasible'].query('case_id == @add_case_id')['p_cig'],
                   results_dataframes['cases_df_feasible'].query('case_id == @add_case_id')['p_sg'], label = 'Depth '+str(depth))
plt.legend()
#%%
pd.DataFrame.to_excel(scores_df,'scores_df_uncorr_var_HierCl_xgb'+dataset_ID+'.xlsx')#_var_HierCl_

#%%
fig, ax = plt.subplots()
ax.errorbar(scores_df['Depth'], scores_df['score_mean'], yerr=scores_df['score_std'], fmt='-o', capsize=5, color='blue', ecolor='black', elinewidth=1.5)
ax.set_xlabel('Depth')
ax.set_ylabel('Mean accuracy $\pm$ std')
ax.grid()
fig.tight_layout()
plt.savefig('scores_vs_depth__df_uncorr_var_HierCl_xgb'+dataset_ID+'.pdf')#, format='pdf')
plt.savefig('scores_vs_depth__df_uncorr_var_HierCl_xgb'+dataset_ID+'.png')#, format='png')