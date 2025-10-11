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
path = '~/projects/HP2C-DT'

# dir_names=[dir_name for dir_name in os.listdir(path) if dir_name.startswith('datagen') and 'zip' not in dir_name]#

dir_names = [
    "datagen_ACOPF_cu8_nodes32_LF09_seed3_nc3_ns333_d5_20251030_sensitivity"]

for dir_name in dir_names:
    path_results = os.path.join(path, dir_name)

    results_dataframes, csv_files = open_csv(
        path_results, ['cases_df.csv'])

    perc_stability(results_dataframes['cases_df'], dir_name)

for key, item in results_dataframes.items():
    print(key+': '+str(len(item)))
    #results_dataframes[key+'_drop_duplicates']= item.drop(['case_id'],axis=1).drop_duplicates(keep='first')
    print(key+'_drop_duplicates'+': '+str(len(item.drop_duplicates(keep='first'))))

# %% ---- FILL NAN VALUES WITH NULL ---

#results_dataframes['case_df_op'] = results_dataframes['case_df_op'].fillna(0)

# %% ---- SELECT ONLY FEASIBLE CASES ----

results_dataframes['cases_df_feasible'] = results_dataframes['cases_df'].query(
    'Stability >= 0')

case_id_feasible = list(results_dataframes['cases_df_feasible']['case_id'])

# case_id=case_id_feasible[0]
# results_dataframes['case_df_op_feasible'].query('case_id == @case_id')['P_SG12'] <--- quantities calculated by power flow
# results_dataframes['cases_df'].query('case_id == @case_id')['p_sg_Var10'] <-- quantities sampled

results_dataframes['cases_df_feasible'] = results_dataframes['cases_df'].query(
    'case_id == @case_id_feasible')  # <-- quantities sampled

n_feas_cases = len(case_id_feasible)
n_cases = len(results_dataframes['cases_df'])

results_dataframes['cases_df_feasible'].drop_duplicates(inplace=True)
X = results_dataframes['cases_df_feasible'].drop(['case_id', "Stability", "cell_name"], axis=1)
y = results_dataframes['cases_df_feasible']["Stability"]


# %% Load dummy
random_dir = os.path.join(path, "datagen_ACOPF_slurm30511997_cu10_nodes40_LF09_seed3_nc1_ns127500_d0_20251010_043642_4480_random")
random_file = os.path.join(random_dir, "cases_df.csv")
random_cases_df = pd.read_csv(random_file)
perc_stability(random_cases_df, "")
# random_cases_df = random_cases_df.sample(n=n_cases)
# Filter out feasible cases for the dummy scenario
random_cases_df_feasible = random_cases_df.query('Stability >= 0')
random_cases_df_feasible = random_cases_df_feasible.sample(n=len(X))
X_random = random_cases_df_feasible.drop(["Stability", "cell_name", "case_id"], axis=1)
y_random = random_cases_df_feasible["Stability"]

# %% ---- SELECT ONLY UNFEASIBLE CASES ----
"""
results_dataframes['cases_df_unfeasible'] = results_dataframes['cases_df'].query('Stability < 0')
results_dataframes['cases_df_unfeasible_1'] = results_dataframes['cases_df'].query('Stability == -1')
results_dataframes['cases_df_unfeasible_2'] = results_dataframes['cases_df'].query('Stability == -2')

case_id_Unfeasible = list(results_dataframes['cases_df_unfeasible']['case_id'])
case_id_Unfeasible1 = list(results_dataframes['cases_df_unfeasible_1']['case_id'])
case_id_Unfeasible2 = list(results_dataframes['cases_df_unfeasible_2']['case_id'])

results_dataframes['cases_df_unfeasible'] = results_dataframes['cases_df'].query(
    'case_id == @case_id_Unfeasible')  # <-- quantities sampled
results_dataframes['cases_df_unfeasible_1'] = results_dataframes['cases_df'].query(
    'case_id == @case_id_Unfeasible1')  # <-- quantities sampled
results_dataframes['cases_df_unfeasible_2'] = results_dataframes['cases_df'].query(
    'case_id == @case_id_Unfeasible2')  # <-- quantities sampled
"""
#%%
# cases_id_depth = pd.read_excel('cases_id_depth.xlsx')[['Depth','case_id','CellName']]

# cases_id_depth_feas = cases_id_depth.query('case_id == @case_id_feasible')

#%%
#df = results_dataframes['X'].drop_duplicates(keep='first')
             #_var_HierCl
             
#p_cig_cols = [col for col in results_dataframes['cases_df_feasible'].columns if col.startswith('p_cig')]
#p_sg_cols = [col for col in results_dataframes['cases_df_feasible'].columns if col.startswith('p_sg')]
#results_dataframes['cases_df_feasible']['p_cig'] = results_dataframes['cases_df_feasible'][p_cig_cols].sum(axis=1)
#results_dataframes['cases_df_feasible']['p_sg'] = results_dataframes['cases_df_feasible'][p_sg_cols].sum(axis=1)

#exclude_cases = list(results_dataframes['cases_df_feasible'].query('p_cig > 2738.010789')['case_id'])
#df = df.query('case_id != @exclude_cases')

#%%
"""mesh_df = pd.read_excel('mesh.xlsx')

ax = plot_mesh(mesh_df)
for depth in range(0,7):
        add_case_id = list(cases_id_depth.query('Depth == @depth')['case_id'])
        ax.scatter(results_dataframes['cases_df_feasible'].query('case_id == @add_case_id')['p_cig'],
                   results_dataframes['cases_df_feasible'].query('case_id == @add_case_id')['p_sg'], label = 'Depth '+str(depth))
plt.legend()"""


#%%

from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

n_fold = 5 
# n_cases_training=0
# df_training = pd.DataFrame(columns= df.columns)
cases_id_training = []

scores_df=pd.DataFrame(columns=['Depth','score_mean','score_std','n_training_cases','perc_stable'])
#ax = plot_mesh(mesh_df)

#%% Coverage
#%% PCA (fit only on baseline data)
pca = PCA(n_components=3)
X_random_pca = pd.DataFrame(pca.fit_transform(X_random), columns=["PC1", "PC2", "PC3"])
# Sort X features
X = X[X_random.columns]
X_algo_pca = pd.DataFrame(pca.transform(X), columns=["PC1", "PC2", "PC3"])

# Normalize to [0,1] cube in PCA space for binning
scaler = MinMaxScaler()
X_random_scaled = pd.DataFrame(scaler.fit_transform(X_random_pca), columns=X_random_pca.columns)
X_algo_scaled = pd.DataFrame(scaler.transform(X_algo_pca), columns=X_algo_pca.columns)

#%% Discretize into 10x10x10 grid
def bin_coordinates(df, bins=10):
    """
    Assume normalized input (0, 1 range) and bin each feature from 0 to bins.
    If you have 3 features and bin=10, you end up with 10x10x10=1000
    combinations of bins (0,0,0), (0,0,1), ... until (9,9,9).
    i.e., 1000 regions
    """
    # Scale the data by the number of bins
    scaled_values = df * bins
    # Take the floor to assign each value to a bin index
    floored_values = np.floor(scaled_values)
    # Convert the floating-point bin indices to integers
    int_bins = floored_values.astype(int)
    # Clip values to ensure bin indices fall within the valid range [0, bins-1]
    #  (you could have exactly 10.0 and that would floor to 10)
    clipped_bins = np.clip(int_bins, 0, bins - 1)
    return clipped_bins

bins_random = bin_coordinates(X_random_scaled)
bins_algo = bin_coordinates(X_algo_scaled)


#%% Compute bin-wise entropy on random baseline
bin_index_random = list(map(tuple, bins_random.values))
bin_index_algo = list(map(tuple, bins_algo.values))

# Group labels by bin
from collections import defaultdict

# Group entropy values by bin
bin_entropy_values = defaultdict(list)
for idx, ent in zip(bin_index_random, y_random):
    bin_entropy_values[idx].append(ent)

# Average per-bin entropy
bin_entropy = {b: np.mean(ents) for b, ents in bin_entropy_values.items()}

# Select high-entropy bins (threshold > 0.2)
high_entropy_bins = {b for b, h in bin_entropy.items() if h > 0.2}


#%% Compare coverage
# Get which bins are hit
bins_hit_algo = set(bin_index_algo)
bins_hit_random = set(bin_index_random)

# How many high-entropy bins are hit
algo_hits = high_entropy_bins & bins_hit_algo
random_hits = high_entropy_bins & bins_hit_random

print(f"High-entropy bins total: {len(high_entropy_bins)}")
print(f"Your algorithm covers: {len(algo_hits)} bins")
print(f"Random baseline covers: {len(random_hits)} bins")

# How many samples fall into those bins
def count_samples_in_bins(bin_indices, target_bins):
    return sum(1 for b in bin_indices if b in target_bins)

algo_sample_count = count_samples_in_bins(bin_index_algo, high_entropy_bins)
random_sample_count = count_samples_in_bins(bin_index_random, high_entropy_bins)

print(f"Samples from your method in high-entropy bins: {algo_sample_count}")
print(f"Samples from baseline in high-entropy bins: {random_sample_count}")

# Average entropy in bins hit by each method
def average_entropy_in_bins(bin_indices, valid_bins, entropy_dict):
    bins = set(bin_indices) & valid_bins
    if not bins:
        return 0.0
    return np.mean([entropy_dict[b] for b in bins])

avg_entropy_algo = average_entropy_in_bins(bin_index_algo, high_entropy_bins, bin_entropy)
avg_entropy_random = average_entropy_in_bins(bin_index_random, high_entropy_bins, bin_entropy)

print(f"Average entropy in bins hit by your method: {avg_entropy_algo:.3f}")
print(f"Average entropy in bins hit by random baseline: {avg_entropy_random:.3f}")


"""
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
pd.DataFrame.to_excel(scores_df,'scores_df_uncorr_var_HierCl_xgb.xlsx')#_var_HierCl_

#%%
fig, ax = plt.subplots()
ax.errorbar(scores_df['Depth'], scores_df['score_mean'], yerr=scores_df['score_std'], fmt='-o', capsize=5, color='blue', ecolor='black', elinewidth=1.5)
ax.set_xlabel('Depth')
ax.set_ylabel('Mean accuracy $\pm$ std')
ax.grid()
fig.tight_layout()
plt.savefig('scores_vs_depth__df_uncorr_var_HierCl_xgb.pdf')#, format='pdf')
plt.savefig('scores_vs_depth__df_uncorr_var_HierCl_xgb.png')#, format='png')
"""