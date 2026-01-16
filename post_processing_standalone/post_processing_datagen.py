import os
from utils_pp_standalone import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
#%%
#ncig_list=['10','11','12']
path='../results/MareNostrum'

#dir_names=[dir_name for dir_name in os.listdir(path) if dir_name.startswith('datagen') and 'zip' not in dir_name]#

dir_names=['datagen_ACOPF_slurm23172357_cu10_nodes32_LF09_seed3_nc3_ns500_d7_20250627_214226_7664-20250630T085420Z-1-005']

for dir_name in dir_names:
    path_results = os.path.join(path, dir_name)
    
    results_dataframes, csv_files= open_csv(path_results)
    
    perc_stability(results_dataframes['case_df_op'],dir_name)

#%%
columns_in_df=dict()
for key, item in results_dataframes.items():
    print(key)
    columns_in_df[key]=results_dataframes[key].columns
    
#%% ---- SELECT ONLY FEASIBLE CASES ----

results_dataframes['case_df_op_feasible']=results_dataframes['case_df_op'].query('Stability >= 0')

case_id_feasible= list(results_dataframes['case_df_op_feasible']['case_id'])

# case_id=case_id_feasible[0]
# results_dataframes['case_df_op_feasible'].query('case_id == @case_id')['P_SG12'] <--- quantities calculated by power flow
# results_dataframes['cases_df'].query('case_id == @case_id')['p_sg_Var10'] <-- quantities sampled

results_dataframes['cases_df_feasible']=results_dataframes['cases_df'].query('case_id == @case_id_feasible') #<-- quantities sampled

n_feas_cases = len(case_id_feasible)

#%% ----  Remove columns with only 1 value ----
# columns_with_single_values=[]
# for c in columns_in_df['case_df_op_feasible']:
#     if results_dataframes['case_df_op_feasible'][c].unique().size == 1:
#         columns_with_single_values.append(c)
# print(columns_with_single_values) # --> if there is something different from Sn_SGX check, otherwise it is normal (no changes in SG installed power)

# results_dataframes['case_df_op_feasible']=results_dataframes['case_df_op_feasible'].drop(columns_with_single_values,axis=1)

#%% ---- FILL NAN VALUES WITH NULL ---

results_dataframes['case_df_op_feasible']=results_dataframes['case_df_op_feasible'].fillna(0)

#%%
columns_in_df['case_df_op_feasible']= results_dataframes['case_df_op_feasible'].columns

results_dataframes['case_df_op_feasible_X']=results_dataframes['case_df_op_feasible'].drop(['case_id','Stability'],axis=1)

#%%
# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(results_dataframes['case_df_op_feasible_X'])

pca_full = PCA()
pca_full.fit(scaled_data)
explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

X_PCA = pca_full.fit_transform(scaled_data)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.title('Cumulative Explained Variance by Number of Dimensions')
plt.xlabel('Number of Dimensions')
plt.ylabel('Cumulative Explained Variance')
plt.xlim(1, len(cumulative_variance))
plt.ylim(0, 1)
plt.grid()
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance Threshold')  # Optional threshold line
plt.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot()
ax.bar(range(1,len(explained_variance)+1), explained_variance, alpha=0.5, align='center', label='Individual explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.tick_params(axis='x')
plt.tick_params(axis='y')
ax.set_xlim(0,20)
# ax.set_xticks([1,2,3])
# ax.set_xticklabels(['$X_A$','$X_B$','$X_C$'])
plt.tight_layout()

def biplot(score, coeff, labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    plt.figure(figsize=(10,7))
    plt.scatter(xs, ys, alpha=0.5)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0]*5, coeff[i,1]*5, color='r', alpha=0.7)
        if labels is None:
            plt.text(coeff[i,0]*5.2, coeff[i,1]*5.2, f"Var{i+1}", color='g', ha='center', va='center')
        else:
            plt.text(coeff[i,0]*5.2, coeff[i,1]*5.2, labels[i], color='g', ha='center', va='center')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid()
    plt.title("PCA Biplot")
    plt.tight_layout()
    plt.show()

biplot(X_PCA, pca_full.components_.T[0:2,:], labels=results_dataframes['case_df_op_feasible_X'].columns)

#%%

idx_stab=results_dataframes['case_df_op_feasible'].reset_index(drop=True).query('Stability == 1').index
idx_unstab=results_dataframes['case_df_op_feasible'].reset_index(drop=True).query('Stability == 0').index
 
fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(X_PCA[idx_unstab,0],X_PCA[idx_unstab,1],c='r',alpha=0.2)
ax.scatter(X_PCA[idx_stab,0],X_PCA[idx_stab,1],c='g',alpha=0.2)

fig = plt.figure()
ax = fig.add_subplot()
ax.hist2d(X_PCA[idx_stab,0],X_PCA[idx_stab,1], bins=(50, 50), cmap=plt.cm.Greens_r.reversed())
ax.hist2d(X_PCA[idx_unstab,0],X_PCA[idx_unstab,1], bins=(50, 50), cmap=plt.cm.Reds_r.reversed())

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

# Stable group histogram
h1, xedges1, yedges1 = np.histogram2d(X_PCA[idx_stab, 0], X_PCA[idx_stab, 2], bins=(50, 50))
h1_masked = np.ma.masked_where(h1 == 0, h1)
ax.pcolormesh(xedges1, yedges1, h1_masked.T, cmap=plt.cm.Greens_r.reversed())

# Unstable group histogram
h2, xedges2, yedges2 = np.histogram2d(X_PCA[idx_unstab, 0], X_PCA[idx_unstab, 2], bins=(50, 50))
h2_masked = np.ma.masked_where(h2 == 0, h2)
ax.pcolormesh(xedges2, yedges2, h2_masked.T, cmap=plt.cm.Reds_r.reversed())

plt.show()


from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr


corr_matrix=results_dataframes['case_df_op_feasible_X'].corr()


#%% ----- build results matrices -----

n_buses = 118

def fill_and_sort(df_partial, prefix, total):
    if prefix.startswith('tau_droop'):
        all_cols = [f'{prefix}_{i}' for i in range(1, total + 1)]
    else:
        all_cols = [f'{prefix}{i}' for i in range(1, total + 1)]
    for col in all_cols:
        if col not in df_partial.columns:
            df_partial[col] = 0
    if prefix.startswith('tau_droop'):
        return df_partial[sorted(df_partial.columns, key=lambda x: int(x.split('_')[-1]))]
    else:
        return df_partial[sorted(df_partial.columns, key=lambda x: int(x.split(prefix)[-1]))]

# List of (prefix, source_df_key) for all variable blocks
prefixes_and_sources = [
    ('P_SG', 'case_df_op_feasible'),
    ('Q_SG', 'case_df_op_feasible'),
    ('P_GFOR', 'case_df_op_feasible'),
    ('Q_GFOR', 'case_df_op_feasible'),
    ('Sn_GFOR', 'case_df_op_feasible'),
    ('P_GFOL', 'case_df_op_feasible'),
    ('Q_GFOL', 'case_df_op_feasible'),
    ('Sn_GFOL', 'case_df_op_feasible'),
    ('PL', 'case_df_op_feasible'),
    ('QL', 'case_df_op_feasible'),
    ('V', 'case_df_op_feasible'),
    ('theta', 'case_df_op_feasible'),
    # ('tau_droop_f_gfor', 'cases_df_feasible'),
    # ('tau_droop_u_gfor', 'cases_df_feasible'),
    # ('tau_droop_f_gfol', 'cases_df_feasible'),
    # ('tau_droop_u_gfol', 'cases_df_feasible'),
]

# Dictionary to store results
processed_blocks = {}

# Process each prefix
for prefix, source_df_key in prefixes_and_sources:
    source_df = results_dataframes[source_df_key]
    column_names = [var for var in columns_in_df[source_df_key] if var.startswith(prefix)]
    df_partial = source_df[column_names]
    processed_blocks[prefix] = fill_and_sort(df_partial, prefix, n_buses)

# # Unpack if you want individual variables
# P_SG = processed_blocks['P_SG']
# Q_SG = processed_blocks['Q_SG']
# P_GFOR = processed_blocks['P_GFOR']
# Q_GFOR = processed_blocks['Q_GFOR']
# Sn_GFOR = processed_blocks['Sn_GFOR']
# P_GFOL = processed_blocks['P_GFOL']
# Q_GFOL = processed_blocks['Q_GFOL']
# Sn_GFOL = processed_blocks['Sn_GFOL']
# PL = processed_blocks['PL']
# QL = processed_blocks['QL']
# V = processed_blocks['V']
# theta = processed_blocks['theta']
## tau_droop_f_gfor = processed_blocks['tau_droop_f_gfor']
## tau_droop_u_gfor = processed_blocks['tau_droop_u_gfor']
## tau_droop_f_gfol = processed_blocks['tau_droop_f_gfol']
## tau_droop_u_gfol = processed_blocks['tau_droop_u_gfol']

prefixes_and_sources = [
    ('tau_droop_f_gfor', 'cases_df_feasible'),
    ('tau_droop_u_gfor', 'cases_df_feasible'),
    ('tau_droop_f_gfol', 'cases_df_feasible'),
    ('tau_droop_u_gfol', 'cases_df_feasible'),
]


for prefix, source_df_key in prefixes_and_sources:
    source_df = results_dataframes[source_df_key]
    column_names = [var for var in columns_in_df[source_df_key] if var.startswith(prefix)]
    
    df_partial = results_dataframes['case_df_op_feasible'][['case_id']].merge(source_df[column_names+['case_id']], on='case_id', how='left').drop(['case_id'],axis=1)
    processed_blocks[prefix] = fill_and_sort(df_partial, prefix, n_buses)

prefixes_and_sources = [
    ('Stability', 'case_df_op_feasible')]

for prefix, source_df_key in prefixes_and_sources:
    source_df = results_dataframes[source_df_key]
    column_names = [var for var in columns_in_df[source_df_key] if var.startswith(prefix)]
    df_partial = source_df[column_names]
    new_cols = [f'Stability{i}' for i in range(1, n_buses)]
    df_partial[new_cols] = pd.DataFrame({col: df_partial['Stability'].values for col in new_cols}, index=df_partial.index)
    processed_blocks[prefix] = df_partial

# Adjust angles greater than 180°
processed_blocks['theta'] = processed_blocks['theta'] - (processed_blocks['theta'] > 180) * 360

processed_blocks['theta']=processed_blocks['theta']*np.pi/180

processed_blocks['Sn_GFOL']=processed_blocks['Sn_GFOL']/100
processed_blocks['Sn_GFOR']=processed_blocks['Sn_GFOR']/100

#%%

processed_blocks_scaled= {}
scaler = MinMaxScaler()#StandardScaler()

for key, item in processed_blocks.items():
    processed_blocks_scaled[key] = scaler.fit_transform(item)
      

#%%

# cosphi_SG=np.cos(np.arctan(np.array(Q_SG)/np.array(P_SG)))

dataframes = list(processed_blocks_scaled.values())  # preserves order of insertion
        
array_3d = np.stack([df for df in dataframes])

array_3d = array_3d.transpose(1, 0, 2)

#%%
vmin = np.min(array_3d)
vmax = np.max(array_3d)

fig, ax = plt.subplots(10, 10, subplot_kw=dict(xticks=[], yticks=[]))
for i, axi in enumerate(ax.flat):
    pcm = axi.pcolormesh(array_3d[i], cmap='gray', shading='auto', vmin=vmin, vmax=vmax)

#%%

array_2d = np.empty((n_feas_cases,0))
for key, item in processed_blocks_scaled.items():
    array_2d = np.concatenate((array_2d, item), axis=1)
    
from sklearn.decomposition import PCA
pca = PCA(n_components=10, svd_solver='randomized')
model = PCA(100).fit(array_2d)

fig, ax = plt.subplots()
ax.plot(np.cumsum(model.explained_variance_ratio_))
ax.set_xlabel('n components')
ax.set_ylabel('cumulative variance');

from sklearn.manifold import Isomap
model = Isomap(n_components=2)
proj = model.fit_transform(array_2d)
proj.shape
