#!/usr/bin/env python
# coding: utf-8

# # Data Generation Post-processing

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

# In[3]:


path = '../results/MareNostrum'

dir_name = 'datagen_ACOPF_slurm23172357_cu10_nodes32_LF09_seed3_nc3_ns500_d7_20250627_214226_7664-20250630T085420Z-1-005'

path_results = os.path.join(path, dir_name)

results_dataframes, csv_files = open_csv(path_results, ['cases_df.csv', 'case_df_op.csv'])


# In[4]:


columns_in_df = dict()
for key, item in results_dataframes.items():
    columns_in_df[key] = list(results_dataframes[key].columns)


# In[5]:


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
        
for key, item in columns_in_df.items():
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
# - case_df_op: after power flow quantities
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

# In[6]:


# %% ---- FILL NAN VALUES WITH NULL ---

results_dataframes['case_df_op'] = results_dataframes['case_df_op'].fillna(0)

# %% ---- FIX VALUES ----

Sn_cols = [col for col in results_dataframes['case_df_op']
           if col.startswith('Sn')]
results_dataframes['case_df_op'][Sn_cols] = results_dataframes['case_df_op'][Sn_cols]/100 #p.u. system base 100 MVA

theta_cols = [col for col in results_dataframes['case_df_op']
              if col.startswith('theta')]
# Adjust angles greater than 180°
results_dataframes['case_df_op'][theta_cols] = results_dataframes['case_df_op'][theta_cols] - \
    (results_dataframes['case_df_op'][theta_cols] > 180) * 360

results_dataframes['case_df_op'][theta_cols] = results_dataframes['case_df_op'][theta_cols] * np.pi/180

# add total demand variables
PL_cols = [
    col for col in results_dataframes['case_df_op'].columns if col.startswith('PL')]
results_dataframes['case_df_op']['PD'] = results_dataframes['case_df_op'][PL_cols].sum(
    axis=1)

QL_cols = [
    col for col in results_dataframes['case_df_op'].columns if col.startswith('QL')]
results_dataframes['case_df_op']['QD'] = results_dataframes['case_df_op'][QL_cols].sum(
    axis=1)


# ### Data Set Composition

# In[7]:


perc_stability(results_dataframes['case_df_op'], dir_name)


# In[8]:


# %% ---- SELECT ONLY FEASIBLE CASES ----
# from data frame with power flow results: case_df_op
results_dataframes['case_df_op_feasible'] = results_dataframes['case_df_op'].query(
    'Stability >= 0')

# from data frame with sampled quantities: cases_df
results_dataframes['cases_df_feasible'] = results_dataframes['cases_df'].query(
    'Stability >= 0')
case_id_feasible = list(results_dataframes['case_df_op_feasible']['case_id'])

# %% ---- SELECT ONLY UNFEASIBLE CASES (from data frame with sampled quantities: cases_df)----

results_dataframes['cases_df_unfeasible'] = results_dataframes['cases_df'].query('Stability < 0')
results_dataframes['cases_df_unfeasible_1'] = results_dataframes['cases_df'].query('Stability == -1')
results_dataframes['cases_df_unfeasible_2'] = results_dataframes['cases_df'].query('Stability == -2')

case_id_Unfeasible = list(results_dataframes['cases_df_unfeasible']['case_id'])
case_id_Unfeasible1 = list(results_dataframes['cases_df_unfeasible_1']['case_id'])
case_id_Unfeasible2 = list(results_dataframes['cases_df_unfeasible_2']['case_id'])


# In[9]:


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


# In[10]:


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


# Mesh obtained from parsing the logs file of the data generator process: the mesh shows the cell splitting process. It is obtained from the parsing_dimensions.py code.
# 
# Plot the mesh on top of the OPs scatter plot.

# In[12]:


mesh_df = pd.read_excel('mesh.xlsx')


# In[13]:


fig, ax = plt.subplots()
ax.scatter(dimensions_caseid_unfeasible['p_cig'], dimensions_caseid_unfeasible['p_sg'],color='silver', label='Unfeasable OP')
ax.scatter(dimensions_caseid_feasible.query('Stability ==0')['p_cig'], dimensions_caseid_feasible.query('Stability ==0')['p_sg'], color='r',label='Unstable PF')
ax.scatter(dimensions_caseid_feasible.query('Stability ==1')['p_cig'], dimensions_caseid_feasible.query('Stability ==1')['p_sg'], color='g', label='Stable PF')
ax.set_xlabel('$P_{CIG}$ [MW]')
ax.set_ylabel('$P_{SG}$ [MW]')
plot_mesh(mesh_df, ax)
plt.legend()


# In[14]:


fig, ax = plt.subplots()
ax.scatter(dimensions_caseid_unfeasible['p_cig'], dimensions_caseid_unfeasible['p_sg'],color='silver', label='Unfeasable OP')
ax.scatter(dimensions_caseid_feasible_sampled.query('Stability ==0')['p_cig'], dimensions_caseid_feasible_sampled.query('Stability ==0')['p_sg'], color='r',label='Unstable Sampled OP')
ax.scatter(dimensions_caseid_feasible_sampled.query('Stability ==1')['p_cig'], dimensions_caseid_feasible_sampled.query('Stability ==1')['p_sg'], color='g', label='Stable Sampled OP')
ax.set_xlabel('$P_{CIG}$ [MW]')
ax.set_ylabel('$P_{SG}$ [MW]')
plot_mesh(mesh_df, ax)
plt.legend()


# Dataframes with the cases_id, the exploration depth at which they have been evaluated and the corresponding cell name (as in the cell_info.csv file).
# It is obtained from the parsing_dimensions.py code.

# In[15]:


df_depth = pd.read_excel('cases_id_depth.xlsx')
df_depth


# In[16]:


df_feasibility_balancing = pd.DataFrame(columns=['depth','feasibility','cumulative_feasibility','feasiblity_no_2','balance','cumulative_balancing'])
cum_case_id_depth=[]
for idx, depth in enumerate(df_depth['Depth'].unique()):
    df_feasibility_balancing.loc[idx, 'depth']=depth
    case_id_depth = df_depth.query('Depth == @depth')['case_id']
    cum_case_id_depth.extend(case_id_depth)
    feas_case_id_depth = list(set(case_id_depth) & set(case_id_feasible))
    cum_feas_case_id_depth = list(set(cum_case_id_depth) & set(case_id_feasible))
    
    df_feasibility_balancing.loc[idx, 'feasibility']= len(feas_case_id_depth)/len(case_id_depth) 
    df_feasibility_balancing.loc[idx, 'cumulative_feasibility']= len(cum_feas_case_id_depth)/len(cum_case_id_depth) 

    feas_stab_depth = len(results_dataframes['cases_df_feasible'].query('case_id == @feas_case_id_depth and Stability ==1'))
    cum_feas_stab_case_id_depth = len(results_dataframes['cases_df_feasible'].query('case_id == @cum_feas_case_id_depth and Stability ==1'))
    
    df_feasibility_balancing.loc[idx, 'balance']= feas_stab_depth/len(feas_case_id_depth) 
    df_feasibility_balancing.loc[idx, 'cumulative_balancing']= cum_feas_stab_case_id_depth/len(cum_feas_case_id_depth) 
    


# In[17]:


df_feasibility_balancing


# In[18]:


import matplotlib.pyplot as plt

# Assuming your DataFrame is called df
# Plot 1: Feasibility
plt.figure(figsize=(10, 5))
plt.plot(df_feasibility_balancing['depth'], df_feasibility_balancing['feasibility'], marker='o', label='Feasibility')
plt.plot(df_feasibility_balancing['depth'], df_feasibility_balancing['cumulative_feasibility'], marker='s', label='Cumulative Feasibility')
plt.xlabel('Depth')
plt.ylabel('Feasibility')
plt.title('Feasibility vs Depth')
plt.grid(True)
plt.legend()
plt.show()


# In[19]:


# Plot 2: Balancing
plt.figure(figsize=(10, 5))
plt.plot(df_feasibility_balancing['depth'], df_feasibility_balancing['balance'], marker='o', label='Balance')
plt.plot(df_feasibility_balancing['depth'], df_feasibility_balancing['cumulative_balancing'], marker='s', label='Cumulative Balance')
plt.xlabel('Depth')
plt.ylabel('Balance')
plt.title('Balance vs Depth')
plt.grid(True)
plt.legend()
plt.show()


# ## Training Stability Assessment Models
# ### Data Cleaning

# In[20]:


columns_in_df = dict()
for key, item in results_dataframes.items():
    columns_in_df[key] = results_dataframes[key].columns


# In[21]:


# %% ----  Remove columns with only 1 value ----
columns_with_single_values = []
for c in columns_in_df['case_df_op_feasible']:
    if results_dataframes['case_df_op_feasible'][c].unique().size == 1:
        columns_with_single_values.append(c)
# --> if there is something different from Sn_SGX check, otherwise it is normal (no changes in SG installed power)
print(columns_with_single_values)

results_dataframes['case_df_op_feasible'] = results_dataframes['case_df_op_feasible'].drop(
    columns_with_single_values, axis=1)


# In[22]:


# Create dataframe with only input quantities
results_dataframes['case_df_op_feasible_X'] = results_dataframes['case_df_op_feasible'].drop(['case_id', 'Stability'], axis=1)


# In[23]:


# %% ---- Check correlated variables Option #1 ----
def get_correlated_columns(df, c_threshold=0.95, method='pearson'):

    correlated_features_tuples = []
    correlated_features = pd.DataFrame(columns=['Feat1', 'Feat2', 'Corr'])
    correlation = df.corr(method=method)
    count = 0
    for i in correlation.index:
        for j in correlation:
            if i != j and abs(correlation.loc[i, j]) >= c_threshold:
                # if tuple([j,i]) not in correlated_features_tuples:
                correlated_features_tuples.append(tuple([i, j]))
                correlated_features.loc[count, 'Feat1'] = i
                correlated_features.loc[count, 'Feat2'] = j
                correlated_features.loc[count, 'Corr'] = correlation.loc[i, j]
                count = count+1

    return correlated_features


correlated_features = get_correlated_columns(
    results_dataframes['case_df_op_feasible_X'])

grouped_corr_feat = correlated_features.groupby('Feat1').count().reset_index()

keep_var=[]
while not grouped_corr_feat.empty:
    # Pick the first remaining Feat1
    var = grouped_corr_feat.iloc[0]['Feat1']
    keep_var.append(var)

    # Find all features correlated with this one
    to_remove = correlated_features.query('Feat1 == @var')['Feat2'].tolist()

    # Drop all rows where Feat1 is in to_remove
    grouped_corr_feat = grouped_corr_feat[~grouped_corr_feat['Feat1'].isin(to_remove)]
    grouped_corr_feat = grouped_corr_feat[~grouped_corr_feat['Feat1'].isin(keep_var)]

df_taus = results_dataframes['case_df_op_feasible'][['case_id']].merge(results_dataframes['cases_df_feasible'][[
                                                                       col for col in columns_in_df['cases_df_feasible'] if col.startswith('tau_droop')]+['case_id']], on='case_id', how='left').drop(['case_id'], axis=1)

results_dataframes['case_df_op_feasible_uncorr_X'] = pd.concat([results_dataframes['case_df_op_feasible_X'][keep_var].reset_index(drop=True), df_taus],axis=1)
results_dataframes['case_df_op_feasible_uncorr'] = results_dataframes['case_df_op_feasible_uncorr_X']
results_dataframes['case_df_op_feasible_uncorr']['case_id'] = results_dataframes['case_df_op_feasible']['case_id'].reset_index(drop=True)
results_dataframes['case_df_op_feasible_uncorr']['Stability'] = results_dataframes['case_df_op_feasible']['Stability'].reset_index(drop=True)

results_dataframes['case_df_op_feasible_uncorr'].to_csv('DataSet_training_uncorr_var.csv')


# In[26]:


print_columns_groups('case_df_op_feasible_uncorr', results_dataframes['case_df_op_feasible_uncorr'].columns)


# In[27]:


# %% ---- Check correlated variables Option #2 ----

results = pd.concat([results_dataframes['case_df_op_feasible_X'].reset_index(drop=True), df_taus.reset_index(drop=True)], axis=1).apply(
#results = results_dataframes['case_df_op_feasible_X'].reset_index(drop=True).apply(
     lambda col: pointbiserialr(col, results_dataframes['case_df_op_feasible']['Stability']), result_type='expand').T
results.columns = ['correlation', 'p_value']
results['abs_corr'] = abs(results['correlation'])

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
X = pd.concat([results_dataframes['case_df_op_feasible_X'].reset_index(
    drop=True), df_taus.reset_index(drop=True)], axis=1)
# X = results_dataframes['case_df_op_feasible_X'].reset_index(
#     drop=True)
corr = spearmanr(X).correlation

# Ensure the correlation matrix is symmetric
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)

# We convert the correlation matrix to a distance matrix before performing
# hierarchical clustering using Ward's linkage.
distance_matrix = 1 - np.abs(corr)
dist_linkage = hierarchy.ward(squareform(distance_matrix))
# dendro = hierarchy.dendrogram(
#     dist_linkage, labels=X.columns.to_list(), ax=ax1, leaf_rotation=90
# )
# dendro_idx = np.arange(0, len(dendro["ivl"]))

# ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
# ax2.set_xticks(dendro_idx)
# ax2.set_yticks(dendro_idx)
# ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
# ax2.set_yticklabels(dendro["ivl"])
# _ = fig.tight_layout()


cluster_ids = hierarchy.fcluster(dist_linkage, 0.01, criterion="distance")
cluster_id_to_feature_ids = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)

selected_features_names_dict = defaultdict(list)
for i, selected_features in cluster_id_to_feature_ids.items():
    selected_features_names_dict[i]=X.columns[selected_features]

keep_var = []
for i, selected_features in selected_features_names_dict.items():
    if len(selected_features)==1:
        keep_var.append(selected_features[0])
    elif len(selected_features)>1:
         keep_var.append(results.loc[selected_features,'abs_corr'].sort_values(ascending=False).index[0])

results_dataframes['case_df_op_feasible_uncorr_HierCl_X'] = X[keep_var]
results_dataframes['case_df_op_feasible_uncorr_HierCl'] = pd.concat([X[keep_var], results_dataframes['case_df_op_feasible'][['case_id', 'Stability']].reset_index(drop=True)],axis=1)

results_dataframes['case_df_op_feasible_uncorr_HierCl'].to_csv('DataSet_training_uncorr_var_HierCl.csv')


# In[28]:


print_columns_groups('case_df_op_feasible_uncorr_HierCl', results_dataframes['case_df_op_feasible_uncorr_HierCl'].columns)


# In[29]:


from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def kfold_cv_depth(df, type_corr_analysis, plot_depth_exploration=True, n_fold=5):
    df_training = pd.DataFrame(columns= df.columns)
    cases_id_training = []
    scores_df=pd.DataFrame(columns=['Depth','score_mean','score_std','n_training_cases','perc_stable'])

    if plot_depth_exploration:
        ax = plot_mesh(mesh_df)
    
    for depth in range(0,7):
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

            if plot_depth_exploration:
                ax.scatter(results_dataframes['cases_df_feasible'].query('case_id == @add_case_id')['p_cig'],
                       results_dataframes['cases_df_feasible'].query('case_id == @add_case_id')['p_sg'], label = 'Depth '+str(depth))
    plt.legend()
    return scores_df
    


# In[ ]:


#%%
    pd.DataFrame.to_excel(scores_df,'scores_df_'+type_corr_analysis+'_xgb.xlsx')
    
    #%%
    fig, ax = plt.subplots()
    ax.errorbar(scores_df['Depth'], scores_df['score_mean'], yerr=scores_df['score_std'], fmt='-o', capsize=5, color='blue', ecolor='black', elinewidth=1.5)
    ax.set_xlabel('Depth')
    ax.set_ylabel('Mean accuracy $\pm$ std')
    ax.grid()
    fig.tight_layout()
    plt.savefig('scores_vs_depth__df_uncorr_var_HierCl_xgb.pdf')#, format='pdf')
    plt.savefig('scores_vs_depth__df_uncorr_var_HierCl_xgb.png')#, format='png')

