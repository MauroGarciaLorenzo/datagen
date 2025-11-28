from matplotlib import offsetbox
from collections import defaultdict
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
import os
from utils_pp_standalone import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA, KernelPCA
import seaborn as sns
from scipy.stats import pointbiserialr
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import copy
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
path = 'D:/'
dir_name=[dir_name for dir_name in os.listdir(path) if '_2862' in dir_name and 'zip' not in dir_name][0]# if dir_name.startswith('datagen') and 'zip' not in dir_name]#
print(dir_name)

#%%
#for dir_name in dir_names:
path_results = os.path.join(path, dir_name)
df_op='df_op'#'case_df_op'
results_dataframes, csv_files = open_csv(
    path_results, ['cases_df.csv', df_op+'.csv'])

perc_stability(results_dataframes[df_op], dir_name)

dataset_ID = dir_name[-5:]

for key, item in results_dataframes.items():
    print(key+': '+str(len(item)))
    #results_dataframes[key+'_drop_duplicates']= item.drop(['case_id'],axis=1).drop_duplicates(keep='first')
    print(key+'_drop_duplicates'+': '+str(len(item.drop_duplicates(keep='first'))))

#%%
# results_dataframes= dict()
# results_dataframes[df_op] = pd.DataFrame()
# results_dataframes['cases_df'] = pd.DataFrame()

# for dataset_ID in dataset_ID_list:
#     results_dataframes[df_op]  = pd.concat([results_dataframes[df_op], results_dataframes_datasets[dataset_ID][df_op]],axis=0).reset_index(drop=True)
#     results_dataframes['cases_df'] = pd.concat([results_dataframes['cases_df'], results_dataframes_datasets[dataset_ID]['cases_df']],axis=0).reset_index(drop=True)

# %% ---- FILL NAN VALUES WITH NULL ---

results_dataframes[df_op] = results_dataframes[df_op].fillna(0)

# %% ---- FIX VALUES ----

Sn_cols = [col for col in results_dataframes[df_op]
           if col.startswith('Sn')]
results_dataframes[df_op][Sn_cols] = results_dataframes[df_op][Sn_cols]/100

theta_cols = [col for col in results_dataframes[df_op]
              if col.startswith('theta')]
# # Adjust angles greater than 180°
# results_dataframes[df_op][theta_cols] = results_dataframes[df_op][theta_cols] - \
#     (results_dataframes[df_op][theta_cols] > 180) * 360

# results_dataframes['case_df_op'][theta_cols] = results_dataframes['case_df_op'][theta_cols] * np.pi/180

# add total demand variables
PL_cols = [col for col in results_dataframes[df_op].columns if col.startswith('PL')]
results_dataframes[df_op]['PD'] = results_dataframes[df_op][PL_cols].sum(
    axis=1)

QL_cols = [col for col in results_dataframes[df_op].columns if col.startswith('QL')]
results_dataframes[df_op]['QD'] = results_dataframes[df_op][QL_cols].sum(
    axis=1)

P_SG_cols = [col for col in results_dataframes[df_op].columns if col.startswith('P_SG')]
P_GFOL_cols = [col for col in results_dataframes[df_op].columns if col.startswith('P_GFOL')]
P_GFOR_cols = [col for col in results_dataframes[df_op].columns if col.startswith('P_GFOR')]

Q_SG_cols = [col for col in results_dataframes[df_op].columns if col.startswith('Q_SG')]
Q_GFOL_cols = [col for col in results_dataframes[df_op].columns if col.startswith('Q_GFOL')]
Q_GFOR_cols = [col for col in results_dataframes[df_op].columns if col.startswith('Q_GFOR')]


# %% ---- SELECT ONLY FEASIBLE CASES ----

results_dataframes['case_df_op_feasible'] = results_dataframes['df_op'].query(
    'Stability >= 0')

case_id_feasible = list(results_dataframes['case_df_op_feasible']['case_id'])

print(len(case_id_feasible))

print(len(set(case_id_feasible)))

results_dataframes['case_df_op_feasible'].groupby('case_id')['case_id'].count()

# case_id=case_id_feasible[0]
# results_dataframes['case_df_op_feasible'].query('case_id == @case_id')['P_SG12'] <--- quantities calculated by power flow
# results_dataframes['cases_df'].query('case_id == @case_id')['p_sg_Var10'] <-- quantities sampled

results_dataframes['cases_df_feasible'] = results_dataframes['cases_df'].query(
    'case_id == @case_id_feasible')  # <-- quantities sampled

print(len(results_dataframes['cases_df_feasible']['case_id']))

n_feas_cases = len(case_id_feasible)

results_dataframes['case_df_op_feasible_X'] = results_dataframes['case_df_op_feasible'].drop(['case_id', 'Stability','cell_name'], axis=1)

# %% ---- SELECT ONLY UNFEASIBLE CASES ----

results_dataframes['case_df_op_unfeasible'] = results_dataframes[df_op].query(
    'Stability < 0')

# %%
columns_in_df = dict()
for key, item in results_dataframes.items():
    print(key)
    columns_in_df[key] = results_dataframes[key].columns

# %% ----  Remove columns with only 1 value ----
columns_with_single_values = []
for c in columns_in_df['case_df_op_feasible']:
    if results_dataframes['case_df_op_feasible'][c].unique().size == 1:
        columns_with_single_values.append(c)
# --> if there is something different from Sn_SGX check, otherwise it is normal (no changes in SG installed power)
print(columns_with_single_values)

results_dataframes['case_df_op_feasible'] = results_dataframes['case_df_op_feasible'].drop(
    columns_with_single_values, axis=1)
results_dataframes['case_df_op_feasible_X'] = results_dataframes['case_df_op_feasible_X'].drop(
    columns_with_single_values, axis=1)

# %% ----  Check if there are extra taus ----

df_taus = results_dataframes['case_df_op_feasible'][['case_id']].merge(results_dataframes['cases_df_feasible'][[
                                                                       col for col in columns_in_df['cases_df_feasible'] if col.startswith('tau_droop')]+['case_id']], on='case_id', how='left').drop(['case_id'], axis=1)

df_Sn_GFOL = results_dataframes['case_df_op_feasible'][[col for col in columns_in_df['case_df_op_feasible'] if col.startswith('Sn_GFOL')]].reset_index(drop=True)
df_taus_GFOL_droopf = df_taus[['tau_droop_f_gfol_'+col.split('GFOL')[1] for col in df_Sn_GFOL.columns]]
df_taus_GFOL_droopu = df_taus[['tau_droop_u_gfol_'+col.split('GFOL')[1] for col in df_Sn_GFOL.columns]]
df_taus_GFOL_droopf[np.array(df_Sn_GFOL==0)]=0
df_taus_GFOL_droopu[np.array(df_Sn_GFOL==0)]=0

df_Sn_GFOR = results_dataframes['case_df_op_feasible'][[col for col in columns_in_df['case_df_op_feasible'] if col.startswith('Sn_GFOR')]].reset_index(drop=True)
df_taus_GFOR_droopf = df_taus[['tau_droop_f_gfor_'+col.split('GFOR')[1] for col in df_Sn_GFOR.columns]]
df_taus_GFOR_droopu = df_taus[['tau_droop_u_gfor_'+col.split('GFOR')[1] for col in df_Sn_GFOR.columns]]
df_taus_GFOR_droopf[np.array(df_Sn_GFOR==0)]=0
df_taus_GFOR_droopu[np.array(df_Sn_GFOR==0)]=0

df_taus_fixed = pd.concat([df_taus_GFOL_droopf,df_taus_GFOL_droopu,df_taus_GFOR_droopf,df_taus_GFOR_droopu],axis=1)

#%%
df_taus_fixed['Stability'] = results_dataframes['case_df_op_feasible']['Stability'].reset_index(drop=True)
df_Sn_GFOL['Stability'] = results_dataframes['case_df_op_feasible']['Stability'].reset_index(drop=True)
df_Sn_GFOR['Stability'] = results_dataframes['case_df_op_feasible']['Stability'].reset_index(drop=True)

#%%
theta_rad_abs = np.abs(results_dataframes['case_df_op_feasible'][theta_cols]*np.pi/180)
df_slack = pd.DataFrame(columns =['slack_bus','slack_theta'], index = theta_rad_abs.index)
for ii in theta_rad_abs.index:
    df_slack.loc[ii, 'slack_bus'] = theta_rad_abs.loc[ii].index[theta_rad_abs.loc[ii].argmin()]
    df_slack.loc[ii, 'slack_theta'] = theta_rad_abs.loc[ii].min()
    
results_dataframes['case_df_op_feasible']['slack_bus']= df_slack['slack_bus']

slack_case=dict()
for sl_bus in df_slack['slack_bus'].unique():
    slack_case[sl_bus] = list(results_dataframes['case_df_op_feasible'].query('slack_bus == @sl_bus')['case_id'])

#%%
theta_rad = results_dataframes['case_df_op_feasible'][theta_cols]*np.pi/180
theta_rad_slack_26 = copy.copy(theta_rad)
df_slack['delta_slack'] = 0

print(df_slack.groupby('slack_bus').count())

for ii in df_slack.query('slack_bus != "theta26"').index:
    slack_bus = df_slack.loc[ii,'slack_bus']
    delta_slack = theta_rad.loc[ii,'theta26'] - theta_rad.loc[ii,slack_bus]
    df_slack.loc[ii,'delta_slack'] = delta_slack
    theta_rad_slack_26.loc[ii,theta_cols] = theta_rad.loc[ii,theta_cols] - delta_slack
    
#%%
results_dataframes['raw_data']=results_dataframes['case_df_op_feasible'].drop(theta_cols, axis=1).reset_index(drop=True)
results_dataframes['raw_data'] = pd.concat([results_dataframes['raw_data'],df_taus_fixed.drop('Stability',axis=1),
                                            theta_rad_slack_26.reset_index(drop=True)],axis=1)

#%%
model = XGBClassifier(n_estimators=350)
estimator = Pipeline([('scaler', RobustScaler()),('xgb',XGBClassifier(n_estimators=350))])
df = results_dataframes['raw_data']
 
X = df.drop(['case_id','Stability','cell_name','slack_bus'],axis=1).reset_index(drop=True)
Y = df[['Stability']].reset_index(drop=True).values.astype(int).ravel()

X_train, X_test, y_train, y_test = train_test_split(X, Y , train_size=0.8, shuffle=True, random_state=42)
# w_neg, w_pos = 10.0, 1.0  # tune
# sw = np.where(y_train == 0, w_neg, w_pos)

#model=estimator.fit(X_train, y_train)
model.fit(X_train, y_train)
#model =DecisionTreeClassifier().fit(X_train,y_train)
# proba = model.predict_proba(X_test)[:,1]

y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')

descr= results_dataframes['raw_data'].describe()

#%%
#estimator = Pipeline([('scaler', RobustScaler()), ('xgb', XGBClassifier())])
model = XGBClassifier()
param_grid = {'learning_rate':[0.1,0.2,0.25, 0.3],#, 0.35], #np.arange(0.1,0.7,0.2),
              'max_depth':[4,3,5,6,7],#7,8,9],#[5,6,7],
              'subsample':[0.5,0.8,1],
              'n_estimators':[300,350]
    }

best_params_grid=dict()
scores_depth = dict()
best_model, best_params, means, stds, params = GSkFCV(param_grid, X_train, y_train, model, 'accuracy', n_folds=3)


best_params_grid[type_corr_analysis+dataset_ID]={'learning_rate': best_params['xgb__eta'],
          'max_depth': best_params['xgb__max_depth'],
          'subsample': best_params['xgb__subsample'], 'n_estimators': best_params['xgb__n_estimators']}


scores_depth[type_corr_analysis+dataset_ID] = kfold_cv_depth(df_dict, type_corr_analysis, dataset_ID, cases_id_depth_feas, plot_depth_exploration=False, n_fold=5, params=best_params_grid[type_corr_analysis+dataset_ID])

#%%

def boxplot_stability(df, stability,columns, ax=None):
    if ax == None:
        fig =  plt.figure()
    ax = df.query('Stability == @stability')[[col for col in df.columns if col.startswith(columns)]].boxplot(figsize=(10, 6))
    ax.set_title(columns+ [' Unstable Cases' if stability==0 else ' Stable Cases'][0])
    ax.set_xticklabels([col.split('_')[-1] for col in df.columns if col.startswith(columns)], rotation=45)
    return ax
#%%
ax = boxplot_stability(df_Sn_GFOL, 0,'Sn_GFOL')
boxplot_stability(df_Sn_GFOL, 1,'Sn_GFOL', ax)
    
#%%
fig =  plt.figure()
ax = df_taus_fixed.query('Stability == 0')[[col for col in df_taus_fixed.columns if col.startswith('tau_droop_f_gfor')]].boxplot(figsize=(10, 6))
ax.set_title('tau_droop_f_gfor Unstable Cases')
ax.set_xticklabels([col.split('_')[-1] for col in df_taus_fixed.columns if col.startswith('tau_droop_f_gfor')])
    
fig =  plt.figure()
ax = df_taus_fixed.query('Stability == 1')[[col for col in df_taus_fixed.columns if col.startswith('tau_droop_f_gfor')]].boxplot(figsize=(15, 6))
ax.set_title('tau_droop_f_gfor Stable Cases')
ax.set_xticklabels([col.split('_')[-1] for col in df_taus_fixed.columns if col.startswith('tau_droop_f_gfor')])

fig =  plt.figure()
ax = df_taus_fixed.query('Stability == 0')[[col for col in df_taus_fixed.columns if col.startswith('tau_droop_f_gfol')]].boxplot(figsize=(10, 6))
ax.set_title('tau_droop_f_gfol Unstable Cases')
ax.set_xticklabels([col.split('_')[-1] for col in df_taus_fixed.columns if col.startswith('tau_droop_f_gfol')])

fig =  plt.figure()
ax = df_taus_fixed.query('Stability == 1')[[col for col in df_taus_fixed.columns if col.startswith('tau_droop_f_gfol')]].boxplot(figsize=(15, 6))
ax.set_title('tau_droop_f_gfol Stable Cases')
ax.set_xticklabels([col.split('_')[-1] for col in df_taus_fixed.columns if col.startswith('tau_droop_f_gfol')])

fig =  plt.figure()
ax = df_taus_fixed.query('Stability == 0')[[col for col in df_taus_fixed.columns if col.startswith('tau_droop_u_gfol')]].boxplot(figsize=(10, 6))
ax.set_title('tau_droop_u_gfol Unstable Cases')
ax.set_xticklabels([col.split('_')[-1] for col in df_taus_fixed.columns if col.startswith('tau_droop_u_gfol')])

fig =  plt.figure()
ax = df_taus_fixed.query('Stability == 1')[[col for col in df_taus_fixed.columns if col.startswith('tau_droop_u_gfol')]].boxplot(figsize=(15, 6))
ax.set_title('tau_droop_u_gfol Stable Cases')
ax.set_xticklabels([col.split('_')[-1] for col in df_taus_fixed.columns if col.startswith('tau_droop_u_gfol')])


fig =  plt.figure()
ax = df_taus_fixed.query('Stability == 1 and tau_droop_u_gfol_32 !=0')[['tau_droop_u_gfol_32']].boxplot(figsize=(15, 6))

# %% ---- Check correlated variables Option #1 ----
def get_correlated_columns(df, c_threshold=0.95, method='pearson'):
    uncorrelated = []
    correlated_features_tuples = []
    correlated_features = pd.DataFrame(columns=['Feat1', 'Feat2', 'Corr'])
    correlation = df.corr(method=method)
    count = 0

    for i in correlation.index:
        corr_found = False
        for j in correlation:
            if i != j and abs(correlation.loc[i, j]) >= c_threshold:
                # if tuple([j,i]) not in correlated_features_tuples:
                correlated_features_tuples.append(tuple([i, j]))
                correlated_features.loc[count, 'Feat1'] = i
                correlated_features.loc[count, 'Feat2'] = j
                correlated_features.loc[count, 'Corr'] = correlation.loc[i, j]
                count = count+1
                
                corr_found = True
        if corr_found == False:
            uncorrelated.append(i)
    return correlated_features, uncorrelated

#%%
correlated_features, uncorrelated_features = get_correlated_columns(results_dataframes['raw_data'].drop(['Stability', 'cell_name','case_id','slack_bus'],axis=1))
    #results_dataframes['case_df_op_feasible_X'])

grouped_corr_feat = correlated_features.groupby('Feat1').count().sort_values(by='Feat2',ascending=False).reset_index()

#%%
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

#%%
grouped_corr_feat = correlated_features.groupby('Feat1').count().sort_values(by='Feat2',ascending=False).reset_index()

#%%
keep_var_2=[]
correlated_features_copy= copy.copy(correlated_features)
while not grouped_corr_feat.empty:
    # Pick the first remaining Feat1
    var = grouped_corr_feat.iloc[0]['Feat1']
    keep_var_2.append(var)

    # Find all features correlated with this one
    to_remove = correlated_features.query('Feat1 == @var')['Feat2'].tolist()

    # Drop all rows where Feat1 is in to_remove
    correlated_features_copy = correlated_features_copy[~correlated_features_copy['Feat1'].isin(to_remove)]
    correlated_features_copy = correlated_features_copy[~correlated_features_copy['Feat1'].isin(keep_var)]
    
    correlated_features_copy = correlated_features_copy[~correlated_features_copy['Feat2'].isin(to_remove)]
    correlated_features_copy = correlated_features_copy[~correlated_features_copy['Feat2'].isin(keep_var)]

    grouped_corr_feat = correlated_features_copy.groupby('Feat1').count().sort_values(by='Feat2',ascending=False).reset_index()

#%%[
results_dataframes['keep_var1'] = results_dataframes['raw_data'][uncorrelated_features + keep_var + ['Stability']]
results_dataframes['keep_var2'] = results_dataframes['raw_data'][uncorrelated_features + keep_var_2 + ['Stability']]

model = XGBClassifier()
#estimator = Pipeline([('scaler', RobustScaler()),('xgb',XGBClassifier())])
df = results_dataframes['keep_var2']
 
X = df.drop(['Stability'],axis=1).reset_index(drop=True)
Y = df[['Stability']].reset_index(drop=True).values.astype(int).ravel()

X_train, X_test, y_train, y_test = train_test_split(X, Y , train_size=0.8, shuffle=True, random_state=42)
# w_neg, w_pos = 10.0, 1.0  # tune
# sw = np.where(y_train == 0, w_neg, w_pos)

#model=estimator.fit(X_train, y_train)
model.fit(X_train, y_train)
#model =DecisionTreeClassifier().fit(X_train,y_train)
# proba = model.predict_proba(X_test)[:,1]

y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')

#%%

#%%

from scipy.stats import pointbiserialr

# Assume df is your DataFrame and 'target' is binary (0/1)
target = 'Stability'
numeric_cols = results_dataframes['raw_data'].drop(['cell_name','case_id','slack_bus'],axis=1).columns.drop(target)

corrs = []
for col in numeric_cols:
    corr, p_value = pointbiserialr(df[target], df[col])
    corrs.append((col, corr, p_value))

corr_df = pd.DataFrame(corrs, columns=['feature', 'correlation', 'p_value'])
corr_df = corr_df.sort_values(by='correlation', key=abs, ascending=False)
print(corr_df.head(10))


#%%
results_dataframes['case_df_op_feasible_uncorr_X'] = pd.concat([results_dataframes['case_df_op_feasible_X'][keep_var].reset_index(drop=True), df_taus_fixed],axis=1)
results_dataframes['case_df_op_feasible_uncorr'] = results_dataframes['case_df_op_feasible_uncorr_X']
results_dataframes['case_df_op_feasible_uncorr']['case_id'] = results_dataframes['case_df_op_feasible']['case_id'].reset_index(drop=True)
results_dataframes['case_df_op_feasible_uncorr']['Stability'] = results_dataframes['case_df_op_feasible']['Stability'].reset_index(drop=True)

results_dataframes['case_df_op_feasible_uncorr'].to_csv(path+dir_name+'DataSet_training_uncorr_var'+dataset_ID.replace('ivity','Sensitivity')+'.csv')

# %% ---- Check correlated variables Option #2 ----

results = pd.concat([results_dataframes['case_df_op_feasible_X'].reset_index(drop=True), df_taus_fixed.reset_index(drop=True)], axis=1).apply(
#results = results_dataframes['case_df_op_feasible_X'].reset_index(drop=True).apply(
     lambda col: pointbiserialr(col, results_dataframes['case_df_op_feasible']['Stability']), result_type='expand').T
results.columns = ['correlation', 'p_value']
results['abs_corr'] = abs(results['correlation'])

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
X = pd.concat([results_dataframes['case_df_op_feasible_X'].reset_index(
    drop=True), df_taus_fixed.reset_index(drop=True)], axis=1)
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

# results_dataframes['case_df_op_feasible_uncorr_X'] = results_dataframes['case_df_op_feasible_X'][keep_var]
# results_dataframes['case_df_op_feasible_uncorr'] = results_dataframes['case_df_op_feasible'][keep_var+['case_id', 'Stability']]

results_dataframes['case_df_op_feasible_uncorr_HierCl_X'] = X[keep_var]
results_dataframes['case_df_op_feasible_uncorr_HierCl'] = pd.concat([X[keep_var], results_dataframes['case_df_op_feasible'][['case_id', 'Stability']].reset_index(drop=True)],axis=1)

results_dataframes['case_df_op_feasible_uncorr_HierCl'].to_csv(path+dir_name+'DataSet_training_uncorr_var_HierCl'+dataset_ID.replace('ivity','Sensitivity')+'.csv')
