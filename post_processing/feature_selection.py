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

#%%
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
    
# %% ---- FILL NAN VALUES WITH NULL ---

results_dataframes[df_op] = results_dataframes[df_op].fillna(0)

# %% ---- FIX VALUES ----

Sn_cols = [col for col in results_dataframes[df_op]
           if col.startswith('Sn')]
results_dataframes[df_op][Sn_cols] = results_dataframes[df_op][Sn_cols]/100

theta_cols = [col for col in results_dataframes[df_op]
              if col.startswith('theta')]
# # # Adjust angles greater than 180°
# results_dataframes[df_op][theta_cols] = results_dataframes[df_op][theta_cols] - \
#     (results_dataframes[df_op][theta_cols] > 180) * 360

def wrap_angle(angle):
    return (angle + 180) % 360 - 180

results_dataframes[df_op][theta_cols] = wrap_angle( results_dataframes[df_op][theta_cols])


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

#%%
save_single_values = results_dataframes['case_df_op_feasible'].iloc[0][columns_with_single_values]

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
import copy
theta_rad = results_dataframes['case_df_op_feasible'][theta_cols]*np.pi/180
theta_rad_slack_26 = copy.copy(theta_rad)
df_slack['delta_slack'] = 0.0   # float from the start

print(df_slack.groupby('slack_bus').count())
#%%
for ii in df_slack.query('slack_bus != "theta26"').index:
    slack_bus = df_slack.loc[ii,'slack_bus']
    delta_slack = theta_rad.loc[ii,'theta26'] - theta_rad.loc[ii,slack_bus]
    df_slack.loc[ii,'delta_slack'] = delta_slack
    theta_rad_slack_26.loc[ii,theta_cols] = theta_rad.loc[ii,theta_cols] - delta_slack
    
#%%
results_dataframes['raw_data']=results_dataframes['case_df_op_feasible'].drop(theta_cols, axis=1).reset_index(drop=True)
results_dataframes['raw_data'] = pd.concat([results_dataframes['raw_data'],df_taus_fixed.drop('Stability',axis=1),
                                            theta_rad_slack_26.reset_index(drop=True).drop('theta26',axis=1)],axis=1)

#%%
model = XGBClassifier(n_estimators=600)
#estimator = Pipeline([('scaler', StandardScaler()),('xgb',XGBClassifier(n_estimators=350))])
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
plt.tight_layout()
#%% 
# Per unit of SG P and Q

P_SG_pu, skewness_p_sg_pu, kurt_p_sg_pu = calculate_pu_skewness_kurt(results_dataframes['raw_data'], P_SG_cols, save_single_values)
Q_SG_pu, skewness_q_sg_pu, kurt_q_sg_pu = calculate_pu_skewness_kurt(results_dataframes['raw_data'], Q_SG_cols, save_single_values)

#%%
# per unit of GFOL and GFOR P and Q
df_Sn_CIG = pd.DataFrame(columns=[col.replace('GFOL','CIG') for col in df_Sn_GFOL.columns if col !='Stability'])
                         
for col in df_Sn_CIG.columns:
    if col !='Stability':
        df_Sn_CIG[col]=df_Sn_GFOL[col.replace('CIG','GFOL')]+df_Sn_GFOR[col.replace('CIG','GFOR')]
    # else:
    #     df_Sn_CIG[col] =df_Sn_GFOL[col]
        
Sn_CIG_values = df_Sn_CIG.iloc[0]

P_GFOL_pu, skewness_p_gfol_pu, kurt_p_gfol_pu = calculate_pu_skewness_kurt(results_dataframes['raw_data'],P_GFOL_cols, Sn_CIG_values)
P_GFOR_pu, skewness_p_gfor_pu, kurt_p_gfor_pu = calculate_pu_skewness_kurt(results_dataframes['raw_data'],P_GFOR_cols, Sn_CIG_values)
Q_GFOL_pu, skewness_q_gfol_pu, kurt_q_gfol_pu = calculate_pu_skewness_kurt(results_dataframes['raw_data'],Q_GFOL_cols, Sn_CIG_values)
Q_GFOR_pu, skewness_q_gfor_pu, kurt_q_gfor_pu = calculate_pu_skewness_kurt(results_dataframes['raw_data'],Q_GFOR_cols, Sn_CIG_values)

#%%
theta_rad_slack_26_stab = theta_rad_slack_26.reset_index(drop=True)
theta_rad_slack_26_stab['Stability'] = results_dataframes['raw_data']['Stability']
skewness_theta = calculate_skewness(theta_rad_slack_26_stab).drop('theta26',axis=0)
kurt_theta = calculate_kurt(theta_rad_slack_26_stab).drop('theta26',axis=0)

#%%

plot_distribution_with_changes(skewness_p_gfol_pu, P_GFOL_pu)
plot_distribution_with_changes(skewness_q_gfol_pu, Q_GFOL_pu)
plot_distribution_with_changes(skewness_p_gfor_pu, P_GFOR_pu)
plot_distribution_with_changes(skewness_q_gfor_pu, Q_GFOR_pu)
plot_distribution_with_changes(skewness_p_sg_pu, P_SG_pu)
plot_distribution_with_changes(skewness_q_sg_pu, Q_SG_pu)
plot_distribution_with_changes(skewness_theta, theta_rad_slack_26_stab)

plot_distribution_with_changes(kurt_p_gfol_pu, P_GFOL_pu)
plot_distribution_with_changes(kurt_q_gfol_pu, Q_GFOL_pu)
plot_distribution_with_changes(kurt_p_gfor_pu, P_GFOR_pu)
plot_distribution_with_changes(kurt_q_gfor_pu, Q_GFOR_pu)
plot_distribution_with_changes(kurt_p_sg_pu, P_SG_pu)
plot_distribution_with_changes(kurt_q_sg_pu, Q_SG_pu)
plot_distribution_with_changes(kurt_theta, theta_rad_slack_26_stab)

#%%
results_dataframes['raw_data_pu']= results_dataframes['raw_data'].copy(deep=True)

results_dataframes['raw_data_pu'][P_SG_cols]=P_SG_pu.drop('Stability',axis=1)
results_dataframes['raw_data_pu'][Q_SG_cols]=Q_SG_pu.drop('Stability',axis=1)
results_dataframes['raw_data_pu'][P_GFOL_cols]=P_GFOL_pu.drop('Stability',axis=1)
results_dataframes['raw_data_pu'][Q_GFOL_cols]=Q_GFOL_pu.drop('Stability',axis=1)
results_dataframes['raw_data_pu'][P_GFOR_cols]=P_GFOR_pu.drop('Stability',axis=1)
results_dataframes['raw_data_pu'][Q_GFOR_cols]=Q_GFOR_pu.drop('Stability',axis=1)

#%%
model = XGBClassifier(n_estimators=600)
#estimator = Pipeline([('scaler', StandardScaler()),('xgb',XGBClassifier(n_estimators=350))])
df = results_dataframes['raw_data_pu']
 
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
plt.tight_layout()
#%%

# baseline_model = XGBClassifier(
#     n_estimators=1000,
#     max_depth=5,
#     learning_rate=0.1,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     objective="multi:softprob" if len(np.unique(Y)) > 2 else "binary:logistic",
#     eval_metric="mlogloss" if len(np.unique(Y)) > 2 else "logloss",
#     tree_method="hist"  # or "gpu_hist" if you have GPU
# )

# baseline_model.fit(X_train, y_train)
# y_pred_base = baseline_model.predict(X_test)
# print("Baseline XGB accuracy:", accuracy_score(y_test, y_pred_base))

# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot(cmap='Blues')


#%%
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

from scipy.stats import pointbiserialr

def point_biserial_correlations(X, y):
    results = {}
    for col in X.columns:
        r, p = pointbiserialr(X[col], y)
        results[col] = {"correlation": r, "p_value": p}
    return pd.DataFrame(results).T


from collections import defaultdict
import networkx as nx

def select_uncorrelated_features(correlated_features, corr_with_stab, threshold=0.95):
    # Filter for strong correlations only
    strong_corrs = correlated_features[correlated_features['Corr_abs'] >= threshold]
    
    # Build undirected graph of correlated features
    G = nx.Graph()
    for _, row in strong_corrs.iterrows():
        G.add_edge(row['Feat1'], row['Feat2'])

    # Map feature to target correlation (abs)
    target_corr_map = corr_with_stab['correlation_abs'].to_dict()

    features_to_keep = set()
    features_to_remove = set()

    # Process each connected component (group of correlated features)
    for component in nx.connected_components(G):
        component = list(component)
        print(component)
        
        # Rank by correlation with target (descending), default 0 if missing
        component_sorted = sorted(
            component, 
            key=lambda x: target_corr_map.get(x, 0), 
            reverse=True
        )
        # Keep the best one
        features_to_keep.add(component_sorted[0])
        # Remove the rest
        features_to_remove.update(component_sorted[1:])

    # Add uncorrelated features that aren't in the graph
    all_features = set(corr_with_stab.index)
    unconnected_features = all_features - set(G.nodes)
    features_to_keep.update(unconnected_features)

    return sorted(features_to_keep), sorted(features_to_remove)
#%%
correlated_features, uncorrelated_features = get_correlated_columns(X_train)
correlated_features['Corr_abs']= abs(correlated_features['Corr'])
correlated_features = correlated_features.sort_values(by='Corr_abs', ascending=False).reset_index(drop=True)

#%%
corr_with_stab = point_biserial_correlations(X_train, y_train)
corr_with_stab['correlation_abs']=abs(corr_with_stab['correlation'])
corr_with_stab=corr_with_stab.sort_values("correlation_abs", ascending=False)

#%%

features_to_keep, features_to_remove = select_uncorrelated_features(correlated_features, corr_with_stab)

#%%
# baseline_model = XGBClassifier(
#     n_estimators=1000,
#     max_depth=5,
#     learning_rate=0.1,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     objective="multi:softprob" if len(np.unique(Y)) > 2 else "binary:logistic",
#     eval_metric="mlogloss" if len(np.unique(Y)) > 2 else "logloss",
#     tree_method="hist"  # or "gpu_hist" if you have GPU
# )
model = XGBClassifier(n_estimators=600)

model.fit(X_train[features_to_keep], y_train)
y_pred_base = model.predict(X_test[features_to_keep])
print("Baseline XGB accuracy:", accuracy_score(y_test, y_pred_base))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.tight_layout()

#%%
estimator = Pipeline([('scaler', RobustScaler()),('xgb',XGBClassifier(n_estimators=600))])

estimator.fit(X_train[features_to_keep], y_train)
#model =DecisionTreeClassifier().fit(X_train,y_train)
# proba = model.predict_proba(X_test)[:,1]

y_pred = estimator.predict(X_test[features_to_keep])
score = accuracy_score(y_test, y_pred)
print(score)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')

#%%
from sklearn.inspection import permutation_importance #PFI
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score

def PFI_fun(estimator, X_train, Y_train, X_test, Y_test, scorer):
    '''
    REQUIRES: estimator, X_train, Y_train, X_test, Y_test, scorer
    '''

    considered_features = X_train.columns.to_list()
    
    estimator.fit(X_train[considered_features], Y_train)
    
    r = permutation_importance(estimator, X_test[considered_features], Y_test, 
                               n_repeats=30, random_state=0, scoring=scorer)
    PFI_features=[]
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            PFI_features.append(X_train.columns[i])#estimator.feature_names_in_[i])
            # print(f"{estimator.feature_names_in_[i]:<8}"
            #     f"\t {r.importances_mean[i]:.3f}"
            #     f" ({r.importances_std[i]:.3f})")
    
    
    return PFI_features

estimator = Pipeline([('scaler', RobustScaler()),('xgb',XGBClassifier(n_estimators=600))])

PFI_feat=PFI_fun(estimator, X_train[features_to_keep],y_train,X_test,y_test,make_scorer(accuracy_score))


#%%
estimator = Pipeline([('scaler', RobustScaler()),('xgb',XGBClassifier(n_estimators=600))])

estimator.fit(X_train[PFI_feat], y_train)
#model =DecisionTreeClassifier().fit(X_train,y_train)
# proba = model.predict_proba(X_test)[:,1]

y_pred = estimator.predict(X_test[PFI_feat])
score = accuracy_score(y_test, y_pred)
print(score)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.tight_layout()

#%%
with open(path+dir_name+'/uncorrelated_features.txt', 'w') as f:
    for line in features_to_keep:
        f.write(line + '\n')

with open(path+dir_name+'/PFI_features.txt', 'w') as f:
    for line in PFI_feat:
        f.write(line + '\n')

#%%


from skfeature.function.information_theoretical_based import MRMR
idx = MRMR.mrmr(np.array(X_train), y_train, n_selected_features=20)

#model=estimator.fit(X_train, y_train)
model.fit(X_train[X_train.columns[idx[0]]], y_train)
#model =DecisionTreeClassifier().fit(X_train,y_train)
# proba = model.predict_proba(X_test)[:,1]

y_pred = model.predict(X_test[X_train.columns[idx[0]]])
score = accuracy_score(y_test, y_pred)
print(score)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')

#%%
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.covariance import OAS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


clf1 = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=None).fit(X_train, y_train)
clf2 = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X_train, y_train)
oa = OAS(store_precision=False, assume_centered=False)
clf3 = LinearDiscriminantAnalysis(solver="lsqr", covariance_estimator=oa).fit(X_train, y_train)

score_clf1 = clf1.score(X_train,y_train)
score_clf2 = clf2.score(X_train,y_train)
score_clf3 = clf3.score(X_train,y_train)

#%%

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)


lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
qda = QuadraticDiscriminantAnalysis(store_covariance=True)

lda.fit(X_train, y_train)
lda.score(X_train, y_train)

plot_result(lda, X, y, ax_row[0])
qda.fit(X, y)
plot_result(qda, X_train, y_train, ax_row[1])