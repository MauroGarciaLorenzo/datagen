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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import copy
import json

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
dir_name=[dir_name for dir_name in os.listdir(path) if '_2732' in dir_name and 'zip' not in dir_name][0]# if dir_name.startswith('datagen') and 'zip' not in dir_name]#
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
results_dataframes['raw_data_pu']= results_dataframes['raw_data'].copy(deep=True)

results_dataframes['raw_data_pu'][P_SG_cols]=P_SG_pu.drop('Stability',axis=1)
results_dataframes['raw_data_pu'][Q_SG_cols]=Q_SG_pu.drop('Stability',axis=1)
results_dataframes['raw_data_pu'][P_GFOL_cols]=P_GFOL_pu.drop('Stability',axis=1)
results_dataframes['raw_data_pu'][Q_GFOL_cols]=Q_GFOL_pu.drop('Stability',axis=1)
results_dataframes['raw_data_pu'][P_GFOR_cols]=P_GFOR_pu.drop('Stability',axis=1)
results_dataframes['raw_data_pu'][Q_GFOR_cols]=Q_GFOR_pu.drop('Stability',axis=1)

#%%

with open(path+dir_name+'/PFI_features.txt', 'r') as f:
    PFI_feat = [line.strip() for line in f]

#%%

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

def GSkFCV(param_grid, X_train, Y_train, estimator, scorer, n_folds = 5):
    '''
    REQUIRES: param_grid, X_train, Y_train, PFI_features, estimator, scorer
    '''
    
    seed = 23
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=kfold, scoring=scorer, verbose=1)
    grid_search.fit(X_train, Y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    params = grid_search.cv_results_['params']
    for mean, stdev, param in sorted(zip(means, stds, params), key=lambda x: x[0], reverse=True)[:5]:
        print("%f (%f) with: %r" % (mean, stdev, param))

    return best_model, best_params, means, stds, params
#%%
estimator = Pipeline([('scaler', RobustScaler()), ('xgb', XGBClassifier())])

param_grid = {'xgb__eta':[0.1,0.25, 0.5,0.8], #, 0.35], #np.arange(0.1,0.7,0.2),
              'xgb__max_depth':[5,7,9],#[5,6,7],
              'xgb__subsample':[0.5,0.8,1],
              'xgb__n_estimators':[300,600,1000,1200]
    }

from skopt import BayesSearchCV

X= results_dataframes['raw_data_pu'][PFI_feat].reset_index(drop=True)
Y = results_dataframes['raw_data_pu'][['Stability']].reset_index(drop=True).values.astype(int).ravel()


X_train, X_test, y_train, y_test = train_test_split(X, Y , train_size=0.8, shuffle=True, random_state=42)


gb_bs = BayesSearchCV(estimator, param_grid, cv=5, n_iter=50, n_jobs=-1, random_state=42)
gb_bs.fit(X_train, y_train)


# In[27]:


print(classification_report(y_test, gb_bs.predict(X_test), target_names=['Unstable','Stable']))


# In[28]:

fig = plt.figure(figsize=(4,4))
ConfusionMatrixDisplay.from_estimator(gb_bs, X_test, y_test, display_labels=['Unstable','Stable'], cmap='Blues');
plt.title("Confusion Matrix Test \#2")
plt.show()
plt.tight_layout()

#%%
best_params = gb_bs.best_params_

report = classification_report(y_test, gb_bs.predict(X_test), target_names=['Unstable','Stable'])
report = report+'\n accuracy = '+str(max(gb_bs.cv_results_['mean_test_score']))+'+o-'+ str(gb_bs.cv_results_['std_test_score'][np.argmax(gb_bs.cv_results_['mean_test_score'])])

with open(path+dir_name+'/best_params.txt', 'w') as f:
    json.dump(best_params, f)

with open(path+dir_name+'/report.txt', 'w') as f:
    f.write(report)


#%%
def kfold_cv_depth(df,PFI_feat, cases_id_depth, plot_depth_exploration=False, dimensions_caseid_feasible = None, n_fold=5, params=None, score='accuracy'):
    df_training = pd.DataFrame(columns= df.columns)
    cases_id_training = []
    scores_df=pd.DataFrame(columns=['Depth','score_mean','score_std','n_training_cases','perc_stable'])
    cv = KFold(n_splits=n_fold, shuffle=True, random_state=23)

    if plot_depth_exploration:
        ax = plot_mesh(mesh_df)
    
    for depth in range(0,int(max(cases_id_depth['Depth']))+1):
        add_case_id = list(set(list(cases_id_depth.query('Depth == @depth')['case_id']))&set(list(df['case_id'])))
        cases_id_training.extend(add_case_id)
        df_training = df.query('case_id == @cases_id_training')
        scores_df.loc[depth,'n_training_cases']=len(df_training)
        scores_df.loc[depth,'perc_stable']=len(df_training.query('Stability == 1'))/len(df_training)
    
        if len(df_training)>= n_fold:
            #clf = svm.SVC(kernel='linear', C=1, random_state=42)
            #clf = MLPClassifier(random_state=1, max_iter=5000, activation='relu')
            if params == None:
                clf = Pipeline([('scaler', RobustScaler()), ('xgb', XGBClassifier())])
            else:
                clf = Pipeline([('scaler', RobustScaler()), ('xgb', XGBClassifier(**params))])
            X = df_training[PFI_feat].reset_index(drop=True)
            y = df_training[['Stability']].reset_index(drop=True).values.astype(int).ravel()
            scores = cross_val_score(clf, X, y, cv=cv, scoring=score)
            
            scores_df.loc[depth,'Depth']=depth
            scores_df.loc[depth,'score_mean']=scores.mean()
            scores_df.loc[depth,'score_std']=scores.std()

            if plot_depth_exploration:
                ax.scatter(dimensions_caseid_feasible.query('case_id == @add_case_id')['p_cig'],
                       dimensions_caseid_feasible.query('case_id == @add_case_id')['p_sg'], label = 'Depth '+str(depth))
    plt.legend()
    return scores_df

#%%

xgb_params = {k.replace('xgb__', ''): v for k, v in best_params.items() if k.startswith('xgb__')}

cases_id_depth = pd.read_excel(path+dir_name+'/cases_id_depth'+dataset_ID+'.xlsx')[['Depth','case_id','CellName']]

scores_depth = kfold_cv_depth(results_dataframes['raw_data_pu'], PFI_feat, cases_id_depth, plot_depth_exploration=False, 
                              n_fold=5, params=xgb_params)

#%%
pd.DataFrame.to_excel(scores_depth, path+dir_name+'/scores_depth_PFI_xgb.xlsx')#_var_HierCl_