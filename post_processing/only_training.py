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

from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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
dir_name=[dir_name for dir_name in os.listdir(path) if '_2862' in dir_name and 'zip' not in dir_name][0]#_2862# if dir_name.startswith('datagen') and 'zip' not in dir_name]#
print(dir_name)
#'20251030_sensitivity'
#%%

dataset_ID = dir_name[-5:].replace('ivity','Sensitivity')

#%%
df_dict=dict()
df_dict['DataSet_training_uncorr_var'+dataset_ID] = pd.read_csv(path+dir_name+'/DataSet_training_uncorr_var'+dataset_ID+'.csv').drop('Unnamed: 0', axis=1).drop_duplicates(keep='first')
df_dict['DataSet_training_uncorr_var_HierCl'+dataset_ID] = pd.read_csv(path+dir_name+'/DataSet_training_uncorr_var_HierCl'+dataset_ID+'.csv').drop('Unnamed: 0', axis=1).drop_duplicates(keep='first')

case_id_feasible = list(df_dict['DataSet_training_uncorr_var'+dataset_ID]['case_id'])
cases_id_depth = pd.read_excel(path+dir_name+'/cases_id_depth'+dataset_ID+'.xlsx')[['Depth','case_id','CellName']]

cases_id_depth_feas = cases_id_depth.query('case_id == @case_id_feasible')

#%%
#for dir_name in dir_names:
path_results = os.path.join(path, dir_name)
df_op='df_op'#'case_df_op'
results_dataframes, csv_files = open_csv(
    path_results, ['cases_df.csv', df_op+'.csv'])

perc_stability(results_dataframes[df_op], dir_name)


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

# theta_cols = [col for col in results_dataframes[df_op]
#               if col.startswith('theta')]
# # Adjust angles greater than 180°
# results_dataframes[df_op][theta_cols] = results_dataframes[df_op][theta_cols] - \
#     (results_dataframes[df_op][theta_cols] > 180) * 360

# results_dataframes['case_df_op'][theta_cols] = results_dataframes['case_df_op'][theta_cols] * np.pi/180

# add total demand variables
PL_cols = [
    col for col in results_dataframes[df_op].columns if col.startswith('PL')]
results_dataframes[df_op]['PD'] = results_dataframes[df_op][PL_cols].sum(
    axis=1)

QL_cols = [
    col for col in results_dataframes[df_op].columns if col.startswith('QL')]
results_dataframes[df_op]['QD'] = results_dataframes[df_op][QL_cols].sum(
    axis=1)

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

results_dataframes['raw_data']=results_dataframes['case_df_op_feasible'].reset_index(drop=True)
results_dataframes['raw_data'] = pd.concat([results_dataframes['raw_data'],df_taus_fixed.drop('Stability',axis=1)],axis=1)

case_id_slack_26 = slack_case['theta26']
results_dataframes['raw_data_slack_26'] = results_dataframes['raw_data'].query('case_id == @case_id_slack_26')
#%%
def kfold_cv_depth(df_dict, type_corr_analysis, dataset_ID, cases_id_depth_feas, plot_depth_exploration=False, dimensions_caseid_feasible = None, n_fold=5, params=None, score='accuracy'):
    df = df_dict['DataSet_training_uncorr_var'+type_corr_analysis+dataset_ID]
    df_training = pd.DataFrame(columns= df.columns)
    cases_id_training = []
    scores_df=pd.DataFrame(columns=['Depth','score_mean','score_std','n_training_cases','perc_stable'])
    cv = KFold(n_splits=n_fold, shuffle=True, random_state=23)

    if plot_depth_exploration:
        ax = plot_mesh(mesh_df)
    
    for depth in range(0,int(max(cases_id_depth_feas['Depth']))+1):
        add_case_id = list(cases_id_depth_feas.query('Depth == @depth')['case_id'])
        cases_id_training.extend(add_case_id)
        df_training = df.query('case_id == @cases_id_training')
        scores_df.loc[depth,'n_training_cases']=len(df_training)
        scores_df.loc[depth,'perc_stable']=len(df_training.query('Stability == 1'))/len(df_training)
    
        if len(df_training)>= n_fold:
            #clf = svm.SVC(kernel='linear', C=1, random_state=42)
            #clf = MLPClassifier(random_state=1, max_iter=5000, activation='relu')
            if params == None:
                clf = Pipeline([('scaler', StandardScaler()), ('xgb', XGBClassifier())])
            else:
                clf = Pipeline([('scaler', StandardScaler()), ('xgb', XGBClassifier(**params))])
            X = df_training.drop(['case_id','Stability'],axis=1).reset_index(drop=True)
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

def feval_specificity(preds, dtrain):
    y = dtrain.get_label().astype(int)
    y_hat = (preds >= 0.5).astype(int)   # or use your tuned threshold
    tn = ((y==0)&(y_hat==0)).sum(); fp = ((y==0)&(y_hat==1)).sum()
    spec = tn / (tn + fp) if (tn+fp) else 0.0
    return ("specificity", spec, True)  # higher is better

estimator = Pipeline([('scaler', RobustScaler()), ('xgb', XGBClassifier(eval_metric='aucpr',objective="binary:logistic", random_state=42,
                                                                        sample_weight=sw,
                                                                        eval_set=[(X_test, y_test)], verbose=False))])#scale_pos_weight=0.3 (len(Y)-Y.sum())/Y.sum()
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
model = XGBClassifier()
for type_corr_analysis in ['']: #_HierCl

    #df = df_dict['DataSet_training_uncorr_var'+type_corr_analysis+dataset_ID]
    df = results_dataframes['raw_data_slack_26']
     
    X = df.drop(['case_id','Stability','cell_name'],axis=1).reset_index(drop=True)[['V1','V2']]
    Y = df[['Stability']].reset_index(drop=True).values.astype(int).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, Y , train_size=0.9, shuffle=True, random_state=42)
    # w_neg, w_pos = 10.0, 1.0  # tune
    # sw = np.where(y_train == 0, w_neg, w_pos)

    # model=estimator.fit(X_train, y_train)
    model.fit(X_train, y_train)
    
    # proba = model.predict_proba(X_test)[:,1]

    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(score)
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    
    # disp = ConfusionMatrixDisplay.from_estimator(
    # model, X_test, y_test,
    # display_labels=["Class 0", "Class 1"],
    # cmap='Blues',
    # normalize='true'  # shows percentages per true class
    # )



#%% XGB CLASSIFIER

estimator = Pipeline([('scaler', RobustScaler()), ('xgb', XGBClassifier())])

param_grid = {'xgb__eta':[0.1,0.2,0.25, 0.3],#, 0.35], #np.arange(0.1,0.7,0.2),
              'xgb__max_depth':[4,3,5,6,7],#7,8,9],#[5,6,7],
              'xgb__subsample':[0.5,0.8,1],
              'xgb__n_estimators':[300,350]
    }

best_params_grid=dict()
scores_depth = dict()
for type_corr_analysis in ['']:#,'_HierCl']:
    df = df_dict['DataSet_training_uncorr_var'+type_corr_analysis+dataset_ID]
    
    X_train = df.drop(['case_id','Stability'],axis=1).reset_index(drop=True)
    Y_train = df[['Stability']].reset_index(drop=True).values.astype(int).ravel()

    best_model, best_params, means, stds, params = GSkFCV(param_grid, X_train, Y_train, estimator, 'accuracy')#, n_folds=3)

    
    best_params_grid[type_corr_analysis+dataset_ID]={'learning_rate': best_params['xgb__eta'],
                  'max_depth': best_params['xgb__max_depth'],
                  'subsample': best_params['xgb__subsample'], 'n_estimators': best_params['xgb__n_estimators']}

    
    scores_depth[type_corr_analysis+dataset_ID] = kfold_cv_depth(df_dict, type_corr_analysis, dataset_ID, cases_id_depth_feas, plot_depth_exploration=False, n_fold=5, params=best_params_grid[type_corr_analysis+dataset_ID])

# In[41]:


model = best_model

best_params_grid={'learning_rate': best_params['xgb__eta'],
              'max_depth': best_params['xgb__max_depth'],
              'subsample': best_params['xgb__subsample']}

# save best model parameters
f=open(path+dir_name+'/XGB_best_params_DataSet'+dataset_ID+'.sav','w')
f.write(str(best_params))
f.close()    


# In[42]:

estimator = Pipeline([('scaler', RobustScaler()), ('xgb', XGBClassifier(**best_params_grid))])
for type_corr_analysis in ['']: #_HierCl

    df = df_dict['DataSet_training_uncorr_var'+type_corr_analysis+dataset_ID]
     
    X = df.drop(['case_id','Stability'],axis=1).reset_index(drop=True)
    Y = df[['Stability']].reset_index(drop=True).values.astype(int).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, Y , train_size=0.8, shuffle=True, random_state=42)
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print('accuracy = ',accuracy_score(y_test, y_pred))
    print('precision = ',precision_score(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')



# In[43]:


scores_df_uncorr_HierCl_HPT = kfold_cv_depth(results_dataframes, 'uncorr_HierCl', dimensions_caseid_feasible, cases_id_depth_feas, plot_depth_exploration=False, n_fold=5, params=best_params_grid)


# In[44]:


scores_df_uncorr_HierCl_HPT


# In[45]:


fig, ax = plt.subplots()
ax.errorbar(scores_df_uncorr_HierCl['Depth'], scores_df_uncorr_HierCl['score_mean'], yerr=scores_df_uncorr_HierCl['score_std'], fmt='-o', capsize=5, color='orange', ecolor='black', elinewidth=1.5, label = 'uncorr vars HC')
ax.errorbar(scores_df_uncorr_HierCl_HPT['Depth'], scores_df_uncorr_HierCl_HPT['score_mean'], yerr=scores_df_uncorr_HierCl_HPT['score_std'], fmt='-o', capsize=5, color='blue', ecolor='black', elinewidth=1.5, label = 'uncorr vars HC-HT')
ax.set_xlabel('Depth')
ax.set_ylabel('Mean accuracy $\pm$ std')
ax.grid()
plt.legend()
fig.tight_layout()

#%%
pd.DataFrame.to_excel(scores_depth[type_corr_analysis+dataset_ID], path+dir_name+'/scores_df_uncorr_var_xgb'+dataset_ID+'.xlsx')#_var_HierCl_