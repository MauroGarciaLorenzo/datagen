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
dir_name=[dir_name for dir_name in os.listdir(path) if '_2862' in dir_name and 'zip' not in dir_name][0]# if dir_name.startswith('datagen') and 'zip' not in dir_name]#
print(dir_name)

#%%

dataset_ID = dir_name[-5:]

#%%
df_dict=dict()
df_dict['DataSet_training_uncorr_var'+dataset_ID] = pd.read_csv('DataSet_training_uncorr_var'+dataset_ID+'.csv').drop('Unnamed: 0', axis=1).drop_duplicates(keep='first')
df_dict['DataSet_training_uncorr_var_HierCl'+dataset_ID] = pd.read_csv('DataSet_training_uncorr_var_HierCl'+dataset_ID+'.csv').drop('Unnamed: 0', axis=1).drop_duplicates(keep='first')

case_id_feasible = list(df_dict['DataSet_training_uncorr_var'+dataset_ID]['case_id'])
cases_id_depth = pd.read_excel('cases_id_depth'+dataset_ID+'.xlsx')[['Depth','case_id','CellName']]

cases_id_depth_feas = cases_id_depth.query('case_id == @case_id_feasible')

#%%
def kfold_cv_depth(df_dict, type_corr_analysis, dataset_ID, cases_id_depth_feas, plot_depth_exploration=False, dimensions_caseid_feasible = None, n_fold=5, params=None):
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
            scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
            
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

for type_corr_analysis in ['']: #_HierCl

    df = df_dict['DataSet_training_uncorr_var'+type_corr_analysis+dataset_ID]
     
    X = df.drop(['case_id','Stability'],axis=1).reset_index(drop=True)
    Y = df[['Stability']].reset_index(drop=True).values.astype(int).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, Y , train_size=0.8, shuffle=True, random_state=42)
    w_neg, w_pos = 10.0, 1.0  # tune
    sw = np.where(y_train == 0, w_neg, w_pos)

    # model=estimator.fit(X_train, y_train)
    model.fit(X_train, y_train)
    
    proba = model.predict_proba(X_test)[:,1]

    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    
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
              'xgb__max_depth':[4,3,5],#,6,7],#7,8,9],#[5,6,7],
              'xgb__subsample':[1] #[0.5,1]
    }

best_params_grid=dict()
scores_depth = dict()
for type_corr_analysis in ['','_HierCl']:
    df = df_dict['DataSet_training_uncorr_var'+type_corr_analysis+dataset_ID]
    
    X_train = df.drop(['case_id','Stability'],axis=1).reset_index(drop=True)
    Y_train = df[['Stability']].reset_index(drop=True).values.astype(int).ravel()

    best_model, best_params, means, stds, params = GSkFCV(param_grid, X_train, Y_train, estimator, 'accuracy')#, n_folds=3)

    
    best_params_grid[type_corr_analysis+dataset_ID]={'learning_rate': best_params['xgb__eta'],
                  'max_depth': best_params['xgb__max_depth'],
                  'subsample': best_params['xgb__subsample']}

    
    scores_depth[type_corr_analysis+dataset_ID] = kfold_cv_depth(df_dict, type_corr_analysis, dataset_ID, cases_id_depth_feas, plot_depth_exploration=False, n_fold=5, params=best_params_grid[type_corr_analysis+dataset_ID])

# In[41]:


model = best_model

best_params_grid={'learning_rate': best_params['xgb__eta'],
              'max_depth': best_params['xgb__max_depth'],
              'subsample': best_params['xgb__subsample']}

# save best model parameters
f=open(path+'XGB_best_params_DataSet'+dataset_ID+'.sav','w')
f.write(str(best_params))
f.close()    


# In[42]:


best_params_grid


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
pd.DataFrame.to_excel(scores_df,'scores_df_uncorr_var_xgb'+dataset_ID+'.xlsx')#_var_HierCl_