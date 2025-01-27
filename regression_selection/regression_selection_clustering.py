"""
selects regression model using only using damping indices created from eigencalues in the critical cluster

"""
# from pyearth import Earth
# from pyearth import export
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split
import os

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

#%% Load Data and split it 
Data = pd.read_csv('../identification_of_critical_eigenvalues/Data_DI_Crit.csv').drop(['Unnamed: 0', 'case_id', 'Stability'], axis=1)
Data = Data.dropna()
Data = Data.fillna(0)
#%% Remove correlated variables

corr_matrix=abs(Data.corr())
corr_matrix=corr_matrix.sort_values(by='DI_crit',ascending=False)
corr_matrix=corr_matrix.drop('DI_crit',axis=0)
uncorr_var=Data.columns

for var in corr_matrix.index:
    if var not in corr_matrix.index:
        continue
    corrs=corr_matrix[[var]].drop(var,axis=0)
    corrs_var=corrs.query('{var}>=0.9'.format(var=var)).index
    uncorr_var=list(set(uncorr_var)-set(corrs_var))
    corr_matrix=corr_matrix.drop(corrs_var,axis=0)
   
# split into test train 
X = Data.drop('DI_crit', axis=1)
y=Data['DI_crit']
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.30, random_state=42)


# #%% Train MARS model

# mars_model=Earth(feature_importance_type='gcv')#smooth=False, max_degree=2)
# mars_model.fit(Xtrain,ytrain)

# mars_summary = mars_model.summary()
# feat_imp = mars_model.summary_feature_importances()

# a_file=open('feature_importance.txt','w')
# a_file.write(feat_imp[14:])
# a_file.close()

# feat_imp = pd.read_csv('feature_importance.txt',header=None)
# feat_imp_df=pd.DataFrame()

# for i in range(0,len(feat_imp)):
#     feat_imp_df.loc[i, 'VAR'] = feat_imp.loc[i,0].split(' ')[0]

# feat_imp_df['GCV'] = mars_model.feature_importances_

# feat_imp_df = feat_imp_df.sort_values(by='GCV').reset_index(drop=True)

# feat_imp_df.loc[0, 'cumulative'] = feat_imp_df.loc[0, 'GCV']

# for ii in range(1, len(feat_imp_df)):
#     feat_imp_df.loc[ii, 'cumulative'] = (feat_imp_df.loc[ii - 1, 'cumulative'] + feat_imp_df.loc[ii, 'GCV'])

# feat_imp_df['cumulative'] = feat_imp_df['cumulative'] * 100

# feat_imp_df = feat_imp_df.query('GCV!=0')
# labels = list(feat_imp_df['VAR'])
# fig = plt.figure(figsize=(5, 6))
# ax = fig.add_subplot()
# ax.scatter(feat_imp_df['cumulative'], np.arange(0, len(feat_imp_df)))
# ax.set_yticklabels(labels)
# #ax.yaxis.set_ticks(np.arange(0,len(labels)),labels)
# ax.grid()
# ax.set_title('GCV-based Features Importance \n GD-driven Case', fontsize=15)
# ax.set_xlabel('Importance', fontsize=15)
# fig.tight_layout()
# fig.savefig('Feature_Importance.png')

# %% train other models 

models_list = ['LR','Lasso','Ridge','ElasticNet']
models_dict={'LR': LinearRegression(),'Lasso': Lasso(alpha=0.001),'Ridge': Ridge(alpha=0.005),'ElasticNet': ElasticNet()}

lin_model_trained={}

for name in models_list:
    lin_model_trained[name]=[]
    lin_model_trained[name].append(models_dict[name].fit(Xtrain,ytrain))

#%% calculate r2 scores 

r2_summary=pd.DataFrame()
ind=0
for obj_fun in ['one_objective_function']: #['Min_P_SG','Min_P_losses']:

    # pred_mars=mars_model.predict(Xtest)
    # r2_mars=r2_score(ytest,pred_mars.reshape(-1,1))

    # fig=plt.figure(figsize=(20,5))
    # ax=fig.add_subplot()

    r2_lin_models=[]
    for name in models_list:
        pred=lin_model_trained[name][0].predict(Xtest)
        r2_lin_models.append(r2_score(ytest,lin_model_trained[name][0].predict(Xtest)))
        #ax.scatter(ytest,pred)

    r2_lin_models=pd.DataFrame(r2_lin_models).T
    r2_lin_models.columns=models_list

    r2_summary.loc[ind,'Obj_Fun']=obj_fun
    # r2_summary.loc[ind, 'MARS'] = r2_mars
    for lin_model in models_list:
        r2_summary.loc[ind,lin_model]=r2_lin_models.loc[0,lin_model]

    ind=ind+1

print(r2_summary)

#%% Select Best Model
best_model=pd.DataFrame()
obj_fun_list=['one_objective_function'] #['Min_P_SG','Min_P_losses']
for ii in range(len(obj_fun_list)) :
    obj_fun=obj_fun_list[ii]
    max_r2_ind=np.argmax(r2_summary.query('Obj_Fun == @obj_fun').T[ii].drop('Obj_Fun'))
    best_model.loc[ii, 'Obj_Fun']=obj_fun
    best_model.loc[ii,'Model']=r2_summary.query('Obj_Fun == @obj_fun').T.drop('Obj_Fun').index[max_r2_ind]

pd.DataFrame.to_csv(best_model,'Best_Model_Crit.csv')
pd.DataFrame.to_csv(r2_summary,'R2_summary_Crit.csv')
