"""
selects regression model 

"""
# from pyearth import Earth
# from pyearth import export
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

#%% Load Training Data

Training_data=pd.read_csv('../B.IdentificationOfGroupsOfCriticalEigenvalues/Training_inputs_DI_crit_LHS10_10_nreg9.csv').drop(['Iter_num','DI','PLTOT','f','Exitflag'],axis=1)
Training_data = pd.read_csv('../identification_of_critical_eigenvalues/Training_Inputs_DI_Crit.csv').drop(['Unnamed: 0', 'case_id', 'Stability'], axis=1)


#%% Remove correlated variables

corr_matrix=abs(Training_data.drop(['V1','Theta1'],axis=1).corr())
corr_matrix=corr_matrix.sort_values(by='DI_crit',ascending=False)
corr_matrix=corr_matrix.drop('DI_crit',axis=0)
uncorr_var=Training_data.drop(['V1','Theta1','DI_crit'],axis=1).columns

for var in corr_matrix.index:
    if var not in corr_matrix.index:
        continue
    corrs=corr_matrix[[var]].drop(var,axis=0)
    corrs_var=corrs.query('{var}>=0.9'.format(var=var)).index
    uncorr_var=list(set(uncorr_var)-set(corrs_var))
    corr_matrix=corr_matrix.drop(corrs_var,axis=0)
    

# #%% Train MARS model

Xtrain=Training_data[uncorr_var]
ytrain=Training_data['DI_crit']
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

# %%

models_list = ['LR','Lasso','Ridge','ElasticNet']
models_dict={'LR': LinearRegression(),'Lasso': Lasso(alpha=0.1),'Ridge': Ridge(alpha=0.1),'ElasticNet': ElasticNet()}

lin_model_trained={}

for name in models_list:
    lin_model_trained[name]=[]
    lin_model_trained[name].append(models_dict[name].fit(Xtrain,ytrain))

#%% Load Test data

r2_summary=pd.DataFrame()
ind=0
for obj_fun in ['Min_P_SG','Min_P_losses']:

    Test_data=pd.read_csv('../B.IdentificationOfGroupsOfCriticalEigenvalues/Test_data'+obj_fun+'_inputs_DI_crit.csv').drop(['DI','PLTOT','f','Exitflag','ObjFun'],axis=1)

    Xtest=Test_data[uncorr_var]
    ytest=Test_data['DI_crit']

    # pred_mars=mars_model.predict(Xtest)
    # r2_mars=r2_score(ytest,pred_mars.reshape(-1,1))

    r2_lin_models=[]
    for name in models_list:
        r2_lin_models.append(r2_score(ytest,lin_model_trained[name][0].predict(Xtest)))

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
obj_fun_list=['Min_P_SG','Min_P_losses']
for ii in range(len(obj_fun_list)) :
    obj_fun=obj_fun_list[ii]
    max_r2_ind=np.argmax(r2_summary.query('Obj_Fun == @obj_fun').T[ii].drop('Obj_Fun'))
    best_model.loc[ii, 'Obj_Fun']=obj_fun
    best_model.loc[ii,'Model']=r2_summary.query('Obj_Fun == @obj_fun').T.drop('Obj_Fun').index[max_r2_ind]

pd.DataFrame.to_csv(best_model,'Best_Model.csv')
pd.DataFrame.to_csv(r2_summary,'R2_summary.csv')
