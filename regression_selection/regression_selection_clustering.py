"""
selects regression model using only using damping indices created from eigencalues in the critical cluster

"""
# from pyearth import Earth
# from pyearth import export
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import os
import json
from sklearn.neural_network import MLPRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})




DIs_used = ['DI_crit', 'DI_all']
highest_r2_methods = {}
for DI_method in DIs_used:
    #%% Load Data and split it 
    # with open('../identification_of_critical_eigenvalues/data_number.json', 'r') as file:
    #     data_number = json.load(file)
    # Data = pd.read_csv(f'../identification_of_critical_eigenvalues/DI_crit_{data_number[-4:]}.csv').drop(['Unnamed: 0', 'case_id', 'Stability'], axis=1)

    Data_larger = pd.read_csv(f'../identification_of_critical_eigenvalues/{DI_method}_8464.csv').drop(['Unnamed: 0', 'case_id', 'Stability'], axis=1)
    Data_larger = Data_larger.fillna(0)
    # Data_larger = Data_larger.drop('Stable',axis=1)

    Data_half = Data_larger.iloc[0:int(len(Data_larger)/2)]

    Data_smaller = pd.read_csv(f'../identification_of_critical_eigenvalues/{DI_method}_9253.csv').drop(['Unnamed: 0', 'case_id', 'Stability'], axis=1)
    Data_smaller = Data_smaller.fillna(0)
    Data_smaller = Data_smaller.drop('Stable',axis=1)

    Data_combined = pd.concat([Data_larger, Data_smaller])
    
    
    Data = [Data_half, Data_larger, Data_combined]
    Data_sizes = [Data_half.shape[0], Data_larger.shape[0], Data_combined.shape[0]]
    
    r2_summary=pd.DataFrame()
    for i in range(len(Data)):
        #%% Remove correlated variables
        corr_matrix=abs(Data[i].corr())
        corr_matrix=corr_matrix.sort_values(by=DI_method,ascending=False)
        corr_matrix=corr_matrix.drop(DI_method,axis=0)
        uncorr_var=Data[i].columns
        
        for var in corr_matrix.index:
            if var not in corr_matrix.index:
                continue
            corrs=corr_matrix[[var]].drop(var,axis=0)
            corrs_var=corrs.query('{var}>=0.9'.format(var=var)).index
            uncorr_var=list(set(uncorr_var)-set(corrs_var))
            corr_matrix=corr_matrix.drop(corrs_var,axis=0)
           
        # split into test train 
        X = Data[i].drop(DI_method, axis=1)
        y=Data[i][DI_method]
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.10, random_state=42)
        
        
        #%% Train MARS model
        
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
        
        models_list = ['LR','Lasso','Ridge','ElasticNet','DT','RF','MLP']
        models_dict={'LR': LinearRegression(),'Lasso': Lasso(alpha=0.01, max_iter=5000),'Ridge': Ridge(alpha=0.005),'ElasticNet': ElasticNet(),
                    'DT': DecisionTreeRegressor(),'RF':RandomForestRegressor(),'MLP':MLPRegressor(max_iter=5000)}
        
        lin_model_trained={}
        
        for name in models_list:
            lin_model_trained[name]=[]
            lin_model_trained[name].append(models_dict[name].fit(Xtrain,ytrain))
        
        #%% calculate r2 scores 
        
        obj_func_ind=0
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
        
            r2_summary.loc[i,'Obj_Fun']=obj_fun
            r2_summary.loc[i, 'Data_Sizes'] = Data_sizes[i]
            # r2_summary.loc[obj_func_ind, 'MARS'] = r2_mars
            
            for lin_model in models_list:
                r2_summary.loc[i,lin_model]=r2_lin_models.loc[0,lin_model]
        
            obj_func_ind=obj_func_ind+1
        
        
    #%% get highest scores 
    highest_r2 = r2_summary.iloc[:, :2].copy()
    r2_summary['Method'] = np.nan
    r2_summary['Highest_score'] = np.nan
    for row in range(len(highest_r2)):
        # want the third column to be the method 
        highest_r2.loc[row,'Method']=r2_summary.drop(['Obj_Fun','Data_Sizes' ],axis=1).iloc[row,:].idxmax()
        
        # want the fourth column to be the highest score 
        highest_r2.loc[row,'Highest_score']=r2_summary.drop(['Obj_Fun','Data_Sizes' ],axis=1).iloc[row,:].max()
    
    highest_r2.rename(columns={'Data_Sizes': 'Best_Method'}, inplace=True)
    highest_r2.rename(columns={'LR': 'Highest_Score'}, inplace=True)
    
    
    print(r2_summary)
    
    pd.DataFrame.to_csv(r2_summary,f'R2_summary_{DI_method}.csv')
    pd.DataFrame.to_csv(highest_r2,f'Highest_Scores_{DI_method}.csv')
    
    # plot the highest scores 
    fig = plt.figure(figsize=(5, 6))
    ax = fig.add_subplot()
    ax.scatter(highest_r2['Best_Method'], highest_r2['Highest_score'])
    ax.grid()
    ax.set_title(f'R2 Scores for {DI_method}', fontsize=15)
    ax.set_xlabel('Number of Values', fontsize=15)
    fig.tight_layout()
    # fig.savefig('.png')
    fig.show()


    highest_r2_methods[DI_method] = highest_r2

# plot the graph with both DI generation methods 

# plot the highest scores 
fig = plt.figure(figsize=(5, 6))
ax = fig.add_subplot()
ax.scatter(highest_r2_methods['DI_crit']['Best_Method'], highest_r2_methods['DI_crit']['Highest_score'], label='With Clustering')
ax.scatter(highest_r2_methods['DI_all']['Best_Method'], highest_r2_methods['DI_all']['Highest_score'], label='Without Clustering')
ax.grid()
ax.set_title(f'R2 Scores', fontsize=15)
ax.set_xlabel('Number of Values', fontsize=15)
ax.legend()
fig.tight_layout()
# fig.savefig('.png')
fig.show()


