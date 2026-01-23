import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils_pp_standalone import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#%%
#path = '../results/'
path = 'D:/'
dir_name=[dir_name for dir_name in os.listdir(path) if '_9075' in dir_name and 'zip' not in dir_name][0]# if dir_name.startswith('datagen') and 'zip' not in dir_name]#
print(dir_name)
# dir_names = [
#     #'datagen_ACOPF_slurm23172357_cu10_nodes32_LF09_seed3_nc3_ns500_d7_20250627_214226_7664']
#     'datagen_ACOPF_slurm25105245_cu8_nodes32_LF09_seed3_nc3_ns500_d7_20250731_132256_7665']

#%%
path_results = os.path.join(path, dir_name)
df_op='df_op'#'case_df_op'
results_dataframes, csv_files = open_csv(
    path_results, ['cases_df.csv', df_op+'.csv'])

perc_stability(results_dataframes['cases_df'], dir_name)

dataset_ID = dir_name[-5:]

#%%
new_df_op=pd.DataFrame(columns = results_dataframes[df_op].columns, index=np.arange(0,len(results_dataframes[df_op])))
new_df_op.loc[0:1000, new_df_op.columns] = results_dataframes[df_op].loc[0:1000, new_df_op.columns]

for ii in np.arange(1000,len(results_dataframes['cases_df']),1000):
    aa = results_dataframes[df_op].iloc[ii:ii+1000]
    if len (list(set(aa['Stability'].unique())-set([-1,1,-2,0])))>0:
        print('porco dio')
    new_df_op.iloc[ii:ii+1000, new_df_op.columns.get_indexer(
        new_df_op.columns[0:np.where(new_df_op.columns == 'P_GFOL27')[0][0]]
    )] = aa.iloc[:, new_df_op.columns.get_indexer(
        new_df_op.columns[0:np.where(new_df_op.columns == 'P_GFOL27')[0][0]]
    )].to_numpy()
    
    new_df_op.iloc[ii:ii+1000, new_df_op.columns.get_indexer(['P_GFOL24','Q_GFOL24','Sn_GFOL24'])] = \
        aa[['P_GFOL27','Q_GFOL27','Sn_GFOL27']].to_numpy()
    
    new_df_op.iloc[ii:ii+1000, new_df_op.columns.get_indexer(
        new_df_op.columns[np.where(new_df_op.columns == 'P_GFOL27')[0][0]:np.where(new_df_op.columns == 'case_id')[0][0]]
    )] = aa.iloc[:, new_df_op.columns.get_indexer(
        new_df_op.columns[np.where(new_df_op.columns == 'P_GFOL27')[0][0]:np.where(new_df_op.columns == 'case_id')[0][0]]
    )].to_numpy()
    
    # ✅ fixed line
    new_df_op.iloc[ii:ii+len(aa), new_df_op.columns.get_indexer(['case_id','cell_name','Stability'])] = \
        aa[['P_GFOL24','Q_GFOL24','Sn_GFOL24']].to_numpy()
    
    if len (list(set(new_df_op.loc[ii:ii+999,'Stability'].unique())-set([-1,1,-2,0])))>0:
         print('error')
        
#%%
for ii in np.arange(1000,len(results_dataframes['cases_df']),1000):
    aa = new_df_op.iloc[ii:ii+1000]
    if len (list(set(aa['Stability'].unique())-set([-1,1,-2,0])))>0:
        break


#%%
pd.DataFrame.to_csv(new_df_op,path_results+'/fixed_df_op.csv')