import pandas as pd
import os
#%%
path = 'D:/datagen_ACOPF_LF09_seed3_nc3_ns333_d5_20251115_160729_5148/'


files_list=['df_op','dims_df', 'cases_df','df_computing_times','df_imag','df_real']

for file_name in files_list:
    files_cell = [file for file in os.listdir(path) if file.startswith(file_name)]
    
    df = pd.DataFrame()
    for ff in files_cell:
        df_cell = pd.read_csv(path+ff)
        df = pd.concat([df,df_cell],axis=0)
    df =df.reset_index(drop=True)
    df.to_csv(path+file_name+'.csv')
    
#%%

df_op = pd.read_csv(path+'df_op.csv')
df_op.groupby('Stability').count()