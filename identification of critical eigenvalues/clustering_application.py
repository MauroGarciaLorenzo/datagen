import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
import copy

import joblib

#%% Load the trained kmeans 
kmeans=joblib.load('kmeans3clusters.sav')

#%%

df_real=pd.read_csv('../results/datagen_ACOPF_slurm12492427_cu8_nodes1_seed17_nc5_ns5_d5_20241203_090628_5458/case_df_real.csv')
df_imag=pd.read_csv('../results/datagen_ACOPF_slurm12492427_cu8_nodes1_seed17_nc5_ns5_d5_20241203_090628_5458/case_df_imag.csv')

# remove the empty rows (from the samples that didn't converge)
df_real_clean = df_real.dropna(subset=['2'])
df_imag_clean = df_imag.dropna(subset=['2'])

# remove the first column and the  case_id column
df_real_clean = df_real_clean.drop([df_real_clean.columns[0], 'case_id', 'Stability'], axis=1).reset_index(drop=True)
df_imag_clean = df_imag_clean.drop([df_imag_clean.columns[0], 'case_id', 'Stability'], axis=1).reset_index(drop=True)

n_cases_clean= len(df_real_clean)

#%%
# select the eigenvalues that have the highest real parts in each row 
df_real_clean_max = df_real_clean[df_real_clean.columns[0:2]]
df_imag_clean_max = df_imag_clean[df_imag_clean.columns[0:2]]

#%% plot the modal map 

fig=plt.figure()
ax=fig.add_subplot()
ax.set_xlabel('Real Axis',fontsize=25)
ax.set_ylabel('Imaginary Axis',fontsize=25)
# ax.set_xlim([-80,20])
# ax.set_ylim([200,32-0])
ax.tick_params(labelsize=20)
fig.tight_layout()
plt.grid()
ax.scatter(df_real_clean,df_imag_clean, label='Eigenvalues')
ax.scatter(df_real_clean_max, df_imag_clean_max, label='Max Eigenvalues')
ax.legend(loc='lower center',bbox_to_anchor=(0.45, -0.65),fontsize=15, ncol=2)
ax.set_title('Modal Map', fontsize=25)
plt.show()