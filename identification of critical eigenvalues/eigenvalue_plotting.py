"""
This file ...


"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from datagen import NAN_COLUMN_NAME

#%% load data

df_real=pd.read_csv('../results/datagen_ACOPF_LF09_seed17_nc5_ns5_d5_20241119_115327_8464/case_df_real.csv')
df_imag=pd.read_csv('../results/datagen_ACOPF_LF09_seed17_nc5_ns5_d5_20241119_115327_8464/case_df_imag.csv')

# remove the empty rows (from the samples that didn't converge)
df_real_clean = df_real.dropna(subset=['2'])
df_imag_clean = df_imag.dropna(subset=['2'])

# remove the first column and the  case_id column
df_real_clean = df_real_clean.drop([df_real_clean.columns[0], 'case_id', 'Stability'], axis=1)
df_imag_clean = df_imag_clean.drop([df_imag_clean.columns[0], 'case_id', 'Stability'], axis=1)

n_cases= len(df_real)

# plot the full modal map
fig=plt.figure()
ax=fig.add_subplot()
ax.scatter(df_real_clean,df_imag_clean, label='Eigenvalues')
ax.set_xlabel('Real Axis',fontsize=25)
ax.set_ylabel('Imaginary Axis',fontsize=25)
ax.set_title('Modal Map', fontsize=25)
ax.tick_params(labelsize=20)
ax.legend(loc='lower center',bbox_to_anchor=(0.45, -0.65),fontsize=15, ncol=2)
fig.tight_layout()
plt.grid()
fig.savefig('Modal_Map.png')

# zoom in
fig = plt.figure()
ax=fig.add_subplot()
ax.scatter(df_real_clean,df_imag_clean, label='Eigenvalues')
ax.set_xlabel('Real Axis',fontsize=25)
ax.set_ylabel('Imaginary Axis',fontsize=25)
ax.set_title('Zoom', fontsize=25)
ax.tick_params(labelsize=20)
# ax.set_xlim([-100000,5000])
ax.set_xlim([-20,20])
ax.set_ylim([200,300])
ax.legend(loc='lower center',bbox_to_anchor=(0.45, -0.65),fontsize=15, ncol=2)
fig.tight_layout()
plt.grid()
fig.savefig('Middle_Group_Zoom.png')

