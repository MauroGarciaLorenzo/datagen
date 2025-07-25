import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%

import os

def read_txt_files_from_folder(folder_path):
    """
    Reads all .txt files in the given folder and returns their content as a list of lists.

    Parameters:
    - folder_path (str): Path to the folder containing .txt files.

    Returns:
    - list of lists: Each inner list contains the lines (as strings) from one file.
    """
    all_contents = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as f:
                lines = [float(line.strip()) for line in f.readlines()]
                all_contents.append(lines)

    return all_contents

#%%

path2txt= '../results/'

all_voltages=read_txt_files_from_folder(path2txt)

#%%
#path2res= path2txt+'ACOPF_standalone_NREL_LF09_seed16_nc3_ns100_20250530_165211_8618/'
path2res= path2txt+'MareNostrum/datagen_ACOPF_slurm22543545_cu16_nodes32_LF09_seed16_nc30_ns50_d6_20250613_220923_5611/'

df_op=pd.read_csv(path2res+'case_df_op.csv').query('Stability >=0')#.query('Stability !=-1')
voltage_conv=df_op[[c for c in df_op.columns if c.startswith('V')]]
cases_conv=df_op[['case_id']]

df_v_sampled=pd.DataFrame(columns=voltage_conv.columns)
for idx_case,case in cases_conv.iterrows():
    file_path = os.path.join(path2txt, case[0]+'.txt')
    with open(file_path, 'r') as f:
        lines = [float(line.strip()) for line in f.readlines()]

        df_v_sampled.loc[idx_case]=lines

#%%        
fig=plt.figure()
ax=fig.add_subplot()
for ind in voltage_conv.index:
    ax.plot(voltage_conv.loc[ind])    
    ax.plot(df_v_sampled.loc[ind], c='k', alpha=0.5)
    
#%%

fig=plt.figure()
ax=fig.add_subplot()
for ind in range(0,len(all_voltages)):
    ax.plot(all_voltages[ind], c='k', alpha=0.5)
    
#%%

#path2vset='C:/Users/Francesca/miniconda3/envs/gridcal_original2/datagen/results/ACOPF_standalone_NREL_LF09_seed16_nc3_ns100_20250529_181047_1763/'
path2vset= path2txt+'ACOPF_standalone_NREL_LF09_seed16_nc3_ns100_20250603_112420_3079/'
df_op_vset=pd.read_csv(path2vset+'case_df_op.csv').query('Stability !=-1')
voltage_conv_vset=df_op_vset[[c for c in df_op_vset.columns if c.startswith('V')]]

fig=plt.figure()
ax=fig.add_subplot()
for ind in np.unique(list(voltage_conv.index)+list(voltage_conv_vset.index)):
    if ind==102:
        ax.plot(np.arange(0,118),voltage_conv.loc[ind],'k')
        # ax.plot(np.arange(0,118),voltage_conv_vset.loc[ind],'k', alpha=0.1)
    if ind in list(voltage_conv.index) and ind in list(voltage_conv_vset.index):
        ax.plot(np.arange(0,118),voltage_conv.loc[ind],'b')
        # ax.plot(np.arange(0,118),voltage_conv_vset.loc[ind],'-r')
    elif ind in list(voltage_conv.index) and ind not in list(voltage_conv_vset.index):
        ax.plot(np.arange(0,118),voltage_conv.loc[ind],'b', alpha=0.5)
    #elif ind not in list(voltage_conv.index) and ind in list(voltage_conv_vset.index):
        # ax.plot(np.arange(0,118),voltage_conv_vset.loc[ind],'-r', alpha=0.5)

fig=plt.figure()
ax=fig.add_subplot()
for ind in np.unique(list(voltage_conv.index)+list(voltage_conv_vset.index)):
    if ind==102:
        #ax.plot(np.arange(0,118),voltage_conv.loc[ind],'k')
        ax.plot(np.arange(0,118),voltage_conv_vset.loc[ind],'k')
    if ind in list(voltage_conv.index) and ind in list(voltage_conv_vset.index):
        #ax.plot(np.arange(0,118),voltage_conv.loc[ind],'b')
        ax.plot(np.arange(0,118),voltage_conv_vset.loc[ind],'-r')
    # elif ind in list(voltage_conv.index) and ind not in list(voltage_conv_vset.index):
    #     ax.plot(np.arange(0,118),voltage_conv.loc[ind],'b', alpha=0.5)
    elif ind not in list(voltage_conv.index) and ind in list(voltage_conv_vset.index):
        ax.plot(np.arange(0,118),voltage_conv_vset.loc[ind],'-r', alpha=0.5)

#%%

fig=plt.figure()
ax=fig.add_subplot()
for ind in list(voltage_conv_vset.index):
    if ind==102:
        ax.plot(np.arange(0,118),voltage_conv.loc[ind],'k')
        ax.plot(np.arange(0,118),voltage_conv_vset.loc[ind],'k', alpha=0.1)
    else:
        ax.plot(np.arange(0,118),voltage_conv_vset.loc[ind],'k', alpha=0.1)
        
#%%
voltage_conv_vset_diff=voltage_conv_vset.T.diff(1)
voltage_conv_vset_diff_descr=voltage_conv_vset_diff.T.describe()

#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def plot_histogram_with_normal_counts(df, column, bins=30):
    """
    Plots a histogram (counts, not density) of a DataFrame column with an overlaid normal distribution curve scaled to counts.
    """
    data = df[column].dropna()
    mu, std = np.mean(data), np.std(data)  # ✅ Correct way to compute sample mean and std
    n = len(data)

    # Histogram (returns bin edges and patch container)
    counts, bin_edges, _ = plt.hist(data, bins=bins, alpha=0.6, color='skyblue', edgecolor='black', label='Histogram')

    # Bin centers
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    bin_width = bin_edges[1] - bin_edges[0]

    # Normal PDF scaled to match histogram counts
    pdf_vals = norm.pdf(bin_centers, mu, std) * n * bin_width

    # Plot the fitted normal distribution
    plt.plot(bin_centers, pdf_vals, 'r-', linewidth=2, label=f'Normal fit\nμ={mu:.2f}, σ={std:.2f}')
    plt.title(f'Histogram of {column} with Normal Fit')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


plot_histogram_with_normal_counts(voltage_conv_vset_diff.T.reset_index(drop=True), 'V10')

#%%
import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram_with_kde(df, column, bins=30):
    """
    Plots a histogram with count bars and an overlaid KDE curve for a specified DataFrame column.
    """
    data = df[column].dropna()

    plt.figure(figsize=(8, 5))
    
    # Histogram (bars) with count scale
    sns.histplot(data, bins=bins, stat="count", color='skyblue', edgecolor='black', label='Histogram', kde=True)

    # KDE overlaid
    #sns.kdeplot(data, color='red', linewidth=2, label='KDE')

    plt.title(f'Histogram with KDE for "{column}"')
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_histogram_with_kde(voltage_conv_vset_diff.T.reset_index(drop=True), 'V20')
plot_histogram_with_kde(voltage_conv_vset, 'V1')

#%%

V_diff_min_max=voltage_conv_vset_diff_descr.loc[['min','max']]
V_diff_min_max.loc['min','V1']=voltage_conv_vset['V1'].min()
V_diff_min_max.loc['max','V1']=voltage_conv_vset['V1'].max()

def gen_voltage_profile_RandUnifDiff(V_diff_min_max):

    voltages = [0]*np.shape(V_diff_min_max)[1]
    indx_id=np.zeros([np.shape(V_diff_min_max)[1],2])

    ibus=1
    V= np.random.uniform(low=V_diff_min_max.loc['min','V'+str(ibus)], high=V_diff_min_max.loc['max','V'+str(ibus)], size=1)
    
    voltages[0]=V
    indx_id[0,1]=ibus
    
    for index, ibus in enumerate(range(2,np.shape(V_diff_min_max)[1]+1)):
        V_diff= np.random.uniform(low=V_diff_min_max.loc['min','V'+str(ibus)], high=V_diff_min_max.loc['max','V'+str(ibus)], size=1)
        voltages[index+1]=voltages[index]+V_diff
        indx_id[index+1,0]=index
        indx_id[index+1,1]=ibus

    return voltages, indx_id

voltages, indx_id = gen_voltage_profile_RandUnifDiff(V_diff_min_max)

fig=plt.figure()
ax=fig.add_subplot()
for ind in list(voltage_conv_vset.index):
    if ind==13:
        ax.plot(np.arange(0,118),voltage_conv.loc[ind],'k')
        ax.plot(np.arange(0,118),voltage_conv_vset.loc[ind],'k', alpha=0.1)
    else:
        ax.plot(np.arange(0,118),voltage_conv_vset.loc[ind],'k', alpha=0.1)
        

ax.plot(np.arange(0,118),voltages,'r')

#%%

voltage_conv=voltage_conv.sort_values(by='V80')
fig=plt.figure()
ax=fig.add_subplot()
for ind in list(voltage_conv.index):
    if ind==102:
        ax.plot(np.arange(0,118),voltage_conv.loc[ind],'k')
    else:
        ax.plot(np.arange(0,118),voltage_conv.loc[ind],'k', alpha=0.1)

list(voltage_conv.loc[102])