"""
This file plots the modal map of the eigenvalues on the real-imaginary axis
It also identifies critical eigenvalues through different clustering methods
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
import copy
import joblib

#%% 
def exclude_50Hz_eig(real_selected_df, imag_selected_df, plot_figure=True):
    
    imag_selected_df=imag_selected_df.round(3)

    imag_selected_no50Hz=pd.DataFrame()
    real_selected_no50Hz=pd.DataFrame()

    for i in real_selected_df.columns:
        imag_serie = imag_selected_df[imag_selected_df[i] != 314.159][i]  # Use boolean indexing instead of query
        real_serie = real_selected_df.loc[imag_serie.index,i]
        
        if i ==0:
            imag_selected_no50Hz=pd.DataFrame(imag_serie).copy(deep=True).reset_index(drop=True)
            real_selected_no50Hz=pd.DataFrame(real_serie).copy(deep=True).reset_index(drop=True)
        else:
            real_selected_no50Hz=real_selected_no50Hz.merge(pd.DataFrame(real_serie).reset_index(drop=True), how='outer', left_index=True, right_index=True)
            imag_selected_no50Hz=imag_selected_no50Hz.merge(pd.DataFrame(imag_serie).reset_index(drop=True), how='outer', left_index=True, right_index=True)

    if plot_figure:
        fig=plt.figure()
        ax=fig.add_subplot()
        ax.set_xlabel('Real Axis',fontsize=25)
        ax.set_ylabel('Imaginary Axis',fontsize=25)
        ax.set_xlim(x_region)
        ax.set_ylim(y_region)
        ax.tick_params(labelsize=20)
        fig.tight_layout()
        plt.grid()
        ax.scatter(real_selected_no50Hz,imag_selected_no50Hz, label='Eigenvalues')
        ax.legend(loc='lower center',bbox_to_anchor=(0.45, -0.65),fontsize=15, ncol=2)
        ax.set_title('Modal Map Zoom - No 50Hz eigs', fontsize=25)
        plt.show()
    return real_selected_no50Hz, imag_selected_no50Hz

def check_cluster_memberships(labels_reshape, k):
    ClustersMissingEigs = 0
    SampleClusterPairs = []  # List to store (sample, cluster) pairs
    
    for i in range(labels_reshape.shape[1]):  # For every column (sample)
        for l in range(k):  # For each cluster
            if not any(labels_reshape[:, i] == l): 
                print(f'There is no eigenvalue from sample {i} in cluster {l}')
                ClustersMissingEigs = 1
                SampleClusterPairs.append((i, l))  # Append the (sample, cluster) pair
    
    return ClustersMissingEigs, SampleClusterPairs

#%% import and clean data
df_real=pd.read_csv('../results/datagen_ACOPF_LF09_seed17_nc5_ns5_d5_20241119_115327_8464/case_df_real.csv')
df_imag=pd.read_csv('../results/datagen_ACOPF_LF09_seed17_nc5_ns5_d5_20241119_115327_8464/case_df_imag.csv')

# remove the empty rows (from the samples that didn't converge)
df_real_clean = df_real.dropna(subset=['2'])
df_imag_clean = df_imag.dropna(subset=['2'])

# remove the first column and the  case_id column
df_real_clean = df_real_clean.drop([df_real_clean.columns[0], 'case_id', 'Stability'], axis=1).reset_index(drop=True)
df_imag_clean = df_imag_clean.drop([df_imag_clean.columns[0], 'case_id', 'Stability'], axis=1).reset_index(drop=True)

n_cases_clean= len(df_real_clean)

# select the eigenvalues that have the highest real parts in each row 
df_real_clean_max = df_real_clean[df_real_clean.columns[0:2]]
df_imag_clean_max = df_imag_clean[df_imag_clean.columns[0:2]]

#%% plot the modal map 

fig=plt.figure()
ax=fig.add_subplot()
ax.set_xlabel('Real Axis',fontsize=25)
ax.set_ylabel('Imaginary Axis',fontsize=25)

ax.tick_params(labelsize=20)
fig.tight_layout()
plt.grid()
ax.scatter(df_real_clean,df_imag_clean, label='Eigenvalues')
ax.scatter(df_real_clean_max, df_imag_clean_max, label='Max Eigenvalues')
ax.legend(loc='lower center',bbox_to_anchor=(0.45, -0.65),fontsize=15, ncol=2)
ax.set_title('Modal Map', fontsize=25)
plt.show()

#%% select the targeted region 

complete = 1; # 1 for the complete set, 0 for the sepcific region 
if complete==1:
    x_region = [-115,20]
    y_region = [200,320]
else:
    x_region = [-1,20]
    y_region = [-400,400]
    
#%% Plot the Modal Map 

fig=plt.figure()
ax=fig.add_subplot()
ax.set_xlabel('Real Axis',fontsize=25)
ax.set_ylabel('Imaginary Axis',fontsize=25)
ax.set_xlim(x_region)
ax.set_ylim(y_region)
ax.tick_params(labelsize=20)
fig.tight_layout()
plt.grid()
ax.scatter(df_real_clean,df_imag_clean, label='Eigenvalues')
ax.scatter(df_real_clean_max, df_imag_clean_max, label='Max Eigenvalues')
ax.legend(loc='lower center',bbox_to_anchor=(0.45, -0.65),fontsize=15, ncol=2)
ax.set_title('Modal Map', fontsize=25)
plt.show()

#%% delete the nan values and their cooresponding values from selected_real and selected_imag

# select eigenvalues in the region
selected_eig_real = df_real_clean[(df_real_clean > x_region[0]) & (df_real_clean < x_region[1])]
selected_eig_imag = df_imag_clean[(df_imag_clean > y_region[0]) & (df_imag_clean < y_region[1])]

# create two dataframes, one for real parts and one for imaginary parts.
# Each dataframe will have dimensions n_eig x n_cases
real_selected_df=pd.DataFrame()
imag_selected_df=pd.DataFrame()

for i in range(len(selected_eig_imag)): # for each row
    imag_selected=selected_eig_imag.loc[i,selected_eig_imag.columns].dropna() # removes nan values
    ind_imag_selected=imag_selected.index # indices of non-nan values 
    
    real_selected=selected_eig_real.iloc[i][ind_imag_selected].dropna() # out of the values in imag_selected, delete the nan values 
    imag_selected=imag_selected[real_selected.index]
    
    if i==0:
        real_selected_df=pd.DataFrame(real_selected).copy(deep=True).reset_index(drop=True)
        imag_selected_df=pd.DataFrame(imag_selected).copy(deep=True).reset_index(drop=True)
        
    else:
        real_selected_df=real_selected_df.merge(pd.DataFrame(real_selected).reset_index(drop=True), how='outer', left_index=True, right_index=True)
        imag_selected_df=imag_selected_df.merge(pd.DataFrame(imag_selected).reset_index(drop=True), how='outer', left_index=True, right_index=True)

#%% 
imag_selected_df=imag_selected_df.dropna()
real_selected_df=real_selected_df.dropna()

#%%
## If you want to exclude 50 Hz eigenvalues

# real_selected_no50Hz,imag_selected_no50Hz=exclude_50Hz_eig(real_selected_df, imag_selected_df)

# real_selected_df=real_selected_no50Hz.copy(deep=True).dropna()
# imag_selected_df=imag_selected_no50Hz.copy(deep=True).dropna()



#%%

# Convert DataFrame to numpy arrays to flatten
selected_eig_real_flat = real_selected_df.to_numpy().flatten()
selected_eig_imag_flat = imag_selected_df.to_numpy().flatten()

# Combine the cleaned real and imaginary parts
X = np.vstack([selected_eig_real_flat, selected_eig_imag_flat]).T

    
#%%
# # Convert DataFrame to numpy arrays to flatten
# selected_eig_real_flat = selected_eig_real.to_numpy().flatten()
# selected_eig_imag_flat = selected_eig_imag.to_numpy().flatten()

# # Remove NaN values using a valid mask
# valid_mask = ~np.isnan(selected_eig_real_flat) & ~np.isnan(selected_eig_imag_flat)
# selected_eig_real_clean = selected_eig_real_flat[valid_mask]
# selected_eig_imag_clean = selected_eig_imag_flat[valid_mask]

# # Combine the cleaned real and imaginary parts
# X = np.vstack([selected_eig_real_clean, selected_eig_imag_clean]).T

#%% clustering methods

#%% k means cluster
k = 3 # Number of clusters
kmeans = KMeans(n_init=10, n_clusters=k, random_state=42)
kmeans.fit(X)

joblib.dump(kmeans, 'kmeans3clusters.sav')

labels = kmeans.labels_  # Cluster labels for each data point
centers = kmeans.cluster_centers_  # Coordinates of the cluster centers

# fig = plt.figure()
# ax=fig.add_subplot()
# ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
# ax.set_xlabel('Real Axis',fontsize=25)
# ax.set_ylabel('Imaginary Axis',fontsize=25)
# ax.tick_params(labelsize=20)
# ax.set_xlim(x_region)
# ax.set_ylim(y_region)
# ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='x', label='Centers')  # Plot the centers
# ax.set_title('K-Means Clustering')
# ax.set_xlabel('Real Axis')
# ax.set_ylabel('Imaginary Axis')
# ax.legend()
# ax.grid()
# fig.tight_layout()
# plt.show()


#%% DBScan
# epsvals = [0.1]#, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
# for i in epsvals: 
#     dbscan = DBSCAN(eps=i, min_samples=10)
#     labels = dbscan.fit_predict(X)
    
#     # Plot the results
#     fig = plt.figure()
#     ax=fig.add_subplot()
#     ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
#     ax.set_xlabel('Real Axis',fontsize=25)
#     ax.set_ylabel('Imaginary Axis',fontsize=25)
#     ax.tick_params(labelsize=20)
#     # ax.set_xlim(x_region)
#     # ax.set_ylim(y_region)
#     ax.set_title(f'DBSCAN Clustering with eps = {i}')
#     ax.set_xlabel('Real Axis')
#     ax.set_ylabel('Imaginary Axis')
#     ax.legend()
#     ax.grid()
#     fig.tight_layout()
#     plt.show()
# #     fig.savefig(f'Complete_DBSCAN_eps={i}.png')

# # calculate silhouette score
# # Exclude noise points
# cluster_labels = labels[labels != -1]  # Exclude noise
# filtered_X = X[labels != -1]           # Exclude noise points in data
# silhouette = silhouette_score(filtered_X, cluster_labels)
# print(f"Silhouette Score: {silhouette:.2f}")

#%% Optics 
# xivals = [0.01, 0.05, 0.10, 0.15, 0.2]
# for i in xivals: 
#     optics = OPTICS(min_samples=2, xi=i, min_cluster_size=0.1)
#     optics.fit(X)
#     labels = optics.labels_
    
#     fig = plt.figure()
#     ax=fig.add_subplot()
#     ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
#     ax.set_xlabel('Real Axis',fontsize=25)
#     ax.set_ylabel('Imaginary Axis',fontsize=25)
#     ax.tick_params(labelsize=20)
#     ax.set_xlim(x_region)
#     ax.set_ylim(y_region)
#     ax.set_title('Optics Clustering')
#     ax.set_xlabel('Real Axis')
#     ax.set_ylabel('Imaginary Axis')
#     ax.legend()
#     ax.grid()
#     fig.tight_layout()
#     plt.show()
#     fig.savefig(f'Complete_OPTICS_xi={i}.png')

#%% 
labels_reshape=labels.reshape(imag_selected_df.shape)
ClustersMissingEigs, SampleClusterPairs = check_cluster_memberships(labels_reshape, k)

# if there are samples that do not have an eigenvalue in each cluster
if ClustersMissingEigs == 1:
    for i in range(len(SampleClusterPairs)): # for each samplecluster pair         
        center = centers[SampleClusterPairs[i][1]] # cluster centroid 
        
        # values in the sample:
        real = real_selected_df.T.iloc[SampleClusterPairs[i][0],:]
        imag = imag_selected_df.T.iloc[SampleClusterPairs[i][0],:]
        samplevals = pd.concat([real, imag], axis=1).reset_index(drop=True)
        
        min_distance = float('inf')  # Start with a very large number
        min_index = -1  # To store the index of the minimum distance
        
        # for each value in sample 
        for j in range(len(samplevals)):
            # Calculate the Euclidean distance
            distance = np.sqrt(np.sum((samplevals.iloc[j] - center) ** 2))
            
            # Check if this distance is the minimum
            if distance < min_distance:
                min_distance = distance
                min_index = j
        
        # reassign the eigenvalue to that cluster 
        min_index


for l in range(k):
    X_cl=X[labels==l]
    if X_cl[:,0].max()>=0:
        print('Cluster '+str(l)+' collects critical eigenvalues')
        crit_cluster=l

#%% Calculate the damping ratios
labels_reshape_T=labels_reshape.T
unique_labels = np.unique(labels)

# DI[1] has the damping index values for cluster 1, DI[2] has the values for cluster 2, etc. 
drs = {} 
# go through each label 
for i in unique_labels:
    # initialize an empty DI array that is the same size as labels_reshape_T
    dr = np.full_like(labels_reshape_T, np.nan, dtype = float)
    for row_idx, row in enumerate(labels_reshape_T):
        for col_idx, value in enumerate(row):
            if value == i: # if the label is the label we are looking for
                dr[row_idx, col_idx] =  (-real_selected_df.T.iloc[row_idx, col_idx]/np.sqrt(real_selected_df.T.iloc[row_idx, col_idx]**2+imag_selected_df.T.iloc[row_idx, col_idx]**2))
    drs[i] = dr

#%% calculate the damping index 

dr=drs[crit_cluster]
DIs=np.zeros([dr.shape[0],1])
for i in range(dr.shape[0]):
    if np.all(np.isnan(dr[i, :])):
        DIs[i, 0] = np.nan  # Assign NaN if no valid values exist
    else:
        DIs[i,0]=1-min(dr[i,~np.isnan(dr[i,:])])
    












