"""
This file plots the modal map of the eigenvalues on the real-imaginary axis
It also identifies critical eigenvalues through different clustering methods
"""

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

#%% import and clean data
df_real=pd.read_csv('../results/datagen_ACOPF_LF09_seed17_nc5_ns5_d5_20241119_115327_8464/case_df_real.csv')
df_imag=pd.read_csv('../results/datagen_ACOPF_LF09_seed17_nc5_ns5_d5_20241119_115327_8464/case_df_imag.csv')

# remove the empty rows (from the samples that didn't converge)
df_real_clean = df_real.dropna(subset=['2'])
df_imag_clean = df_imag.dropna(subset=['2'])

# remove the first column and the  case_id column
df_real_clean = df_real_clean.drop([df_real_clean.columns[0], 'case_id', 'Stability'], axis=1)
df_imag_clean = df_imag_clean.drop([df_imag_clean.columns[0], 'case_id', 'Stability'], axis=1)

n_cases_clean= len(df_real_clean)


# select the eigenvalues that have the highest real parts in each row 
df_real_clean_max = df_real_clean[df_real_clean.columns[0:2]]
df_imag_clean_max = df_imag_clean[df_imag_clean.columns[0:2]]


#%% plot the modal map 

# fig=plt.figure()
# ax=fig.add_subplot()
# ax.set_xlabel('Real Axis',fontsize=25)
# ax.set_ylabel('Imaginary Axis',fontsize=25)
# # ax.set_xlim([-80,20])
# # ax.set_ylim([200,320])
# ax.tick_params(labelsize=20)
# fig.tight_layout()
# plt.grid()
# ax.scatter(df_real_clean,df_imag_clean, label='Eigenvalues')
# ax.scatter(df_real_clean_max, df_imag_clean_max, label='Max Eigenvalues')
# ax.legend(loc='lower center',bbox_to_anchor=(0.45, -0.65),fontsize=15, ncol=2)
# ax.set_title('Modal Map', fontsize=25)
# plt.show()

#%% clean the data before clustering

complete = 1;
if complete==1:
    x_region = [-80,20]
    y_region = [200,320]
else:
    x_region = [-1,20]
    y_region = [-400,400]

# select eigenvalues in the region
selected_eig_real = df_real_clean[(df_real_clean > x_region[0]) & (df_real_clean < x_region[1])]
selected_eig_imag = df_imag_clean[(df_imag_clean > y_region[0]) & (df_imag_clean < y_region[1])]

# Convert DataFrame to numpy arrays to flatten
selected_eig_real_flat = selected_eig_real.to_numpy().flatten()
selected_eig_imag_flat = selected_eig_imag.to_numpy().flatten()

# Remove NaN values using a valid mask
valid_mask = ~np.isnan(selected_eig_real_flat) & ~np.isnan(selected_eig_imag_flat)
selected_eig_real_clean = selected_eig_real_flat[valid_mask]
selected_eig_imag_clean = selected_eig_imag_flat[valid_mask]

# Combine the cleaned real and imaginary parts
X = np.vstack([selected_eig_real_clean, selected_eig_imag_clean]).T

# scale the data 
# X_scaled = StandardScaler().fit_transform(X)

#%% clustering methods

#%% k means cluster
# k = 3 # Number of clusters
# kmeans = KMeans(n_init=10, n_clusters=k, random_state=42)
# kmeans.fit(X)

# labels = kmeans.labels_  # Cluster labels for each data point
# centers = kmeans.cluster_centers_  # Coordinates of the cluster centers

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


#%% SpectralClustering model
## note: this method doesn't work with the complete map because there is too much data 
# n_clusters = 2  # Number of clusters you want to identify
# sc = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
# # Fit the model and predict the cluster labels
# labels = sc.fit_predict(X)
# # Visualize the clusters
# fig = plt.figure()
# ax=fig.add_subplot()
# ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
# ax.set_xlabel('Real Axis',fontsize=25)
# ax.set_ylabel('Imaginary Axis',fontsize=25)
# ax.tick_params(labelsize=20)
# ax.set_xlim(x_region)
# ax.set_ylim(y_region)
# ax.set_title('Spectral Clustering')
# ax.set_xlabel('Real Axis')
# ax.set_ylabel('Imaginary Axis')
# ax.legend()
# ax.grid()
# fig.tight_layout()
# plt.show()

# # calculate silhouette score
# silhouette = silhouette_score(X, labels)
# print(f"Silhouette Score: {silhouette:.2f}")

#%% DBScan
# epsvals = [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
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
#     fig.savefig(f'Complete_DBSCAN_eps={i}.png')

# # calculate silhouette score
# # Exclude noise points
# cluster_labels = labels[labels != -1]  # Exclude noise
# filtered_X = X[labels != -1]           # Exclude noise points in data
# silhouette = silhouette_score(filtered_X, cluster_labels)
# print(f"Silhouette Score: {silhouette:.2f}")

#%% Optics 
xivals = [0.01, 0.05, 0.10, 0.15, 0.2]
for i in xivals: 
    optics = OPTICS(min_samples=2, xi=i, min_cluster_size=0.1)
    optics.fit(X)
    labels = optics.labels_
    
    fig = plt.figure()
    ax=fig.add_subplot()
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
    ax.set_xlabel('Real Axis',fontsize=25)
    ax.set_ylabel('Imaginary Axis',fontsize=25)
    ax.tick_params(labelsize=20)
    ax.set_xlim(x_region)
    ax.set_ylim(y_region)
    ax.set_title('Optics Clustering')
    ax.set_xlabel('Real Axis')
    ax.set_ylabel('Imaginary Axis')
    ax.legend()
    ax.grid()
    fig.tight_layout()
    plt.show()
    fig.savefig(f'Complete_OPTICS_xi={i}.png')


#%% Calculate the damping index

# separate the groups 
# I have X with two columns and I have labels with the label 
# I want to split the X values into an array for each label 
# or I can keep them together and have the damping index do a loop and calcualte it for each label 


# damping index of group 1
#DI_crit_eig=1-(-crit_eig_real[:,0]/np.sqrt(crit_eig_real[:,0]**2+crit_eig_imag[:,0]**2))

# this equation calculates the DI of all of the eigenvalues (not in groups)
DI_tot = 1-(-X[:,0]/np.sqrt(X[:,0]**2+X[:,1]**2))
DI_tot_min = min(DI_tot)
DI_tot_max = max(DI_tot)

# calculate the damping index for each group 
DI = {}
DI_min = {}
DI_max = {}
unique_labels = np.unique(labels)
for i in unique_labels:
    X_i = X[labels == i]
    DI_i =1-(-X_i[:,0]/np.sqrt(X_i[:,0]**2+X_i[:,1]**2))
    DI[i] = DI_i  # Store the result in a dictionary
    DI_min[i] = min(DI_i)
    DI_max[i] = max(DI_i)











