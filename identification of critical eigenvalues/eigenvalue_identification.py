"""
This file plots the modal map of the eigenvalues on the real-imaginary axis
It also identifies critical eigenvalues through clustering
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN


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

#%% plot the full modal map
# fig=plt.figure()
# ax=fig.add_subplot()
# ax.scatter(df_real_clean,df_imag_clean, label='Eigenvalues')
# ax.set_xlabel('Real Axis',fontsize=25)
# ax.set_ylabel('Imaginary Axis',fontsize=25)
# ax.set_title('Modal Map', fontsize=25)
# ax.tick_params(labelsize=20)
# ax.legend(loc='lower center',bbox_to_anchor=(0.45, -0.65),fontsize=15, ncol=2)
# fig.tight_layout()
# plt.grid()
# fig.savefig('Modal_Map.png')

# zoom in
# fig = plt.figure(figsize=(10, 15))
# ax=fig.add_subplot()
# ax.scatter(df_real_clean,df_imag_clean, label='Eigenvalues')
# ax.set_xlabel('Real Axis',fontsize=25)
# ax.set_ylabel('Imaginary Axis',fontsize=25)
# ax.set_title('Zoom', fontsize=25)
# ax.tick_params(labelsize=20)
# ax.set_xlim([-100,20])
# ax.set_ylim([-400,400])
# ax.legend(loc='lower center',bbox_to_anchor=(0.45, -0.65),fontsize=15, ncol=2)
# fig.tight_layout()
# plt.grid()
# fig.savefig('ClusteringRegion.png')

#%% Identify clustering region
x_region = [-60,20]
y_region = [150,400]

# select eigenvalues in that region
selected_eig_real = df_real_clean[(df_real_clean > x_region[0]) & (df_real_clean < x_region[1])]
selected_eig_imag = df_imag_clean[(df_imag_clean > y_region[0]) & (df_imag_clean < y_region[1])]

#%% clean the data before clustering

# Convert DataFrame to numpy arrays before flattening
selected_eig_real_flat = selected_eig_real.to_numpy().flatten()
selected_eig_imag_flat = selected_eig_imag.to_numpy().flatten()

# Remove NaN values using a valid mask
valid_mask = ~np.isnan(selected_eig_real_flat) & ~np.isnan(selected_eig_imag_flat)
selected_eig_real_clean = selected_eig_real_flat[valid_mask]
selected_eig_imag_clean = selected_eig_imag_flat[valid_mask]

# Combine the cleaned real and imaginary parts
X = np.vstack([selected_eig_real_clean, selected_eig_imag_clean]).T

# selected region
fig = plt.figure(figsize=(10, 15))
ax=fig.add_subplot()
ax.scatter(selected_eig_real_clean,selected_eig_imag_clean, label='Eigenvalues')
ax.set_xlabel('Real Axis',fontsize=25)
ax.set_ylabel('Imaginary Axis',fontsize=25)
ax.set_title('Zoom', fontsize=25)
ax.tick_params(labelsize=20)
ax.set_xlim(x_region)
ax.set_ylim(y_region)
ax.legend(loc='lower center',bbox_to_anchor=(0.45, -0.65),fontsize=15, ncol=2)
fig.tight_layout()
plt.grid()
fig.savefig('ClusteringRegion.png')

# Combine the cleaned real and imaginary parts
X = np.vstack([selected_eig_real_clean, selected_eig_imag_clean]).T

#plot the selected region
plt.scatter(X[:, 0], X[:, 1])
plt.title("Before")
plt.show()
print(X.shape)


#%% run clustering
#%% k means cluster
# k = 3  # Number of clusters
# kmeans = KMeans(n_clusters=k, random_state=42)
# kmeans.fit(X)
#
# labels = kmeans.labels_  # Cluster labels for each data point
# centers = kmeans.cluster_centers_  # Coordinates of the cluster centers
#
# plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')  # Scatter plot of data points
# plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='x', label='Centers')  # Plot the centers
# plt.title('K-Means Clustering')
# plt.xlabel('Real Axis')
# plt.ylabel('Imaginary Axis')
# plt.legend()
# plt.show()

# #%% Create the SpectralClustering model
# n_clusters = 3  # Number of clusters you want to identify
# sc = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
# # Fit the model and predict the cluster labels
# labels = sc.fit_predict(X)
# # Visualize the clusters
# plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
# plt.title("Spectral Clustering Results")
# plt.xlabel("Real Axis")
# plt.ylabel("Imaginary Axis")
# plt.show()

#%% DBScan
dbscan = DBSCAN(eps=3, min_samples=10)
labels = dbscan.fit_predict(X)
# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
plt.title("DBSCAN Clustering")
plt.xlabel("Real Axis")
plt.ylabel("Imaginary Axis")
plt.show()



#%% Calculate the damping index

#DI_crit_eig=1-(-crit_eig_real[:,0]/np.sqrt(crit_eig_real[:,0]**2+crit_eig_imag[:,0]**2))