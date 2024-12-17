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
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
import copy
import joblib

#%% Functions
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
        for l in k:  # For each cluster
            if not any(labels_reshape[:, i] == l): 
                print(f'There is no eigenvalue from sample {i} in cluster {l}')
                ClustersMissingEigs = 1
                SampleClusterPairs.append((i, l))  # Append the (sample, cluster) pair
    
    return ClustersMissingEigs, SampleClusterPairs

def plot_clusters(cluster_method, X, labels, x_region, y_region, model, reassignment = 0, target_cluster_points = 0,target_point=0, centroid = 0 ):

    unique_labels = np.unique(labels)
    fig = plt.figure()
    ax=fig.add_subplot()
    scatter=ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(label)), markersize=10) for label in unique_labels]
    # ax.legend(handles=handles, labels=[str(label) for label in unique_labels], loc='lower left', bbox_to_anchor=(0, 0), fontsize=15, ncol=1)
    if cluster_method == 'KMeans':
        centers = kmeans.cluster_centers_
        ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='x', label='Centers')  
        ax.text(-100, 210, 'n_clusters: ' + str(kmeans.n_clusters))
    elif cluster_method == 'Optics':
        ax.text(-100, 210, 'min_samples: ' + str(optics.min_samples) + '\nmax_eps: ' + str(optics.max_eps) + '\nNumber of clusters: ' + str(len(unique_labels)))
    elif cluster_method == 'DBSCAN':
        ax.text(-100, 210, 'eps: ' + str(dbscan.eps) + ' min_samples: ' + str(dbscan.min_samples)  + '\nNumber of clusters: ' + str(len(unique_labels)))
    elif cluster_method == 'HDBSCAN':
        ax.text(-100, 210, 'min_cluster_size' + str(hdbscan.min_cluser_size) + '\nmin_samples: ' + str(hdbscan.min_samples)  + '\nNumber of clusters: ' + str(len(unique_labels)))
    else: print("wrong")
    if reassignment == 1:
        ax.scatter(target_cluster_points[:, 0], target_cluster_points[:, 1], color='yellow', edgecolor='black', s=100, label='Target Cluster')
        ax.scatter(target_point[0], target_point[1], color='red', edgecolor='black', s=150, label='Reassigned Point')
        ax.scatter(centroid[:, 0], centroid[:, 1], c='red', s=200, marker='x', label='Centroid')
    fig.subplots_adjust(top=0.5, bottom=0.2, left=0.2, right=0.9)  # Adjust these values
    ax.set_xlim(x_region)
    ax.set_ylim(y_region)
    ax.tick_params(labelsize=20)
    ax.set_xlabel('Real Axis',fontsize=25)
    ax.set_ylabel('Imaginary Axis',fontsize=25)
    ax.set_title(cluster_method + ' Clustering')
    ax.grid()
    fig.tight_layout(pad=4.0)
    plt.show()
    
def reassign_noise_points(X, labels):
    noise_points = X[labels == -1]
    if len(noise_points) > 0:
        clf = NearestCentroid()
        clf.fit(X[labels != -1], labels[labels != -1])  # Fit only non-noise points
        noise_labels = clf.predict(noise_points)
        labels[labels == -1] = noise_labels
    return labels

    
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

#%% plot the full modal map with highest values 

# select the eigenvalues that have the highest real parts in each row 
df_real_clean_max = df_real_clean[df_real_clean.columns[0:2]]
df_imag_clean_max = df_imag_clean[df_imag_clean.columns[0:2]]

# fig=plt.figure()
# ax=fig.add_subplot()
# ax.set_xlabel('Real Axis',fontsize=25)
# ax.set_ylabel('Imaginary Axis',fontsize=25)
# ax.tick_params(labelsize=20)
# fig.tight_layout()
# plt.grid()
# ax.scatter(df_real_clean,df_imag_clean, label='Eigenvalues')
# ax.scatter(df_real_clean_max, df_imag_clean_max, label='Max Eigenvalues')
# ax.legend(loc='lower left', fontsize=15, ncol=1)  
# ax.set_title('Modal Map', fontsize=25)
# plt.show()

    
#%% plot the modal map for the region 

x_region = [-115,20]
y_region = [200,320]

# fig=plt.figure()
# ax=fig.add_subplot()
# ax.set_xlabel('Real Axis',fontsize=25)
# ax.set_ylabel('Imaginary Axis',fontsize=25)
# ax.set_xlim(x_region)
# ax.set_ylim(y_region)
# ax.tick_params(labelsize=20)
# fig.tight_layout(pad=4.0)
# plt.grid()
# ax.scatter(df_real_clean,df_imag_clean, label='Eigenvalues')
# ax.scatter(df_real_clean_max, df_imag_clean_max, label='Max Eigenvalues')
# ax.legend(loc='lower left', fontsize=15, ncol=1)  
# ax.set_title('Modal Map', fontsize=25)
# plt.show()

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

# drop all rows which contain any naan values 
imag_selected_df=imag_selected_df.dropna()
real_selected_df=real_selected_df.dropna()

#%% If you want to exclude 50 Hz eigenvalues

# real_selected_no50Hz,imag_selected_no50Hz=exclude_50Hz_eig(real_selected_df, imag_selected_df)
# real_selected_df=real_selected_no50Hz.copy(deep=True).dropna()
# imag_selected_df=imag_selected_no50Hz.copy(deep=True).dropna()

#%% create X 

# Convert DataFrame to numpy arrays to flatten
selected_eig_real_flat = real_selected_df.to_numpy().flatten()
selected_eig_imag_flat = imag_selected_df.to_numpy().flatten()

# Combine the cleaned real and imaginary parts
X = np.vstack([selected_eig_real_flat, selected_eig_imag_flat]).T


#%% K Means cluster
cluster_method = 'KMeans'

# test for different kvals (number of clusters)
n_clusters = [2,3,4,5,6]
K_Means_Results = np.column_stack((n_clusters, np.zeros(len(n_clusters))))
for idx, i in enumerate(n_clusters):
    kmeans = KMeans(n_init=10, n_clusters=i, random_state=42)
    kmeans.fit(X)  
    labels = kmeans.labels_  # Cluster labels for each data point
    K_Means_Results[idx,1] = silhouette_score(X, labels)
    
# the best silhouette score is when n_clusters = 3 
kmeans = KMeans(n_init=10, n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_  # Cluster labels for each data point
plot_clusters(cluster_method, X, labels, x_region, y_region, kmeans)


joblib.dump(kmeans, 'kmeans3clusters.sav')

#%% Optics 
cluster_method = 'Optics'

# test for different min_samples 
min_samples = [2200]
Optics_results = np.zeros((len(min_samples), 2))
counter = 0
for i in min_samples: 
    optics = OPTICS(min_samples=i)
    optics.fit(X)
    labels = optics.labels_
    labels = reassign_noise_points(X, labels)
    
    Optics_results[counter, 0] = i  # min samples value 
    Optics_results[counter, 1] = silhouette_score(X, labels)  # silhouette score
    
    print(f"min_samples: {i},  Silhouette Score: {Optics_results[counter, 1]}")
    
    print(cluster_method)
    unique_labels = np.unique(labels)
    labels_reshape=labels.reshape(imag_selected_df.shape)
    ClustersMissingEigs, SampleClusterPairs = check_cluster_memberships(labels_reshape, unique_labels)
    
    plot_clusters(cluster_method, X, labels, x_region, y_region, optics)
    counter += 1
    
# # the best silhouette score occurs when min_samples is approximately 2300 
optics = OPTICS(min_samples=2200)
optics.fit(X)
labels = optics.labels_
plot_clusters(cluster_method, X, labels, x_region, y_region, optics)

#%% DBSCAN
cluster_method = 'DBSCAN'

# k distance graph 
# min_samples = 20  # Adjust this value to see the effect on the elbow
# nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X)
# distances, indices = nbrs.kneighbors(X)
# distances = np.sort(distances[:, -1])  # Distances to the `min_samples-th` nearest neighbor
# plt.plot(distances)
# plt.xlabel("Points sorted by distance")
# plt.ylabel(f"{min_samples}-th Nearest Neighbor Distance")
# plt.title("k-Distance Plot")
# plt.show()

dbscan = DBSCAN()
eps = [0.5, 1, 1.5] #chosen because they are elbow of kdistance graph 
min_samples = [50,60,70] 
DBSCAN_results = np.zeros((len(eps) * len(min_samples), 4))

# Manually loop over the hyperparameter combinations
counter = 0
for i in eps:
    for j in min_samples:
        dbscan = DBSCAN(eps=i, min_samples=j)
        labels = dbscan.fit_predict(X)
        labels = reassign_noise_points(X, labels)

        DBSCAN_results[counter, 0] = i  # eps value
        DBSCAN_results[counter, 1] = j  # min_samples value
        DBSCAN_results[counter, 2] = silhouette_score(X, labels)  # silhouette score
        DBSCAN_results[counter, 3] = len(np.unique(labels))
        
        print(cluster_method)
        unique_labels = np.unique(labels)
        labels_reshape=labels.reshape(imag_selected_df.shape)
        
        if len(unique_labels) < 15:
            ClustersMissingEigs, SampleClusterPairs = check_cluster_memberships(labels_reshape, unique_labels)
        
        print(f"eps: {i}, min_samples: {j}, Silhouette Score: {DBSCAN_results[counter, 2]}, Number of Clusters: {len(unique_labels)}")
        
        counter += 1

# the best results are when eps = and min_samples= 
dbscan = DBSCAN(eps=0.5, min_samples=40)
dbscan.fit(X)
labels = dbscan.labels_
plot_clusters(cluster_method, X, labels, x_region, y_region, dbscan)


#%% HDBSCAN 
# cluster_method = 'HDBSCAN'

# # test for different values of min_cluster_sizevals and min_samplesvals
# min_cluster_size = [50, 100, 150]
# min_samples = [30, 50, 100]
# HDBSCAN_results = np.zeros((len(min_cluster_size) * len(min_samples), 3))

# counter = 0
# for i in min_cluster_size:
#     for j in min_samples:
#         hdbscan = HDBSCAN(min_cluster_size = i, min_samples = j)
#         hdbscan.fit(X)
#         labels = hdbscan.labels_  
#         labels = reassign_noise_points(X, labels)
        
#         # Store the eps, min_samples, and silhouette score
#         HDBSCAN_results[counter, 0] = i  
#         HDBSCAN_results[counter, 1] = j  
#         HDBSCAN_results[counter, 2] = silhouette_score(X, labels)  # silhouette score
        
        # print(f"min_cluster_size: {i}, min_samples: {j}, Silhouette Score: {HDBSCAN_results[counter, 2]}")
        
#         counter += 1
        
#         print(cluster_method)
#         unique_labels = np.unique(labels)
#         labels_reshape=labels.reshape(imag_selected_df.shape)
#         ClustersMissingEigs, SampleClusterPairs = check_cluster_memberships(labels_reshape, unique_labels)

       
# # the best results are when min_cluster_size = and when min_samples = 
# hdbscan = HDBSCAN(min_cluser_size = , min_samples = )
# hdbscan.fit(X)
# labels = hdbscan.labels_ 
# plot_clusters(cluster_method, X, labels, x_region, y_region, hdbscan)

#%% check to see if there are samples that do not have an eigenvalue in each cluster
unique_labels = np.unique(labels)
labels_reshape=labels.reshape(imag_selected_df.shape)
ClustersMissingEigs, SampleClusterPairs = check_cluster_memberships(labels_reshape, unique_labels)

# # if there are samples that do not have an eigenvalue in each cluster
if ClustersMissingEigs == 1:
    for i in range(len(SampleClusterPairs)): # for each samplecluster pair 
        
        # calculate the centroid of the cluster 
        centroid = np.array([X[labels == SampleClusterPairs[i][1]].mean(axis=0)])
        
        # values in the sample:
        real = real_selected_df.T.iloc[SampleClusterPairs[i][0],:]
        imag = imag_selected_df.T.iloc[SampleClusterPairs[i][0],:]
        samplevals = pd.concat([real, imag], axis=1).reset_index(drop=True)
        
        min_distance = float('inf')  # Start with a very large number
        min_index = -1  # To store the index of the minimum distance
        
        # for each value in sample 
        for j in range(len(samplevals)):
            # Calculate the Euclidean distance between the center and the sample 
            distance = np.sqrt(np.sum((samplevals.iloc[j].values - centroid) ** 2))
            
            # Check if this distance is the minimum
            if distance < min_distance:
                min_distance = distance
                min_index = j
                
        # plot the clustering region and highlight the target cluster and the target point 
        target_cluster_points = X[labels == SampleClusterPairs[i][1]]
        target_point = samplevals.iloc[min_index].values
        plot_clusters(cluster_method, X, labels, x_region, y_region, optics, 1, target_cluster_points, target_point,  centroid)
        
        # reassign the eigenvalue to that cluster 
        labels_reshape[min_index,SampleClusterPairs[i][0]] = SampleClusterPairs[i][1]

for l in range(unique_labels):
    X_cl=X[labels==l]
    if X_cl[:,0].max()>=0:
        print('Cluster '+str(l)+' collects critical eigenvalues')
        crit_cluster=l
        
        
#%% Calculate the damping ratios
labels_reshape_T=labels_reshape.T

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
    
    
#%% create the csv with the inputs and add a column with the DIs  

# may want to remove empty rows
inputs=pd.read_csv('../results/datagen_ACOPF_LF09_seed17_nc5_ns5_d5_20241119_115327_8464/case_df_op.csv')
inputs['DI_crit'] = DIs
pd.DataFrame.to_csv(inputs, 'Inputs_DI_Crit.csv', index=False)












