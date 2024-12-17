"""
Calculates the damping index 
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import HDBSCAN
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
import copy
import joblib

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


#%% import X, real_selected_df, imag_selected_df

real_selected_df = pd.read_csv('../identification_of_critical_eigenvalues/real_selected_df.csv')
imag_selected_df = pd.read_csv('../identification_of_critical_eigenvalues/imag_selected_df.csv')
X= np.loadtxt('../identification_of_critical_eigenvalues/X.csv', delimiter=',')


#%% find the best model and run it again  

Results = pd.read_csv('../identification_of_critical_eigenvalues/clustering_selection_results.csv')

# Get the row with the highest silhouette score
highest_silhouette_row = Results.loc[Results['silhouette_score'].idxmax()]
print("The best model is:\n",  highest_silhouette_row)

best_method = highest_silhouette_row.iloc[0]
best_parameter1 = highest_silhouette_row.iloc[1]
best_parameter2 = highest_silhouette_row.iloc[2] 

if best_method == "KMeans":
    kmeans = KMeans(n_init=10, n_clusters=int(best_parameter1), random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_  
elif best_method == "Optics":
    optics = OPTICS(min_samples=int(best_parameter1))
    optics.fit(X)
    labels = optics.labels_
elif best_method == "DBSCAN":
    dbscan = DBSCAN(eps=best_parameter1, min_samples=int(best_parameter2))
    dbscan.fit(X)
    labels = dbscan.labels_
    
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
        
        # reassign the eigenvalue to that cluster 
        labels_reshape[min_index,SampleClusterPairs[i][0]] = SampleClusterPairs[i][1]
    
#%% Calculate the damping ratios
labels_reshape=labels.reshape(imag_selected_df.shape)
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
       
# identify critical eigenvalues 
for l in unique_labels:
    X_cl=X[labels==l]
    if X_cl[:,0].max()>=0:
        print('Cluster '+str(l)+' collects critical eigenvalues')
        crit_cluster=l

#%% calculate the damping index 

dr=drs[crit_cluster]
DIs=np.zeros([dr.shape[0],1])
for i in range(dr.shape[0]):
    if np.all(np.isnan(dr[i, :])):
        DIs[i, 0] = np.nan  # Assign NaN if no valid values exist
    else:
        DIs[i,0]=1-min(dr[i,~np.isnan(dr[i,:])])
    
    
#%% create the csv with the inputs and add a column with the DIs  

case_op=pd.read_csv('../results/datagen_ACOPF_LF09_seed17_nc5_ns5_d5_20241119_115327_8464/case_df_op.csv')
# remove the rows with na in column V1 (the cases that didn't converge)
case_op = case_op.dropna(subset=['V1'])
case_op['DI_crit'] = DIs
pd.DataFrame.to_csv(case_op, 'Training_Inputs_DI_Crit.csv', index=False)


