import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

file_path = r'../results/ACOPF_standalone_seed16_nc3_ns100_20241014_162634_9593/case_df_op.csv'
data = pd.read_csv(file_path)

# understand data
print("The data has", data.shape[0], "rows and", data.shape[1], "columns")
# print(data.head()) # Display the first few rows of the DataFrame
# print(data.dtypes) # data types
print(data.isna().sum()) # see if there is data missing

# get rid of missing data
data = data.dropna()

#PCA with percentage of variance preserved
pca = PCA(n_components=0.95) # to preserve 95% of the dimensionality
PCAPercent = pca.fit_transform(data)

# plot explained variance as a function of dimensions
pca = PCA()
pca.fit(data)
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.title('Cumulative Explained Variance by Number of Dimensions')
plt.xlabel('Number of Dimensions')
plt.ylabel('Cumulative Explained Variance')
plt.xlim(1, len(cumulative_variance))
plt.ylim(0, 1)
plt.grid()
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance Threshold')  # Optional threshold line
plt.legend()
plt.show()

print("Explained Variance:", explained_variance)
print("Cumulative Explained Variance:", cumulative_variance)
n_components = np.argmax(cumulative_variance >= 0.95) + 1  # Number of components to retain for 95% variance
print("Number of components to retain for 95%:", n_components)

#reverse the transformation and get the origional data
#data_recovered= pca.inverse_transform(PCAPercent)

# Calculation of correlation coefficients
corr = data.iloc[:,1:].corr(method='spearman')  # method{‘pearson’, ‘kendall’, ‘spearman’}
print(corr.isna().sum().sum())  # Check for NaN values in the correlation matrix
corr = corr.fillna(0)
print(corr.isna().sum().sum())