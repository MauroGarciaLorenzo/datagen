from matplotlib import offsetbox
from collections import defaultdict
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
import os
from utils_pp_standalone import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA, KernelPCA
import seaborn as sns
from scipy.stats import pointbiserialr

# %%

plt.rcParams.update({"figure.figsize": [8, 4],
                     "text.usetex": True,
                     "font.family": "serif",
                     "font.serif": "Computer Modern",
                     "axes.labelsize": 20,
                     "axes.titlesize": 20,
                     'figure.titlesize': 20,
                     "legend.fontsize": 20,
                     "xtick.labelsize": 16,
                     "ytick.labelsize": 16,
                     "savefig.dpi": 130,
                    'legend.fontsize': 20,
                     'legend.handlelength': 2,
                     'legend.loc': 'upper right'})

# %%
path = '../results/'

dir_names=[dir_name for dir_name in os.listdir(path)]# if dir_name.startswith('datagen') and 'zip' not in dir_name]#

# dir_names = [
#     #'datagen_ACOPF_slurm23172357_cu10_nodes32_LF09_seed3_nc3_ns500_d7_20250627_214226_7664']
#     'datagen_ACOPF_slurm25105245_cu8_nodes32_LF09_seed3_nc3_ns500_d7_20250731_132256_7665']

for dir_name in dir_names[1:]:
    path_results = os.path.join(path, dir_name)

    results_dataframes, csv_files = open_csv(
        path_results, ['cases_df.csv', 'case_df_op.csv'])

    perc_stability(results_dataframes['case_df_op'], dir_name)
    
    dataset_ID = dir_name[-5:]

# %% ---- FILL NAN VALUES WITH NULL ---

results_dataframes['case_df_op'] = results_dataframes['case_df_op'].fillna(0)

# %% ---- FIX VALUES ----

Sn_cols = [col for col in results_dataframes['case_df_op']
           if col.startswith('Sn')]
results_dataframes['case_df_op'][Sn_cols] = results_dataframes['case_df_op'][Sn_cols]/100

theta_cols = [col for col in results_dataframes['case_df_op']
              if col.startswith('theta')]
# Adjust angles greater than 180°
results_dataframes['case_df_op'][theta_cols] = results_dataframes['case_df_op'][theta_cols] - \
    (results_dataframes['case_df_op'][theta_cols] > 180) * 360

results_dataframes['case_df_op'][theta_cols] = results_dataframes['case_df_op'][theta_cols] * np.pi/180

# add total demand variables
PL_cols = [
    col for col in results_dataframes['case_df_op'].columns if col.startswith('PL')]
results_dataframes['case_df_op']['PD'] = results_dataframes['case_df_op'][PL_cols].sum(
    axis=1)

QL_cols = [
    col for col in results_dataframes['case_df_op'].columns if col.startswith('QL')]
results_dataframes['case_df_op']['QD'] = results_dataframes['case_df_op'][QL_cols].sum(
    axis=1)

# %% ---- SELECT ONLY FEASIBLE CASES ----

results_dataframes['case_df_op_feasible'] = results_dataframes['case_df_op'].query(
    'Stability >= 0')

case_id_feasible = list(results_dataframes['case_df_op_feasible']['case_id'])

print(len(case_id_feasible))

print(len(set(case_id_feasible)))

results_dataframes['case_df_op_feasible'].groupby('case_id')['case_id'].count()

# case_id=case_id_feasible[0]
# results_dataframes['case_df_op_feasible'].query('case_id == @case_id')['P_SG12'] <--- quantities calculated by power flow
# results_dataframes['cases_df'].query('case_id == @case_id')['p_sg_Var10'] <-- quantities sampled

results_dataframes['cases_df_feasible'] = results_dataframes['cases_df'].query(
    'case_id == @case_id_feasible')  # <-- quantities sampled

print(len(results_dataframes['cases_df_feasible']['case_id']))

n_feas_cases = len(case_id_feasible)

results_dataframes['case_df_op_feasible_X'] = results_dataframes['case_df_op_feasible'].drop([
                                                                                             'case_id', 'Stability'], axis=1)

# %% ---- SELECT ONLY UNFEASIBLE CASES ----

results_dataframes['case_df_op_unfeasible'] = results_dataframes['case_df_op'].query(
    'Stability < 0')

# %%
columns_in_df = dict()
for key, item in results_dataframes.items():
    print(key)
    columns_in_df[key] = results_dataframes[key].columns

# %% ----  Remove columns with only 1 value ----
columns_with_single_values = []
for c in columns_in_df['case_df_op_feasible']:
    if results_dataframes['case_df_op_feasible'][c].unique().size == 1:
        columns_with_single_values.append(c)
# --> if there is something different from Sn_SGX check, otherwise it is normal (no changes in SG installed power)
print(columns_with_single_values)

results_dataframes['case_df_op_feasible'] = results_dataframes['case_df_op_feasible'].drop(
    columns_with_single_values, axis=1)
results_dataframes['case_df_op_feasible_X'] = results_dataframes['case_df_op_feasible_X'].drop(
    columns_with_single_values, axis=1)

# %% ---- Check correlated variables Option #1 ----
def get_correlated_columns(df, c_threshold=0.95, method='pearson'):

    correlated_features_tuples = []
    correlated_features = pd.DataFrame(columns=['Feat1', 'Feat2', 'Corr'])
    correlation = df.corr(method=method)
    count = 0
    for i in correlation.index:
        for j in correlation:
            if i != j and abs(correlation.loc[i, j]) >= c_threshold:
                # if tuple([j,i]) not in correlated_features_tuples:
                correlated_features_tuples.append(tuple([i, j]))
                correlated_features.loc[count, 'Feat1'] = i
                correlated_features.loc[count, 'Feat2'] = j
                correlated_features.loc[count, 'Corr'] = correlation.loc[i, j]
                count = count+1

    return correlated_features


correlated_features = get_correlated_columns(
    results_dataframes['case_df_op_feasible_X'])

grouped_corr_feat = correlated_features.groupby('Feat1').count().reset_index()

keep_var=[]
while not grouped_corr_feat.empty:
    # Pick the first remaining Feat1
    var = grouped_corr_feat.iloc[0]['Feat1']
    keep_var.append(var)

    # Find all features correlated with this one
    to_remove = correlated_features.query('Feat1 == @var')['Feat2'].tolist()

    # Drop all rows where Feat1 is in to_remove
    grouped_corr_feat = grouped_corr_feat[~grouped_corr_feat['Feat1'].isin(to_remove)]
    grouped_corr_feat = grouped_corr_feat[~grouped_corr_feat['Feat1'].isin(keep_var)]

df_taus = results_dataframes['case_df_op_feasible'][['case_id']].merge(results_dataframes['cases_df_feasible'][[
                                                                       col for col in columns_in_df['cases_df_feasible'] if col.startswith('tau_droop')]+['case_id']], on='case_id', how='left').drop(['case_id'], axis=1)

results_dataframes['case_df_op_feasible_uncorr_X'] = pd.concat([results_dataframes['case_df_op_feasible_X'][keep_var].reset_index(drop=True), df_taus],axis=1)
results_dataframes['case_df_op_feasible_uncorr'] = results_dataframes['case_df_op_feasible_uncorr_X']
results_dataframes['case_df_op_feasible_uncorr']['case_id'] = results_dataframes['case_df_op_feasible']['case_id'].reset_index(drop=True)
results_dataframes['case_df_op_feasible_uncorr']['Stability'] = results_dataframes['case_df_op_feasible']['Stability'].reset_index(drop=True)

results_dataframes['case_df_op_feasible_uncorr'].to_csv('DataSet_training_uncorr_var'+dataset_ID+'.csv')

# %% ---- Check correlated variables Option #2 ----

results = pd.concat([results_dataframes['case_df_op_feasible_X'].reset_index(drop=True), df_taus.reset_index(drop=True)], axis=1).apply(
#results = results_dataframes['case_df_op_feasible_X'].reset_index(drop=True).apply(
     lambda col: pointbiserialr(col, results_dataframes['case_df_op_feasible']['Stability']), result_type='expand').T
results.columns = ['correlation', 'p_value']
results['abs_corr'] = abs(results['correlation'])

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
X = pd.concat([results_dataframes['case_df_op_feasible_X'].reset_index(
    drop=True), df_taus.reset_index(drop=True)], axis=1)
# X = results_dataframes['case_df_op_feasible_X'].reset_index(
#     drop=True)
corr = spearmanr(X).correlation

# Ensure the correlation matrix is symmetric
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)

# We convert the correlation matrix to a distance matrix before performing
# hierarchical clustering using Ward's linkage.
distance_matrix = 1 - np.abs(corr)
dist_linkage = hierarchy.ward(squareform(distance_matrix))
# dendro = hierarchy.dendrogram(
#     dist_linkage, labels=X.columns.to_list(), ax=ax1, leaf_rotation=90
# )
# dendro_idx = np.arange(0, len(dendro["ivl"]))

# ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
# ax2.set_xticks(dendro_idx)
# ax2.set_yticks(dendro_idx)
# ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
# ax2.set_yticklabels(dendro["ivl"])
# _ = fig.tight_layout()


cluster_ids = hierarchy.fcluster(dist_linkage, 0.01, criterion="distance")
cluster_id_to_feature_ids = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)

selected_features_names_dict = defaultdict(list)
for i, selected_features in cluster_id_to_feature_ids.items():
    selected_features_names_dict[i]=X.columns[selected_features]

keep_var = []
for i, selected_features in selected_features_names_dict.items():
    if len(selected_features)==1:
        keep_var.append(selected_features[0])
    elif len(selected_features)>1:
         keep_var.append(results.loc[selected_features,'abs_corr'].sort_values(ascending=False).index[0])

# results_dataframes['case_df_op_feasible_uncorr_X'] = results_dataframes['case_df_op_feasible_X'][keep_var]
# results_dataframes['case_df_op_feasible_uncorr'] = results_dataframes['case_df_op_feasible'][keep_var+['case_id', 'Stability']]

results_dataframes['case_df_op_feasible_uncorr_HierCl_X'] = X[keep_var]
results_dataframes['case_df_op_feasible_uncorr_HierCl'] = pd.concat([X[keep_var], results_dataframes['case_df_op_feasible'][['case_id', 'Stability']].reset_index(drop=True)],axis=1)

results_dataframes['case_df_op_feasible_uncorr_HierCl'].to_csv('DataSet_training_uncorr_var_HierCl'+dataset_ID+'.csv')

# %%
columns_in_df = dict()
for key, item in results_dataframes.items():
    print(key)
    columns_in_df[key] = results_dataframes[key].columns

# %%
prefixes_and_sources = [
    ('P_SG', 'case_df_op_feasible_uncorr'),
    ('Q_SG', 'case_df_op_feasible_uncorr'),
    ('P_GFOR', 'case_df_op_feasible_uncorr'),
    ('Q_GFOR', 'case_df_op_feasible_uncorr'),
    ('Sn_GFOR', 'case_df_op_feasible_uncorr'),
    ('P_GFOL', 'case_df_op_feasible_uncorr'),
    ('Q_GFOL', 'case_df_op_feasible_uncorr'),
    ('Sn_GFOL', 'case_df_op_feasible_uncorr'),
    ('PL', 'case_df_op_feasible_uncorr'),
    ('QL', 'case_df_op_feasible_uncorr'),
    ('V', 'case_df_op_feasible_uncorr'),
    ('theta', 'case_df_op_feasible_uncorr'),
]


def group_variables(prefixes_and_sources):
    dict_var = dict()
    single_buses = []
    for prefix, source_df_key in prefixes_and_sources:

        column_names = [
            var for var in columns_in_df[source_df_key] if var.startswith(prefix)]
        single_buses.extend([int(i.split(prefix)[1]) for i in column_names])
        dict_var[prefix] = column_names

    return dict_var, single_buses


dict_var, single_buses = group_variables(prefixes_and_sources)

single_buses = np.unique(single_buses)
dict_var = {k: v for k, v in dict_var.items() if v}


# %% ----- build results matrices -----

def create_processed_blocks_Notaus(dict_va, single_buses, columns_in_df, results_dataframes):
    prefixes_and_sources = []

    for key, _ in dict_var.items():
        prefixes_and_sources.append((key, 'case_df_op_feasible'))

    processed_blocks = {}

    # Process each prefix
    for prefix, source_df_key in prefixes_and_sources:
        column_names = [prefix+str(bus) for bus in single_buses]
        df_partial = pd.DataFrame(
            columns=column_names, index=results_dataframes[source_df_key].index)
        for col in column_names:
            if col in columns_in_df[source_df_key]:
                df_partial[col] = results_dataframes[source_df_key][[col]]
            else:
                df_partial[col] = 0
        processed_blocks[prefix] = df_partial

    # for Sn in ['Sn_GFOL', 'Sn_GFOR']:
    #     if Sn in processed_blocks.keys():
    #         processed_blocks[Sn]=processed_blocks[Sn]/100

    # if 'theta' in processed_blocks.keys():
    #     # Adjust angles greater than 180°
    #     processed_blocks['theta'] = processed_blocks['theta'] - (processed_blocks['theta'] > 180) * 360

    #     processed_blocks['theta']=processed_blocks['theta']*np.pi/180

    return processed_blocks


def create_processed_blocks_with_taus(dict_var, results_dataframes, columns_in_df, n_buses):
    prefixes_and_sources = []

    for key, _ in dict_var.items():
        prefixes_and_sources.append((key, 'case_df_op_feasible'))

    prefixes_and_sources.extend([
        ('tau_droop_f_gfor', 'cases_df_feasible'),
        ('tau_droop_u_gfor', 'cases_df_feasible'),
        ('tau_droop_f_gfol', 'cases_df_feasible'),
        ('tau_droop_u_gfol', 'cases_df_feasible'),
    ])

    processed_blocks = {}

    for prefix, source_df_key in prefixes_and_sources:
        source_df = results_dataframes[source_df_key]
        column_names = [
            var for var in columns_in_df[source_df_key] if var.startswith(prefix)]

        if prefix.startswith('tau_droop'):
            df_partial = results_dataframes['case_df_op_feasible'][['case_id']].merge(
                source_df[column_names+['case_id']], on='case_id', how='left').drop(['case_id'], axis=1)
        else:
            df_partial = source_df[column_names]
        processed_blocks[prefix] = fill_and_sort(
            df_partial, prefix, np.arange(1, n_buses+1))

    # for Sn in ['Sn_GFOL', 'Sn_GFOR']:
    #     if Sn in processed_blocks.keys():
    #         processed_blocks[Sn]=processed_blocks[Sn]/100

    # if 'theta' in processed_blocks.keys():
    #     # Adjust angles greater than 180°
    #     processed_blocks['theta'] = processed_blocks['theta'] - (processed_blocks['theta'] > 180) * 360

    #     processed_blocks['theta']=processed_blocks['theta']*np.pi/180

    return processed_blocks


def fill_and_sort(df_partial, prefix, list_bus):
    if prefix.startswith('tau_droop'):
        all_cols = [f'{prefix}_{i}' for i in list_bus]
    else:
        all_cols = [f'{prefix}{i}' for i in list_bus]
    for col in all_cols:
        if col not in df_partial.columns:
            df_partial[col] = 0
    if prefix.startswith('tau_droop'):
        return df_partial[sorted(df_partial.columns, key=lambda x: int(x.split('_')[-1]))]
    else:
        return df_partial[sorted(df_partial.columns, key=lambda x: int(x.split(prefix)[-1]))]

# for prefix, source_df_key in prefixes_and_sources:
#     source_df = results_dataframes[source_df_key]
#     column_names = [var for var in columns_in_df[source_df_key] if var.startswith(prefix)]

#     df_partial = results_dataframes['case_df_op_feasible'][['case_id']].merge(source_df[column_names+['case_id']], on='case_id', how='left').drop(['case_id'],axis=1)
#     processed_blocks[prefix] = fill_and_sort(df_partial, prefix, n_buses)


# prefixes_and_sources = [
#     ('Stability', 'case_df_op_feasible')]

# for prefix, source_df_key in prefixes_and_sources:
#     source_df = results_dataframes[source_df_key]
#     column_names = [var for var in columns_in_df[source_df_key] if var.startswith(prefix)]
#     df_partial = source_df[column_names]
#     new_cols = [f'Stability{i}' for i in range(1, n_buses)]
#     df_partial[new_cols] = pd.DataFrame({col: df_partial['Stability'].values for col in new_cols}, index=df_partial.index)
#     processed_blocks[prefix] = df_partial

# %%
processed_blocks_PFcontrolrole = create_processed_blocks_Notaus(
    dict_var, single_buses, columns_in_df, results_dataframes)

n_buses = 118
processed_blocks_PFcontrolrole_taus = create_processed_blocks_with_taus(
    dict_var, results_dataframes, columns_in_df, n_buses)

# TODO: sort first uncorr var and then corr

# %% ---- SCALE DATA ----

processed_blocks_scaled = {}
scaler = StandardScaler()  # MinMaxScaler()#

for key, item in processed_blocks_PFcontrolrole_taus.items():
    processed_blocks_scaled[key] = scaler.fit_transform(item)

df_full = results_dataframes['case_df_op_feasible'][['case_id']].merge(results_dataframes['cases_df_feasible'][[
                                                                       col for col in columns_in_df['cases_df_feasible'] if col.startswith('tau_droop')]+['case_id']], on='case_id', how='left').drop(['case_id'], axis=1)

# pd.concat([results_dataframes['case_df_op_feasible_uncorr_X'].reset_index(drop=True), df_full],axis=1)
df_full = results_dataframes['case_df_op_feasible_uncorr_X']

df_full_scaled = scaler.fit_transform(df_full)

# %% ---- Initial PCA Analysis ----

pca = PCA()
X_PCA = pca.fit_transform(df_full_scaled)
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

fig = plt.figure()
ax = fig.add_subplot()
ax.bar(range(1, len(explained_variance)+1), explained_variance,
       alpha=0.5, align='center', label='Individual explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.tick_params(axis='x')
plt.tick_params(axis='y')
# ax.set_xlim(0,20)
# ax.set_xticks([1,2,3])
# ax.set_xticklabels(['$X_A$','$X_B$','$X_C$'])
plt.tight_layout()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1),
         cumulative_variance, marker='o', linestyle='--')
plt.title('Cumulative Explained Variance by Number of Dimensions')
plt.xlabel('Number of Dimensions')
plt.ylabel('Cumulative Explained Variance')
plt.xlim(1, len(cumulative_variance))
plt.ylim(0, 1)
plt.grid()
plt.axhline(y=0.95, color='r', linestyle='--',
            label='95% Variance Threshold')  # Optional threshold line
plt.legend()
plt.show()


# %% ---- VOLTAGE PROFILE ANALYSIS ----

# fig=plt.figure()
# ax=fig.add_subplot()
# ax.plot(processed_blocks['V'].T, c= 'skyblue')
# ax.set_ylabel('V [p.u.]')
# ax.set_xlabel('Buses')
# ax.set_xticks([list(np.arange(0,118,10))],labels=[str(i) for i in np.arange(0,118,10)])
# ax.title('Voltage profiles of feasible OPs')
# ax.grid()
# plt.tight_layout()

# # Suppose df is your original DataFrame of shape (2000, 119)
# # Columns 0-117: voltages | Column 118: stability label
# df = pd.concat([processed_blocks['V'], results_dataframes['case_df_op_feasible'][['Stability']]],axis=1)
# df.columns = list(range(118)) + ['stability']  # name the last column
# df['case_id'] = df.index  # keep track of each row (sample)

# # Melt the wide DataFrame to long format
# df_long = df.melt(id_vars=['case_id', 'stability'], var_name='bus', value_name='voltage')
# df_long['bus'] = df_long['bus'].astype(int)  # bus index for x-axis

# # Plot using Seaborn
# sns.lineplot(data=df_long, x='bus', y='voltage', hue='stability', errorbar='sd')
# plt.title("Mean Voltage Profile by Stability Class (±95% CI)")
# plt.xlabel("Bus Index")
# plt.ylabel("Voltage")
# plt.legend(title='Stability', labels=['Stable (0)', 'Unstable (1)'])
# plt.grid(True)
# plt.show()

# %%
# Scale the data
scaler = MinMaxScaler()  # StandardScaler()
scaled_data = scaler.fit_transform(results_dataframes['case_df_op_feasible_X'])

pca_full = PCA()
pca_full.fit(scaled_data)
explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

X_PCA = pca_full.fit_transform(scaled_data)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1),
         cumulative_variance, marker='o', linestyle='--')
plt.title('Cumulative Explained Variance by Number of Dimensions')
plt.xlabel('Number of Dimensions')
plt.ylabel('Cumulative Explained Variance')
plt.xlim(1, len(cumulative_variance))
plt.ylim(0, 1)
plt.grid()
plt.axhline(y=0.95, color='r', linestyle='--',
            label='95% Variance Threshold')  # Optional threshold line
plt.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot()
ax.bar(range(1, len(explained_variance)+1), explained_variance,
       alpha=0.5, align='center', label='Individual explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.tick_params(axis='x')
plt.tick_params(axis='y')
ax.set_xlim(0, 20)
# ax.set_xticks([1,2,3])
# ax.set_xticklabels(['$X_A$','$X_B$','$X_C$'])
plt.tight_layout()


def biplot(score, coeff, labels=None):
    xs = score[:, 0]
    ys = score[:, 2]
    n = coeff.shape[0]
    plt.figure(figsize=(10, 7))
    plt.scatter(xs, ys, alpha=0.5)
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0]*5, coeff[i, 1]*5, color='r', alpha=0.7)
        if labels is None:
            plt.text(coeff[i, 0]*5.2, coeff[i, 1]*5.2,
                     f"Var{i+1}", color='g', ha='center', va='center')
        else:
            plt.text(coeff[i, 0]*5.2, coeff[i, 1]*5.2, labels[i],
                     color='g', ha='center', va='center')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid()
    plt.title("PCA Biplot")
    plt.tight_layout()
    plt.show()


biplot(X_PCA, pca_full.components_.T[0:2, :],
       labels=results_dataframes['case_df_op_feasible_X'].columns)

# %%

idx_stab = results_dataframes['case_df_op_feasible'].reset_index(
    drop=True).query('Stability == 1').index
idx_unstab = results_dataframes['case_df_op_feasible'].reset_index(
    drop=True).query('Stability == 0').index

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(X_PCA[idx_unstab, 0], X_PCA[idx_unstab, 1], c='r', alpha=0.2)
ax.scatter(X_PCA[idx_stab, 0], X_PCA[idx_stab, 1], c='g', alpha=0.2)

fig = plt.figure()
ax = fig.add_subplot()
ax.hist2d(X_PCA[idx_stab, 0], X_PCA[idx_stab, 1],
          bins=(50, 50), cmap=plt.cm.Greens_r.reversed())
ax.hist2d(X_PCA[idx_unstab, 0], X_PCA[idx_unstab, 1],
          bins=(50, 50), cmap=plt.cm.Reds_r.reversed())


fig, ax = plt.subplots()

# Stable group histogram
h1, xedges1, yedges1 = np.histogram2d(
    X_PCA[idx_stab, 0], X_PCA[idx_stab, 1], bins=(50, 50))
h1_masked = np.ma.masked_where(h1 == 0, h1)
ax.pcolormesh(xedges1, yedges1, h1_masked.T, cmap=plt.cm.Greens_r.reversed())

# Unstable group histogram
h2, xedges2, yedges2 = np.histogram2d(
    X_PCA[idx_unstab, 0], X_PCA[idx_unstab, 1], bins=(50, 50))
h2_masked = np.ma.masked_where(h2 == 0, h2)
ax.pcolormesh(xedges2, yedges2, h2_masked.T, cmap=plt.cm.Reds_r.reversed())

plt.show()


# %%


# %%

# cosphi_SG=np.cos(np.arctan(np.array(Q_SG)/np.array(P_SG)))

# preserves order of insertion
dataframes = list(processed_blocks_scaled.values())

array_3d = np.stack([df for df in dataframes])

array_3d = array_3d.transpose(1, 0, 2)

array_3d_red = array_3d[np.arange(0, len(array_3d), 10), :, :]

# %%
vmin = np.min(array_3d_red)
vmax = np.max(array_3d_red)

fig, ax = plt.subplots(10, 10, subplot_kw=dict(
    xticks=[], yticks=[]))  # 10, 10,
for i, axi in enumerate(ax.flat):
    pcm = axi.pcolormesh(
        array_3d_red[i], cmap='nipy_spectral', shading='auto', vmin=vmin, vmax=vmax)

fig, ax = plt.subplots(5, 5, subplot_kw=dict(xticks=[], yticks=[]))  # 10, 10,
for i in np.arange(0, 10000, 100):
    pcm = ax.flat[int(i/100)].pcolormesh(array_3d_red[i],
                                         cmap='nipy_spectral', shading='auto', vmin=vmin, vmax=vmax)

# %%

array_2d = np.empty((n_feas_cases, 0))
for key, item in processed_blocks_scaled.items():
    array_2d = np.concatenate((array_2d, item), axis=1)

array_2d_red = array_2d[np.arange(0, len(array_2d), 10), :]

# %%
# from sklearn.decomposition import PCA
# pca = PCA(n_components=10, svd_solver='randomized')
# model = PCA(100).fit(array_2d)

# fig, ax = plt.subplots()
# ax.plot(np.cumsum(model.explained_variance_ratio_))
# ax.set_xlabel('n components')
# ax.set_ylabel('cumulative variance');

# from sklearn.manifold import Isomap
# model = Isomap()#n_components=2)
# proj = model.fit_transform(array_2d_red)
# proj.shape

# %%
# from sklearn.decomposition import PCA, KernelPCA

# kernel_pca = KernelPCA(
#     n_components=2, kernel="rbf", gamma=1, fit_inverse_transform=True, alpha=0.1
# )
# proj = kernel_pca.fit_transform(array_2d_red)
# proj.shape

# fig, ax = plt.subplots()
# ax.plot(np.cumsum(model.explained_variance_ratio_))
# ax.set_xlabel('n components')
# ax.set_ylabel('cumulative variance');

# %%


def plot_components(data, model, stability, images=None, ax=None,
                    thumb_frac=0.05, cmap='nipy_spectral'):
    ax = ax or plt.gca()

    proj = model.fit_transform(data)
    # # ax.plot(proj[:, 0], proj[:, 1], '.k')
    # proj[:,0]=proj1[:,1]
    # proj[:,1]=proj1[:,0]
    ax.plot(proj[stability == 0, 0], proj[stability == 0, 1],
            'o', color='red', label='Unstable')
    ax.plot(proj[stability == 1, 0], proj[stability == 1, 1],
            'o', color='green', label='Stable')

    if images is not None:
        min_dist_2 = (thumb_frac * max(proj.max(0) - proj.min(0))) ** 2
        shown_images = np.array([2 * proj.max(0)])
        for i in range(data.shape[0]):
            dist = np.sum((proj[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                # don't show points that are too close
                continue
            shown_images = np.vstack([shown_images, proj[i]])
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i], cmap=cmap),
                proj[i],
                pad=0.1,
            )
            ax.add_artist(imagebox)

# %%


stability_array_2d_red = np.array(results_dataframes['case_df_op_feasible'].iloc[np.arange(
    0, len(results_dataframes['case_df_op_feasible']), 10)]['Stability'])
# model = Isomap(n_components=2)

# model = KernelPCA(
#     n_components=2, kernel="rbf", gamma=0.13, fit_inverse_transform=True, alpha=0.1)
model = PCA(n_components=2)

fig, ax = plt.subplots(figsize=(10, 10))
plot_components(array_2d_red,
                model, stability_array_2d_red,
                images=array_3d_red)  # [:, ::2, ::2])
