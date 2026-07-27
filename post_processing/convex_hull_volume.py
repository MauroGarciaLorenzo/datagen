import numpy as np
from scipy.spatial import ConvexHull, QhullError
import matplotlib.pyplot as plt
import pandas as pd
import os
from utils_pp_standalone import *
from collections import defaultdict
import re
from sklearn.preprocessing import MinMaxScaler
from utils_dataset_quality_metrics import *

# # Generate random points
# points = np.random.rand(30, 2)

# # Compute convex hull (uses Qhull internally)
# hull = ConvexHull(points)

# # Plot
# plt.plot(points[:,0], points[:,1], 'o')

# for simplex in hull.simplices:
#     plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

# plt.show()


# points = np.random.rand(100, 3)
# hull = ConvexHull(points)

# print("Hull vertices:")
# print(hull.vertices)

# print("Hull volume:")
# print(hull.volume)

# # Create 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot original points
# ax.scatter(points[:, 0], points[:, 1], points[:, 2])

# # Plot hull triangles
# for simplex in hull.simplices:
#     triangle = points[simplex]
#     ax.plot_trisurf(triangle[:, 0],
#                     triangle[:, 1],
#                     triangle[:, 2],
#                     alpha=0.3)

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# plt.show()

#%%
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
path = 'D:/'
for dataset_num, dataset_ID in enumerate(['_2862','_2732']):
    dir_name=[dir_name for dir_name in os.listdir(path) if dataset_ID in dir_name and 'zip' not in dir_name][0]#_2862# _2732 if dir_name.startswith('datagen') and 'zip' not in dir_name]#
    print(dir_name)
    #'20251030_sensitivity'
    #%%
    
    #dataset_ID = dir_name[-5:].replace('ivity','Sensitivity')
    
    #%%
    df_dict=dict()
    df_dict['DataSet_training_uncorr_var'+dataset_ID] = pd.read_csv(path+dir_name+'/DataSet_training_uncorr_var'+dataset_ID+'.csv').drop('Unnamed: 0', axis=1).drop_duplicates(keep='first')
    uncorr_vars = list(set(df_dict['DataSet_training_uncorr_var'+dataset_ID].columns)-set(['Stability','case_id']))
    
    # df_dict['DataSet_training_uncorr_var_HierCl'+dataset_ID] = pd.read_csv(path+dir_name+'/DataSet_training_uncorr_var_HierCl'+dataset_ID+'.csv').drop('Unnamed: 0', axis=1).drop_duplicates(keep='first')
    
    # case_id_feasible = list(df_dict['DataSet_training_uncorr_var'+dataset_ID]['case_id'])
    # cases_id_depth = pd.read_excel(path+dir_name+'/cases_id_depth'+dataset_ID+'.xlsx')[['Depth','case_id','CellName']]
    
    # cases_id_depth_feas = cases_id_depth.query('case_id == @case_id_feasible')
    
    #%%
    #for dir_name in dir_names:
    path_results = os.path.join(path, dir_name)
    df_op='df_op'#'case_df_op'
    results_dataframes, csv_files = open_csv(
        path_results, ['cases_df.csv', df_op+'.csv','dims_df.csv'])
    
    perc_stability(results_dataframes[df_op], dir_name)
    
    
    for key, item in results_dataframes.items():
        print(key+': '+str(len(item)))
        #results_dataframes[key+'_drop_duplicates']= item.drop(['case_id'],axis=1).drop_duplicates(keep='first')
        print(key+'_drop_duplicates'+': '+str(len(item.drop_duplicates(keep='first'))))
    
    # %% ---- FILL NAN VALUES WITH NULL ---
    
    results_dataframes[df_op] = results_dataframes[df_op].fillna(0)
    
    # %% ---- FIX VALUES ----
    
    Sn_cols = [col for col in results_dataframes[df_op]
               if col.startswith('Sn')]
    results_dataframes[df_op][Sn_cols] = results_dataframes[df_op][Sn_cols]/100
    
    theta_cols = [col for col in results_dataframes[df_op]
                  if col.startswith('theta')]
    # # Adjust angles greater than 180°
    # results_dataframes[df_op][theta_cols] = results_dataframes[df_op][theta_cols] - \
    #     (results_dataframes[df_op][theta_cols] > 180) * 360
    
    # results_dataframes['case_df_op'][theta_cols] = results_dataframes['case_df_op'][theta_cols] * np.pi/180
    
    # add total demand variables
    PL_cols = [
        col for col in results_dataframes[df_op].columns if col.startswith('PL')]
    results_dataframes[df_op]['PD'] = results_dataframes[df_op][PL_cols].sum(
        axis=1)
    
    QL_cols = [
        col for col in results_dataframes[df_op].columns if col.startswith('QL')]
    results_dataframes[df_op]['QD'] = results_dataframes[df_op][QL_cols].sum(
        axis=1)
    
    # %% ---- SELECT ONLY FEASIBLE CASES ----
    
    results_dataframes['case_df_op_feasible'] = results_dataframes['df_op'].query(
        'Stability >= 0')
    
    case_id_feasible = list(results_dataframes['case_df_op_feasible']['case_id'])
    case_id_cell0 =  list(results_dataframes['df_op'].query('cell_name == "0"')['case_id'])
    case_id_cell0_feas =  list(results_dataframes['df_op'].query('cell_name == "0" and Stability >=0')['case_id'])
    
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
    
    results_dataframes['case_df_op_feasible_X'] = results_dataframes['case_df_op_feasible'].drop(['case_id', 'Stability','cell_name'], axis=1)
    
    results_dataframes['dims_df_feas']= results_dataframes['dims_df'].query('case_id==@case_id_feasible').reset_index(drop=True)
    # %% ---- SELECT ONLY UNFEASIBLE CASES ----
    
    results_dataframes['case_df_op_unfeasible'] = results_dataframes[df_op].query(
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
    
    # %% ----  Check if there are extra taus ----
    
    df_taus = results_dataframes['case_df_op_feasible'][['case_id']].merge(results_dataframes['cases_df_feasible'][[
                                                                           col for col in columns_in_df['cases_df_feasible'] if col.startswith('tau_droop')]+['case_id']], on='case_id', how='left').drop(['case_id'], axis=1)
    
    df_Sn_GFOL = results_dataframes['case_df_op_feasible'][[col for col in columns_in_df['case_df_op_feasible'] if col.startswith('Sn_GFOL')]].reset_index(drop=True)
    df_taus_GFOL_droopf = df_taus[['tau_droop_f_gfol_'+col.split('GFOL')[1] for col in df_Sn_GFOL.columns]]
    df_taus_GFOL_droopu = df_taus[['tau_droop_u_gfol_'+col.split('GFOL')[1] for col in df_Sn_GFOL.columns]]
    df_taus_GFOL_droopf[np.array(df_Sn_GFOL==0)]=0
    df_taus_GFOL_droopu[np.array(df_Sn_GFOL==0)]=0
    
    df_Sn_GFOR = results_dataframes['case_df_op_feasible'][[col for col in columns_in_df['case_df_op_feasible'] if col.startswith('Sn_GFOR')]].reset_index(drop=True)
    df_taus_GFOR_droopf = df_taus[['tau_droop_f_gfor_'+col.split('GFOR')[1] for col in df_Sn_GFOR.columns]]
    df_taus_GFOR_droopu = df_taus[['tau_droop_u_gfor_'+col.split('GFOR')[1] for col in df_Sn_GFOR.columns]]
    df_taus_GFOR_droopf[np.array(df_Sn_GFOR==0)]=0
    df_taus_GFOR_droopu[np.array(df_Sn_GFOR==0)]=0
    
    df_taus_fixed = pd.concat([df_taus_GFOL_droopf,df_taus_GFOL_droopu,df_taus_GFOR_droopf,df_taus_GFOR_droopu],axis=1)
    
    #%%
    df_taus_fixed['Stability'] = results_dataframes['case_df_op_feasible']['Stability'].reset_index(drop=True)
    df_Sn_GFOL['Stability'] = results_dataframes['case_df_op_feasible']['Stability'].reset_index(drop=True)
    df_Sn_GFOR['Stability'] = results_dataframes['case_df_op_feasible']['Stability'].reset_index(drop=True)
    
    #%%
    
    theta_rad_abs = np.abs(results_dataframes['case_df_op_feasible'][theta_cols]*np.pi/180)
    df_slack = pd.DataFrame(columns =['slack_bus','slack_theta'], index = theta_rad_abs.index)
    for ii in theta_rad_abs.index:
        df_slack.loc[ii, 'slack_bus'] = theta_rad_abs.loc[ii].index[theta_rad_abs.loc[ii].argmin()]
        df_slack.loc[ii, 'slack_theta'] = theta_rad_abs.loc[ii].min()
        
    results_dataframes['case_df_op_feasible']['slack_bus']= df_slack['slack_bus']
    
    slack_case=dict()
    for sl_bus in df_slack['slack_bus'].unique():
        slack_case[sl_bus] = list(results_dataframes['case_df_op_feasible'].query('slack_bus == @sl_bus')['case_id'])
    
    #%%
    import copy
    theta_rad = results_dataframes['case_df_op_feasible'][theta_cols]*np.pi/180
    theta_rad_slack_26 = copy.copy(theta_rad)
    df_slack['delta_slack'] = 0
    
    print(df_slack.groupby('slack_bus').count())
    #%%
    for ii in df_slack.query('slack_bus != "theta26"').index:
        slack_bus = df_slack.loc[ii,'slack_bus']
        delta_slack = theta_rad.loc[ii,'theta26'] - theta_rad.loc[ii,slack_bus]
        df_slack.loc[ii,'delta_slack'] = delta_slack
        theta_rad_slack_26.loc[ii,theta_cols] = theta_rad.loc[ii,theta_cols] - delta_slack
        
    #%%
    results_dataframes['raw_data']=results_dataframes['case_df_op_feasible'].drop(theta_cols, axis=1).reset_index(drop=True)
    results_dataframes['raw_data'] = pd.concat([results_dataframes['raw_data'],#df_taus_fixed.drop('Stability',axis=1),
                                                theta_rad_slack_26.reset_index(drop=True)],axis=1)
    
    
    #%%
    def print_columns_groups(key, columns_list):
        # Group columns by the alphabetic prefix
        groups = defaultdict(list)
        for col in columns_list:
            match = re.match(r"([A-Za-z_]+)", col)  # extract the prefix before any digit
            prefix = match.group(1) if match else col
            groups[prefix].append(col)
        
        print(key+':\n')
    
        # Print grouped columns
        for prefix, cols in groups.items():
            print(f"{prefix}: {cols[0]},...,{cols[-1]}; N. elements: {len(cols)}\n")
            
    print_columns_groups('raw_data', results_dataframes['raw_data'])
    
    #%%
    results_dataframes['raw_data'] = results_dataframes['raw_data'].drop(['case_id','Stability','PD','QD','slack_bus'],axis=1)
    
    #%%
    print_columns_groups('dims_df_feas', results_dataframes['dims_df_feas'])
    results_dataframes['dims_df_feas'] = results_dataframes['dims_df_feas'].drop([col for col in results_dataframes['dims_df_feas'].columns if col.startswith('tau')],axis=1)
    #results_dataframes['dims_df_feas'] = pd.concat([results_dataframes['dims_df_feas'], df_taus_fixed.drop('Stability',axis=1)],axis=1)
    
    #%%
    
    #chv_dims=['p_sg','p_g_for','p_g_fol']#'perc_g_for']#'p_g_for','p_g_fol'] #p_cig
    chv_dims=['p_sg','p_cig','perc_g_for']#'p_g_for','p_g_fol'] #p_cig
    scaler_all = MinMaxScaler().fit(np.array(results_dataframes['dims_df'][chv_dims]))
    scaler_feas = MinMaxScaler().fit(np.array(results_dataframes['dims_df_feas'][chv_dims]))

#%%
    # #fig = plt.figure()
    # #ax = fig.add_subplot(111, projection='3d')
    
    # hull_all_points_sg_cig_perc_g_for = conv_hull_volume(np.array(results_dataframes['dims_df'][chv_dims]),chv_dims, scaler_all)#,plot=True, ax=ax)
    # hull_all_points_cell0_sg_cig_perc_g_for = conv_hull_volume(np.array(results_dataframes['dims_df'].query('case_id == @case_id_cell0')[chv_dims]),chv_dims, scaler_all)#, plot=True, ax=ax)
    # hull_feas_points_sg_cig_perc_g_for = conv_hull_volume(np.array(results_dataframes['dims_df_feas'][chv_dims]),chv_dims,scaler_feas)#, plot=True, ax=ax)
    # hull_feas_points_cell0_sg_cig_perc_g_for = conv_hull_volume(np.array(results_dataframes['dims_df_feas'].query('case_id == @case_id_cell0_feas')[chv_dims]),chv_dims, scaler_feas)#, plot=True, ax=ax)
    
    # chv_increase_perc_g_for = (hull_feas_points_sg_cig_perc_g_for.volume-hull_feas_points_cell0_sg_cig_perc_g_for.volume)/hull_feas_points_cell0_sg_cig_perc_g_for.volume
    
#%%
    #df_depth_DS = dict()
    #for path, dataset_ID in zip(dir_names,dataset_ID_list):
        
    df_depth= pd.read_excel('D:/'+dir_name+'/cases_id_depth'+dataset_ID+'.xlsx')
    fig, axes = plt.subplots(1, 2, figsize=(4*2, 5), sharey=True)
    
    hulls_3d=pd.DataFrame(columns=['Depth','All_points','Only_Feasible'])
    for depth in np.sort(df_depth['Depth'].unique())[::-1]:
        case_id_depth = list(df_depth.query('Depth <=@depth')['case_id'])
        hulls_3d.loc[depth,'Depth']=depth
        
        if depth == 5:
            plot_arg =True
            axes_arg = axes
            plot_params_all = {'marker':'o','color':'silver','linestyle':'solid'}
            plot_params_feas = {'marker':'o','color':'#B0E0E6','linestyle':'solid'}
    
        elif depth == 0:
            plot_arg =True
            axes_arg = axes
            plot_params_all = {'marker':'s','color':'silver','linestyle':'--'}
            plot_params_feas = {'marker':'s','color':'#B0E0E6','linestyle':'--'}
        else:
            plot_arg =False
            axes_arg = None
            plot_params_all = None
            plot_params_feas = None
            
        hulls_3d.loc[depth,'All_points'] = conv_hull_volume(np.array(results_dataframes['dims_df'].query('case_id == @case_id_depth')[chv_dims]),
                                    chv_dims, scaler_all, plot=plot_arg, axes=axes_arg, plot_params=plot_params_all).volume
        hulls_3d.loc[depth,'Only_Feasible'] = conv_hull_volume(np.array(results_dataframes['dims_df_feas'].query('case_id == @case_id_depth')[chv_dims]),
                                     chv_dims, scaler_all, plot=plot_arg, axes=axes_arg, plot_params=plot_params_feas).volume
    
    
    hulls_3d.to_excel('./chv_depth'+dataset_ID+'.xlsx')
    
    print('increase total space= ', (hulls_3d.loc[df_depth['Depth'].unique().max(), 'All_points']-hulls_3d.loc[0, 'All_points'])/hulls_3d.loc[0, 'All_points']*100)
    print('increase feasible space= ', (hulls_3d.loc[df_depth['Depth'].unique().max(), 'Only_Feasible']-hulls_3d.loc[0, 'Only_Feasible'])/hulls_3d.loc[0, 'Only_Feasible']*100)
    axes[0].set_xlabel('$P_{SG}$ [p.u.]')
    axes[0].set_ylabel('$P_{IBR}$ [p.u.]')
    axes[1].set_xlabel('$\%P_{GFM}$')
    axes[0].grid()
    fig.suptitle('TEST \#1')
    # Common legend at the bottom (outside)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, 0)
    )
    
    fig.tight_layout(rect=[0, 0.1, 1, 1])  # Leave space for legend
  
#%%
    if dataset_num == 0:
        fig_cvh, axes_cvh = plt.subplots(1, 2, figsize=(4*2, 5), sharey=True)
    
    axes_cvh[dataset_num].plot(hulls_3d['Depth'],hulls_3d['All_points'], linestyle='-', marker='o', color='silver', label = 'All sampled OPs')
    axes_cvh[dataset_num].plot(hulls_3d['Depth'],hulls_3d['Only_Feasible'], marker='o', color='#B0E0E6', linestyle='-', label ='Feasible PF')
    axes_cvh[dataset_num].set_xlabel('Depth')
    axes_cvh[dataset_num].set_ylabel('CHV')
    axes_cvh[dataset_num].grid()
    axes_cvh[dataset_num].set_title('TEST \#'+str(dataset_num+1))
# Common legend at the bottom (outside)
handles, labels =  axes_cvh[dataset_num].get_legend_handles_labels()
fig_cvh.legend(
    handles, labels,
    loc='lower center',
    ncol=3,
    frameon=True,
    bbox_to_anchor=(0.5, 0)
)

fig_cvh.tight_layout(rect=[0, 0.1, 1, 1])  # Leave space for legend
fig_cvh.savefig("./figures_paper/chv_2862_2732.png", dpi=320)
fig_cvh.savefig("./figures_paper/chv_2862_2732.pdf")

#%%



#%%
def minmax_scale(X: np.ndarray, lower=None, upper=None, eps=1e-12):
    """
    Min-max scale columns to [0,1].
    If lower/upper are not provided, uses column min/max from X.
    """
    X = np.asarray(X, dtype=float)
    if lower is None:
        lower = np.nanmin(X, axis=0)
    if upper is None:
        upper = np.nanmax(X, axis=0)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    rng = np.maximum(upper - lower, eps)
    return (X - lower) / rng

def convex_hull_volume(X: np.ndarray) -> float:
    """
    Compute convex hull volume of points X (n_samples x d).
    In 2D, SciPy returns area as 'volume'; in 3D+, it's true volume.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if n < d + 1:
        raise ValueError(f"Need at least d+1 points (got n={n}, d={d}).")
    try:
        hull = ConvexHull(X)  # uses Qhull internally
        return float(hull.volume)
    except QhullError as e:
        # Often happens with co-linear / co-planar / degenerate data
        raise ValueError(
            "Qhull failed (data may be degenerate). "
            "Try adding small jitter or removing duplicates."
        ) from e

def approx_chv(
    X: np.ndarray,
    k: int = 6,
    n_trials: int = 200,
    normalize: bool = True,
    bounds: tuple | None = None,
    seed: int = 0,
):
    """
    Approximate convex hull volume coverage as in the paper:
    - if d > k, repeatedly choose random k-dim subsets and compute hull volume
    Returns: dict with volumes and summary stats.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if k > 6:
        raise ValueError("Paper uses k ≤ 6 (and Qhull gets expensive beyond that).")
    if d < k:
        raise ValueError(f"Your data has only d={d} features; choose k ≤ d.")

    if normalize:
        if bounds is not None:
            lower, upper = bounds
            Xn = minmax_scale(X, lower=lower, upper=upper)
        else:
            Xn = minmax_scale(X)
    else:
        Xn = X

    rng = np.random.default_rng(seed)
    vols = []
    cols = np.arange(d)

    for _ in range(n_trials):
        subset = rng.choice(cols, size=k, replace=False)
        Xi = Xn[:, subset]

        # drop exact duplicate rows (helps Qhull)
        Xi = np.unique(Xi, axis=0)

        # must still have at least k+1 points
        if Xi.shape[0] < k + 1:
            continue

        try:
            vols.append(convex_hull_volume(Xi))
        except ValueError:
            # skip degenerate subsets
            continue

    vols = np.array(vols, dtype=float)
    if vols.size == 0:
        raise ValueError("All trials failed (likely very degenerate data).")

    return vols, {
        "k": k,
        "n_trials_requested": n_trials,
        "n_trials_success": int(vols.size),
        "volumes": vols,
        "mean": float(np.mean(vols)),
        "std": float(np.std(vols, ddof=1)) if vols.size > 1 else 0.0,
        "p05": float(np.percentile(vols, 5)),
        "p50": float(np.percentile(vols, 50)),
        "p95": float(np.percentile(vols, 95)),
    } 

#%%
print_columns_groups('case_df_op_feasible_X', results_dataframes['case_df_op_feasible_X'])


hulls_vars=pd.DataFrame(columns=['Depth','Only_Feasible'])
for depth in np.sort(df_depth['Depth'].unique())[::-1]:
    case_id_depth = list(df_depth.query('Depth <=@depth')['case_id'])
    hulls_vars.loc[depth,'Depth']=depth
    
    hulls_vars.loc[depth,'Only_Feasible'] = approx_chv(results_dataframes['case_df_op_feasible'].query('case_id == @case_id_depth')[list(set(uncorr_vars)-set([col for col in uncorr_vars if col.startswith('tau')]))])

for depth in np.sort(df_depth['Depth'].unique())[::-1]:
    hulls_vars.loc[depth,'Mean'] =  hulls_vars.loc[depth,'Only_Feasible'][0].mean()
    hulls_vars.loc[depth,'std'] =  hulls_vars.loc[depth,'Only_Feasible'][0].std()
    
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(4*2, 5))
# dataset_ID_title_list = ['TEST \#1','TEST \#2']

# for idx, dataset_ID in enumerate(dataset_ID_list):
ax = axes[0]  # Select subplot

ax.errorbar(
    hulls_vars['Depth'],
    hulls_vars['Mean'],
    hulls_vars['std'],
    fmt='o-',
    ecolor='blue',
    elinewidth=1.5,
    capsize=5,
    capthick=1.5,
    markersize=8,
    color='b',
    label='Infeasible'
)
#%%

ibr_hourly_1_year = pd.read_excel('C:/Users/Francesca/miniconda3/envs/gridcal_original2/datagen_BSC/Setup118/CIG_hourly_oneyear.xlsx')
load_hourly_1_year = pd.read_excel('C:/Users/Francesca/miniconda3/envs/gridcal_original2/datagen_BSC/Setup118/Load_hourly_oneyear.xlsx').set_index('DATETIME')

load_hourly_1_year['PD']=load_hourly_1_year.sum(axis=1)

total_demand = pd.DataFrame(columns= ['TrainingData','YearlyData','TrainingData_Sampled'])
total_demand['TrainingData_Sampled'] = results_dataframes['dims_df']['p_sg']*0.9+results_dataframes['dims_df']['p_cig']*0.9
total_demand['TrainingData'] = results_dataframes['case_df_op_feasible']['PD'].reset_index(drop=True)
total_demand['YearlyData'] = load_hourly_1_year['PD'].reset_index(drop=True)
total_demand.describe()

fig, ax = plt.subplots()

ax.boxplot([load_hourly_1_year['PD']/1e3, total_demand['TrainingData_Sampled']/1e3])
ax.set_xticklabels(['1-year hourly demand', 'Sampled demand'])
ax.set_ylabel('P [GW]')
ax.grid()
fig.tight_layout()

#%%


demand_cov_sampled = results_dataframes['dims_df']['p_cig']/(results_dataframes['dims_df']['p_cig']+results_dataframes['dims_df']['p_sg'])*0.9*100
demand_cov_feas = results_dataframes['case_df_op_feasible'][[col for col in results_dataframes['case_df_op_feasible'].columns if col.startswith('P_GFOL') or col.startswith('P_GFOR')]].sum(axis=1)/results_dataframes['case_df_op_feasible']['PD']*100

fig, ax = plt.subplots()

ax.boxplot([demand_cov_sampled, demand_cov_feas])
ax.set_xticklabels(['All sampled OPs', 'Feasible OPs'])
ax.set_ylabel('IBR demand coverage [\%]')
ax.grid()
fig.tight_layout()
#%%


#Sn_CIG = pd.DataFrame(columns = ['Sn_CIG'+bus.split('Sn_GFOR')[-1] for bus in df_Sn_GFOR.columns], index = np.arange(0,len(df_Sn_GFOR)))
#P_CIG = pd.DataFrame(columns = ['P_CIG'+bus.split('Sn_GFOR')[-1] for bus in df_Sn_GFOR.columns], index = np.arange(0,len(df_Sn_GFOR)))

#for col_pcig, col_sncig in zip(P_CIG.columns, Sn_CIG.columns):
 #  P_CIG[col_pcig] =  results_dataframes['case_df_op_feasible'][[col_pcig.replace('CIG','GFOL')]].values + results_dataframes['case_df_op_feasible'][[col_pcig.replace('CIG','GFOR')]].values
   #Sn_CIG[col_sncig] = df_Sn_GFOR[[col_sncig.replace('CIG','GFOR')]].values +  df_Sn_GFOL[[col_sncig.replace('CIG','GFOL')]].values
   
      
#perc_cig = P_CIG.values/ Sn_CIG.values*100
#perc_cig = pd.DataFrame(perc_cig)
#perc_cig.columns = ['IBR_'+bus.replace('P_CIG','') for bus in P_CIG.columns]

gen_data = pd.read_excel('../../stability_analysis/stability_analysis/data/cases/OperationData_IEEE_118_NREL.xlsx',sheet_name='Generators')
Sn_CIG = gen_data.query('Snom_CIG!=0')[['BusNum','Snom_CIG']]

perc_cig = pd.DataFrame(columns = ['IBR_'+bus.split('Sn_GFOR')[-1] for bus in df_Sn_GFOR.columns], index = np.arange(0,len(df_Sn_GFOR)))

for col in perc_cig.columns:
    bus = int(col.split('_')[-1])
    idx = Sn_CIG.query('BusNum == @bus').index[0]
    perc_cig[col]= results_dataframes['cases_df_feasible']['p_cig_Var'+str(idx)].values/Sn_CIG.loc[idx,'Snom_CIG']

fig, ax = plt.subplots()

perc_cig.boxplot(ax=ax,rot=45)

ax.set_ylabel('$P_{IBR}$ [p.u.]')
fig.tight_layout()
