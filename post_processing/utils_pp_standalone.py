import pandas as pd
import os

def open_csv(path_results, csv_files=None, results_dataframes=None):
    if csv_files == None:
        csv_files = [file for file in os.listdir(path_results) if file.endswith('.csv')]

    if results_dataframes == None:
        results_dataframes=dict()
    for file in csv_files:
        try:
            results_dataframes[file.replace('.csv','')]=pd.read_csv(path_results+'/'+file,sep=',').drop(['Unnamed: 0'],axis=1).drop_duplicates(keep='first').reset_index(drop=True)
        except:
            results_dataframes[file.replace('.csv','')]=pd.read_csv(path_results+'/'+file,sep=',').drop_duplicates(keep='first').reset_index(drop=True)
        try:
            results_dataframes[file.replace('.csv','')]['cell_name'] = ['0' if name == 0.0 or name == '0.0' else str(name) for name in results_dataframes[file.replace('.csv','')]['cell_name']]
        except:
            continue
    return results_dataframes, csv_files 

def perc_stability(df,dir_name):
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", flush=True)
    print(dir_name)
    print('$|\mathcal{D}|$: '+str(len(df)))
    print('Feasible cases: '+str(len(df.query('Stability>=0'))/len(df)*100)+'%')
    print('Stable cases: '+str(len(df.query('Stability == 1'))/len(df)*100)+'% of total cases')
    print('Stable cases: '+str(len(df.query('Stability == 1'))/len(df.query('Stability>=0'))*100)+'% of feasible cases')
    print('Unfeasible cases: '+str(len(df.query('Stability==-1'))/len(df)*100)+'%')
    print('Out of cell cases: '+str(len(df.query('Stability==-2'))/len(df)*100)+'%')

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", flush=True)

def find_closest_row(df, columns, values):
    """
    Find the row in df closest to the provided values in specified columns.

    Parameters:
    - df: pandas DataFrame
    - columns: list of column names to compare
    - values: list of target values corresponding to columns

    Returns:
    - pandas Series: the closest matching row
    """

    if len(columns) != len(values):
        raise ValueError("Length of columns and values must match.")

    # Compute the total absolute distance
    df = df.copy()  # avoid modifying original DataFrame
    df["distance"] = sum(abs(df[col] - val) for col, val in zip(columns, values))

    # Get the row with the minimum distance
    idx = df["distance"].idxmin()
    closest_row = df.loc[idx].drop("distance")
    df = df.drop(idx,axis=0)#.drop("distance")
    return closest_row, idx, df

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_mesh(mesh_df, dimx, dimy, labelx, labely, ax = None):
    # Group by each block based on entropy, delta_entropy, and depth
    grouped = mesh_df.groupby('block_id')

    if ax == None:
        # Create the plot
        fig, ax = plt.subplots()
        ax.set_xlabel(labelx)#"Total $P_{IBR}$ [MW]")
        ax.set_ylabel(labely)#"Total $P_{SG}$ [MW]")
        
    
    for i, group in grouped:
        #group = block_id_group[1]
        try:
            p_cig_row = group[group["dimension"] == dimx].iloc[0]
            p_sg_row = group[group["dimension"] == dimy].iloc[0]
    
            x0, x1 = p_cig_row["lower"], p_cig_row["upper"]
            y0, y1 = p_sg_row["lower"], p_sg_row["upper"]
    
            rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                     linewidth=1, edgecolor='blue', facecolor='none', alpha=0.5)
            ax.add_patch(rect)
        except IndexError:
            # Skip blocks that are missing either p_cig or p_sg
            continue
    
    #ax.set_title("2D Mesh of p_cig vs p_sg")
    plt.grid(True)
    if dimx=='perc_g_for':
        ax.set_xlim(-10,1.1* mesh_df.query('dimension == @dimx')['upper'].max())    # Example range for p_cig
        ax.set_ylim(0.9*mesh_df.query('dimension == @dimy')['lower'].min(), 1.1*mesh_df.query('dimension == @dimy')['upper'].max())   # Example range for p_sg
    elif dimy=='perc_g_for':
        ax.set_xlim(0.9*mesh_df.query('dimension == @dimx')['lower'].min(),1.1* mesh_df.query('dimension == @dimx')['upper'].max())    # Example range for p_cig
        ax.set_ylim(-10, 1.1*mesh_df.query('dimension == @dimy')['upper'].max())   # Example range for p_sg
    else:
         ax.set_xlim(0.9*mesh_df.query('dimension == @dimx')['lower'].min(),1.1* mesh_df.query('dimension == @dimx')['upper'].max())    # Example range for p_cig
         ax.set_ylim(0.9*mesh_df.query('dimension == @dimy')['lower'].min(), 1.1*mesh_df.query('dimension == @dimy')['upper'].max())   # Example range for p_sg
    
   
    plt.tight_layout()

        
    return ax

from scipy.stats import normaltest
from scipy.stats import skew, kurtosis

def calculate_skewness(df):
    skewness = pd.DataFrame(columns=['all_points','stable_points','unstable_points'])
    skewness['all_points']  = df.drop(['Stability'],axis=1).apply(skew)
    skewness['stable_points']  = df.query('Stability == 1').drop(['Stability'],axis=1).apply(skew)
    skewness['unstable_points']  = df.query('Stability == 0').drop(['Stability'],axis=1).apply(skew)
    return skewness

def calculate_kurt(df):
    kurtosis_df = pd.DataFrame(columns=['all_points','stable_points','unstable_points'])
    kurtosis_df['all_points']  = df.drop(['Stability'],axis=1).apply(kurtosis)
    kurtosis_df['stable_points']  = df.query('Stability == 1').drop(['Stability'],axis=1).apply(kurtosis)
    kurtosis_df['unstable_points']  = df.query('Stability == 0').drop(['Stability'],axis=1).apply(kurtosis)
    return kurtosis_df

import numpy as np
def calculate_pu_skewness_kurt(df, columns_list, series_Sn):
    # Make sure divisor is float
    series_Sn = np.asarray(series_Sn, dtype=float)

    # Build a Series indexed by your columns, to align correctly
    divisor = pd.Series(series_Sn, index=columns_list, dtype=float)

    df_pu = (
        df[columns_list]
        .astype(float)
        .div(divisor, axis=1)
    )

    
    df_pu['Stability']=df['Stability']
    
    skewness = calculate_skewness(df_pu)
    kurtosis_df = calculate_kurt(df_pu)
    return df_pu, skewness, kurtosis_df

def distribution_plots(df, var):
# Data
    all_vals = df[var]
    stable_vals = df.query('Stability == 1')[var]
    unstable_vals = df.query('Stability == 0')[var]

    plt.figure(figsize=(8,5))

    plt.hist(all_vals, bins=40, alpha=0.5, label='All', density=True)
    plt.hist(stable_vals, bins=40, alpha=0.5, label='Stability = 1', density=True)
    plt.hist(unstable_vals, bins=40, alpha=0.5, label='Stability = 0', density=True)

    plt.legend()
    plt.xlabel(var)
    plt.ylabel('Density')
    plt.title('Distributions of '+var)
    plt.show()
    
def plot_distribution_with_changes(skewness_or_kurt,df):
    mask_sign_change = (skewness_or_kurt['stable_points'] * skewness_or_kurt['unstable_points'] < 0)
    indices_with_sign_change = skewness_or_kurt.index[mask_sign_change]
    #if len(indices_with_sign_change)>0:
    for var in indices_with_sign_change:
        distribution_plots(df,var)
        