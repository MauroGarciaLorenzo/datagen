import pandas as pd
import os

def open_csv(path_results, csv_files=None):
    if csv_files == None:
        csv_files = [file for file in os.listdir(path_results) if file.endswith('.csv')]

    results_dataframes=dict()
    for file in csv_files:
        results_dataframes[file.replace('.csv','')]=pd.read_csv(path_results+'/'+file,sep=',').drop(['Unnamed: 0'],axis=1).drop_duplicates(keep='first').reset_index(drop=True)
    
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


def plot_mesh(mesh_df, ax = None):
    # Group by each block based on entropy, delta_entropy, and depth
    grouped = mesh_df.groupby('block_id')

    if ax == None:
        # Create the plot
        fig, ax = plt.subplots()
    
    for i, group in grouped:
        #group = block_id_group[1]
        try:
            p_cig_row = group[group["dimension"] == "p_cig"].iloc[0]
            p_sg_row = group[group["dimension"] == "p_sg"].iloc[0]
    
            x0, x1 = p_cig_row["lower"], p_cig_row["upper"]
            y0, y1 = p_sg_row["lower"], p_sg_row["upper"]
    
            rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                     linewidth=1, edgecolor='blue', facecolor='lightblue', alpha=0.4)
            ax.add_patch(rect)
        except IndexError:
            # Skip blocks that are missing either p_cig or p_sg
            continue
    
    ax.set_xlabel("Total $P_{IBR}$ [MW]")
    ax.set_ylabel("Total $P_{SG}$ [MW]")
    #ax.set_title("2D Mesh of p_cig vs p_sg")
    plt.grid(True)
    ax.set_xlim(900, 4700)    # Example range for p_cig
    ax.set_ylim(4000, 1.1*mesh_df.query('dimension == "p_sg"')['upper'].max())   # Example range for p_sg
    plt.tight_layout()

    return ax