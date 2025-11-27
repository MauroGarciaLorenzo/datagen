import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def sensitivity(cases_df, df_op, dimensions, divs_per_cell, generator, use_all_vars=False):
    """This Sensitivity analysis is done by gathering cases and their
    evaluated outputs, then train a Random Forest, getting the importance of
    each variable in the decision. Each variable's division count is
    initialized at 1. In a loop, iterated as many times as specified, we double
    the number of subdivisions for the most influential variable and halve its
    importance. Only use feasible cases for sensitivity calculations

    :param generator:
    :param cases_df: Involved cases
    :param dimensions: Involved dimensions
    :param divs_per_cell: Number of resultant cells from each recursive call
    :param use_all_vars: Whether to use all variables available or a 
        reduced subset designed specifically for the ACOPF problem
    :return: Divisions for each dimension
    """
    print("=== STARTING SENSITIVITY ANALYSIS ===", flush=True)
    # Extract labels for dimensions that are marked as independent
    labels = [dim.label for dim in dimensions if dim.independent_dimension]

    # Discard unfeasible cases for the sensitivity calculations
    cases_df_feas = cases_df.query('Stability >=0')

    if cases_df_feas.empty:
        return dimensions

    if use_all_vars:
        # Run sensitivity using all variables
        dims_df_feas = pd.DataFrame()
        for label in labels:
            # Prepare DataFrame that will hold aggregated values per dimension
            matching_columns = (
                cases_df_feas.filter(regex=r'^' + label + r'_*', axis=1).sum(axis=1))
            dims_df_feas[label] = matching_columns
        dims_df_feas.columns = labels
    else:
        # Run targeted procedure for specific case
        df_op_feas = df_op.query('Stability >=0')
        
        dims_df_feas = pd.DataFrame()
        
        dims_df_feas['p_sg'] = df_op_feas[[col for col in df_op_feas.columns if col.startswith('P_SG')]].sum(axis=1)
        dims_df_feas['p_cig'] = df_op_feas[[col for col in df_op_feas.columns if col.startswith('P_GFOR') or col.startswith('P_GFOL')]].sum(axis=1)
        p_gfor = df_op_feas[[col for col in df_op_feas.columns if col.startswith('P_GFOR')]].sum(axis=1)
        dims_df_feas['perc_g_for'] = p_gfor/dims_df_feas['p_cig']
        
        col_sn_gfor =[col for col in  df_op_feas.columns if col.startswith('Sn_GFOR')]
        taus_gfor = ['tau_droop_u_gfor_'+bus.split('GFOR')[1] for bus in col_sn_gfor]
        taus_gfor = taus_gfor + ['tau_droop_f_gfor_'+bus.split('GFOR')[1] for bus in col_sn_gfor]
    
        col_sn_gfol =[col for col in  df_op_feas.columns if col.startswith('Sn_GFOL')]
        taus_gfol = ['tau_droop_u_gfol_'+bus.split('GFOL')[1] for bus in col_sn_gfol]
        taus_gfol = taus_gfol + ['tau_droop_f_gfol_'+bus.split('GFOL')[1] for bus in col_sn_gfol]
    
        dims_df_feas[taus_gfor] = cases_df_feas[taus_gfor]
        dims_df_feas[taus_gfol] = cases_df_feas[taus_gfol]

    # Convert to NumPy arrays: x = features, y = target (Stability)
    x = np.array(dims_df_feas)
    y = np.array(cases_df_feas["Stability"])
    y = y.astype('int')

    # Standardize, use reproducible seed, and fit Random Forest
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    random_state = generator.integers (0,2**32 - 1)
    model = RandomForestClassifier(random_state=random_state)
    model.fit(x_scaled, y)

    # Get feature importances (sensitivity indicators)
    importances = model.feature_importances_
    print(f"IMPORTANCES: {importances}", flush=True)

    # Reset all dimension divisions to 1 (baseline state)
    for d in dimensions:
        d.divs = 1

    # Compute how many refinement steps to perform
    # Uses log2(divs_per_cell) — assumes divs_per_cell is roughly a power of 2
    splits_per_cell = int(np.round(np.log2(divs_per_cell)))
    for _ in range(splits_per_cell):
        # plot_importances_and_divisions(dimensions, importances)

        # Identify the most important dimension according to Random Forest
        index_max_importance = np.argmax(importances)
        label_max_importance = list(labels)[index_max_importance]
        dim_max_importance = get_dimension(label_max_importance, dimensions)

        # Check if further dividing this dimension exceeds tolerance threshold
        if ((dim_max_importance.borders[1] - dim_max_importance.borders[0]) /
                dim_max_importance.divs < dim_max_importance.tolerance):
            # If too fine, set importance to 0 to skip further splitting
            importances[index_max_importance] = 0

        # If still important and refinable, double the number of divisions
        # and reduce its importance by half (heuristic)
        if importances[index_max_importance] != 0:
            dim_max_importance.divs *= 2
            importances[index_max_importance] /= 2

    # plot_importances_and_divisions(dimensions, importances)

    # Log which dimension was selected for refinement
    success = False
    for d in dimensions:
        if d.divs > 1:
            success = True
            logger.info(f"Selected dimension: {d.label}, divisions: {d.divs}")
    if not success:
        logger.warning("Could not split based on sensitivity. Only one "
                    "cell will appear at the next level of depth")
    logger.debug("=== FINISHED SENSITIVITY ANALYSIS ===")
    return dimensions


def get_dimension(label, dimensions):
    if label == "g_for" or label == "g_fol":
        dim = next(
            (d for d in dimensions
             if d.label == "p_cig"), None)
    else:
        dim = next((d for d in dimensions
                    if d.label == label), None)
    return dim
