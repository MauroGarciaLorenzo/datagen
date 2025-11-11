import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def sensitivity(cases_df, dimensions, divs_per_cell, generator):
    """This Sensitivity analysis is done by gathering cases and their
    evaluated outputs, then train a Random Forest, getting the importance of
    each variable in the decision. Each variable's division count is
    initialized at 1. In a loop, iterated as many times as specified, we double
    the number of subdivisions for the most influential variable and halve its
    importance.

    :param generator:
    :param cases_df: Involved cases
    :param dimensions: Involved dimensions
    :param divs_per_cell: Number of resultant cells from each recursive call
    :return: Divisions for each dimension
    """
    print("=== STARTING SENSITIVITY ANALYSIS ===")
    # Extract labels for dimensions that are marked as independent
    labels = [dim.label for dim in dimensions if dim.independent_dimension]

    # Prepare DataFrame that will hold aggregated values per dimension
    dims_df = pd.DataFrame()
    for label in labels:
            matching_columns = (
                cases_df.filter(regex=r'^' + label + r'_*', axis=1).sum(axis=1))
            dims_df[label] = matching_columns
    dims_df.columns = labels

    # Convert to NumPy arrays: x = features, y = target (Stability)
    x = np.array(dims_df)
    y = np.array(cases_df["Stability"])
    y = y.astype('int')

    # Standardize, use reproducible seed, and fit Random Forest
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    random_state = generator.integers (0,2**32 - 1)
    model = RandomForestClassifier(random_state=random_state)
    model.fit(x_scaled, y)

    # Get feature importances (sensitivity indicators)
    importances = model.feature_importances_
    print(f"IMPORTANCES: {importances}")

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
            print(f"Selected dimension: {d.label}, divisions: {d.divs}")
    if not success:
        print("Could not split based on sensitivity. Only one "
                    "cell will appear at the next level of depth")
    print("=== FINISHED SENSITIVITY ANALYSIS ===")
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
