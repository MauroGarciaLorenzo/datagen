#  Copyright 2002-2023 Barcelona Supercomputing Center (www.bsc.es)

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""This module serves as a comprehensive data generator based on the entropy
of different regions of space. The main objectives are to produce samples
and cases from a given set of dimensions and then evaluate these cases to
determine their stability. Parallel execution is used to evaluate the
stability of each case.
"""
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from scipy.stats import qmc
from sklearn.preprocessing import StandardScaler

from .utils import check_dims, flatten_list, get_dimension
from .dimensions import Cell, Dimension
from .viz import plot_divs, plot_stabilities, plot_importances_and_divisions
from ..tests.utils import unique

try:
    from pycompss.api.task import task
    from pycompss.api.api import compss_wait_on
except ImportError:
    from datagen.dummies.task import task
    from datagen.dummies.api import compss_wait_on


@task(returns=3)
def explore_cell(func, n_samples, entropy, depth, ax, dimensions,
                 cases_heritage_df, dims_heritage_df, use_sensitivity,
                 max_depth, divs_per_cell, generator):
    """Explore every cell in the algorithm while its delta entropy is positive.
    It receives a dataframe (cases_df) and an entropy from its parent, and
    calculates own delta entropy.
    If delta entropy is positive, the cell will subdivide itself according to
    the divisions assigned to each dimension.
    Otherwise, it will return the cases and logs taken until this point.


    :param func: Objective function
    :param n_samples: Number of samples to produce
    :param entropy: Entropy of the father calculated from the cases that fits
    into this cell's space
    :param depth: Recursivity depth
    :param ax: Plottable object
    :param dimensions: Cell dimensions
    :param cases_heritage_df: Inherited cases dataframe
    :param dims_heritage_df: Inherited dims dataframe
    :param use_sensitivity: Boolean indicating whether sensitivity analysis is
    used or not
    :param max_depth: Maximum recursivity depth (it won't subdivide itself if
    exceeded)
    :param dims_heritage_df: Inherited dims dataframe
    :param divs_per_cell: Number of resultant cells from each recursive call
    :param generator: Numpy generator for random values
    :return children_total: List of children dimensions, entropy,
    delta_entropy and depth
    :return cases_df: Concatenation of inherited cases and those produced by
    the cell
    :return dims_df: Concatenation of inherited dims and those produced by
    the cell
    """
    print("New cell", flush=True)
    # Generate samples (n_samples for each dimension)
    samples_df = gen_samples(n_samples, dimensions, generator)
    # Generate cases (n_cases (attribute of the class Dimension) for each dim)
    cases_df, dims_df = gen_cases(samples_df, dimensions, generator)

    # Eval each case
    stabilities = [eval_stability(case, func) for case in cases_df.values]
    stabilities = compss_wait_on(stabilities)
    cases_df["Stability"] = stabilities

    # Add rectangle to plot axes representing cell borders
    if ax is not None and len(dimensions) == 2:
        plot_stabilities(ax, cases_df, dims_df)

    cases_df = pd.concat([cases_df, cases_heritage_df], ignore_index=True)
    dims_df = pd.concat([dims_df, dims_heritage_df], ignore_index=True)

    entropy, delta_entropy = eval_entropy(stabilities, entropy)

    # Finish recursivity if entropy decreases or cell become too small
    if delta_entropy < 0 or not check_dims(dimensions) or depth >= max_depth:
        print("Stopped cell:")
        print("    Entropy: ", entropy)
        print("    Delta entropy: ", delta_entropy)
        print("    Depth: ", depth)
        return (dimensions, entropy, delta_entropy, depth), cases_df, dims_df
    else:
        if use_sensitivity:
            dimensions = sensitivity(cases_df, dimensions, divs_per_cell, generator)
        children_grid = gen_grid(dimensions)

        if ax is not None and len(dimensions) == 2:
            plot_divs(ax, children_grid)

        cases_df, dims_df, children_total = (
            explore_grid(ax, cases_df, children_grid, depth, dims_df, func,
                         n_samples, use_sensitivity, max_depth, divs_per_cell,
                         generator))
        return children_total, cases_df, dims_df


def explore_grid(ax, cases_df, grid, depth, dims_df, func, n_samples,
                 use_sensitivity, max_depth, divs_per_cell, generator):
    """
    For a given grid (children grid) and cases taken, this function is in
    charge of distributing those samples among those cells and, finally,
    retrieving its results.

    :param ax: Plottable object
    :param cases_df: Concatenation of inherited cases and those produced by
    the cell
    :param grid: Children grid
    :param depth: Recursivity depth
    :param dims_df: Samples dataframe(one for each case)
    :param func: Objective function
    :param n_samples: Number of samples to produce
    :param use_sensitivity: Boolean indicating whether sensitivity analysis is
    used or not
    :param max_depth: Maximum recursivity depth (it won't subdivide itself if
    exceeded)
    :param divs_per_cell: Number of resultant cells from each recursive call
    :param generator: Numpy generator for random values
    :return: Children dims, cases and parameters
    """
    total_cases_df, total_dims_df, total_entropies = get_children_parameters(
        grid, dims_df, cases_df)
    children_total_params = []
    list_cases_children_df = []
    list_dims_children_df = []
    for children_cell, cases_heritage_df, dims_heritage_df, entropy_children \
            in zip(grid, total_cases_df, total_dims_df, total_entropies):
        dim = children_cell.dimensions
        child_total_params, cases_children_df, dims_children_df = (
            explore_cell(func, n_samples, entropy_children, depth + 1, ax, dim,
                         cases_heritage_df, dims_heritage_df, use_sensitivity,
                         max_depth, divs_per_cell, generator))
        children_total_params.append(child_total_params)
        list_cases_children_df.append(cases_children_df)
        list_dims_children_df.append(dims_children_df)
    children_total_params = compss_wait_on(children_total_params)
    list_cases_children_df = compss_wait_on(list_cases_children_df)
    list_dims_children_df = compss_wait_on(list_dims_children_df)
    cases_df = pd.concat(list_cases_children_df, ignore_index=True)
    dims_df = pd.concat(list_dims_children_df, ignore_index=True)
    children_total_params = flatten_list(children_total_params)
    return cases_df, dims_df, children_total_params


def generate_columns(label, dim):
    """Assigns names for every variable in a dimension.

    :param dim: Involved dimension
    :return: Names of de variables
    """
    return [f"{label}_Var{v}" for v in range(len(dim.variables))]


def process_p_cig_dimension(samples_df, p_cig, generator):
    """ Assigns values to g_for and g_fol dimensions.

    p_cig samples values must be distributed between g_for and g_fol assigning
    a random value between 0 and 1 as a one-to-one percentage, resulting in
    g_for plus g_fol equalling p_cig. g_for variables are calculated as usual,
    while g_fol variables are complimentary to g_for to sum g_fol:
        g_fol_i = p_cig_i - g_for_i

    :param generator:
    :param samples_df: Involved samples
    :param p_cig: p_cig dimension
    :return: Cases obtained and samples extended (one sample for each case)
    """
    cases = []
    dims = []

    for _, sample in samples_df.iterrows():
        # Obtain p_cig cases
        cases_p_cig_df = pd.DataFrame(
            p_cig.get_cases_extreme("p_cig", sample["p_cig"], generator),
            columns=generate_columns("p_cig", p_cig)).dropna()
        n_rows = len(cases_p_cig_df)
        dims_p_cig_df = pd.DataFrame(
            np.repeat(sample["p_cig"], n_rows), columns=["p_cig"])

        # Obtain the complimentary g_for and g_fol percentages
        grid_forming_perc = sample["perc_g_for"]
        g_for_sample = sample["p_cig"] * grid_forming_perc
        g_fol_sample = sample["p_cig"] - g_for_sample

        # Obtain g_for and g_fol cases
        cases_g_for = []
        cases_g_fol = []
        dims_g_for = []
        dims_g_fol = []
        for i in range(len(cases_p_cig_df)):
            # Compose g_for dimension
            # Pick bounds of each variable. The min value is p_cig dimension's
            # min bound, and max is the value sampled for ith p_cig's variable
            g_for_variables = np.array([
                (p_cig.variables[x, 0], cases_p_cig_df.iloc[i, x])
                for x in range(len(p_cig.variables))])
            g_for = Dimension(variables=g_for_variables, n_cases=1, divs=1,
                              borders=(p_cig.borders[0], sample["p_cig"]),
                              is_true_dimension=False, tolerance=p_cig.tolerance)
            # Create g_for case
            case_g_for = (g_for.get_cases_extreme("g_for", g_for_sample,
                                                  generator))[0]
            if not np.isnan(case_g_for).any():
                dims_g_for.append(g_for_sample)
                cases_g_for.append(case_g_for)
                dims_g_fol.append(g_fol_sample)
                # Compose g_fol subtracting p_cig from g_for case variables
                cases_g_fol.append(
                    [cases_p_cig_df.iloc[i, x] - case_g_for[x]
                     for x in range(len(p_cig.variables))])

        cases_g_for_df = pd.DataFrame(
            cases_g_for,
            columns=[f"g_for_Var{v}" for v in range(len(p_cig.variables))])
        dims_g_for_df = pd.DataFrame(dims_g_for, columns=["g_for"])
        cases_g_fol_df = pd.DataFrame(
            cases_g_fol,
            columns=[f"g_fol_Var{v}" for v in range(len(p_cig.variables))])
        dims_g_fol_df = pd.DataFrame(dims_g_fol, columns=["g_fol"])

        # Error check
        check_sum = (cases_g_fol_df.sum(axis=1) + cases_g_for_df.sum(axis=1)
                     - cases_p_cig_df.sum(axis=1)).to_numpy()
        if not np.isclose(check_sum, 0).all():
            raise ValueError("Sum of g_for and g_fol must equal p_cig")

        # Concat p_cig, g_for and g_fol into a complete case dataframe
        sample_cases_df = pd.concat(
            [cases_p_cig_df, cases_g_for_df, cases_g_fol_df], axis=1)
        sample_dims_df = pd.concat(
            [dims_p_cig_df, dims_g_for_df, dims_g_fol_df], axis=1)

        # Concat samples and cases of p_cig, g_for and g_fol
        sample_cases_df = sample_cases_df.dropna().reset_index(drop=True)
        sample_dims_df = sample_dims_df.dropna().reset_index(drop=True)

        cases.append(sample_cases_df)
        dims.append(sample_dims_df)

    cases_df = pd.concat(cases, axis=0, ignore_index=True)
    dims_df = pd.concat(dims, axis=0, ignore_index=True)
    return cases_df, dims_df


def process_other_dimensions(samples_df, label, dim, generator):
    """
    This method assigns values to the variables within a generic dimension.

    :param generator:
    :param samples_df: Dataframe containing every sample in this cell
    :param dim: Involved dimension
    :return: Cases obtained and samples extended (one sample for each case)
    """
    total_cases = []
    total_dim = []
    for _, sample in samples_df.iterrows():
        cases = dim.get_cases_extreme(label, sample[label], generator)
        for case in cases:
            if not np.isnan(case).any():
                total_cases.append(case)
                total_dim.append(sample[label])

    dims_df = pd.DataFrame(total_dim, columns=[label])
    cases_df = pd.DataFrame(total_cases, columns=generate_columns(label, dim))
    return cases_df, dims_df


def gen_cases(samples_df, dimensions, generator):
    """Produces sum combinations of the samples given. Each sample sum
    combination is called a "case".

    :param generator:
    :param samples_df: Involved samples (dataframe)
    :param dimensions: Involved dimensions
    :return cases_df: Samples-driven produced cases dataframe
    :return dims_df: Samples dataframe(one for each case)
    """
    total_cases = []
    total_dims = []

    for label, dim in dimensions.items():
        if label != "p_load":
            if label == "p_cig":
                partial_cases, partial_dims = process_p_cig_dimension(samples_df,
                                                                      dim, generator)
            else:
                partial_cases, partial_dims = process_other_dimensions(samples_df,
                                                                       label, dim,
                                                                       generator)
            total_cases.append(partial_cases)
            total_dims.append(partial_dims)
    if "p_load" in dimensions:
        partial_cases, partial_dims = (process_other_dimensions
                                       (samples_df,"p_load",
                                        dimensions["p_load"], generator))
        total_cases.append(partial_cases)
        total_dims.append(partial_dims)

    total_cases_df = pd.concat(total_cases, axis=1)
    total_dims_df = pd.concat(total_dims, axis=1)
    return total_cases_df, total_dims_df


def gen_samples(n_samples, dimensions, generator):
    """Generates n_samples samples, which represent total sum of the variables
    within a dimension.

    :param generator:
    :param n_samples: Number of samples to produce
    :param dimensions: Involved dimensions
    :return: DataFrame containing these samples with columns named after
    dimension labels
    """
    sampler = qmc.LatinHypercube(d=len(dimensions), seed=generator)
    samples = sampler.random(n=n_samples)

    lower_bounds = np.array([dim.borders[0] for _,dim in dimensions.items()])
    upper_bounds = np.array([dim.borders[1] for _,dim in dimensions.items()])

    samples_scaled = lower_bounds + samples * (upper_bounds - lower_bounds)

    df_samples = pd.DataFrame(samples_scaled,
                              columns=[label for label in dimensions])

    return df_samples


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
    labels = unique(col.rsplit('_Var')[0]
                    for col in cases_df.columns if '_Var' in col)
    dims_df = pd.DataFrame()
    for label in labels:
        matching_columns = (
            cases_df.filter(regex=r'^' + label + r'_*', axis=1).sum(axis=1))
        dims_df[label] = matching_columns
    dims_df.columns = labels
    x = np.array(dims_df)
    y = np.array(cases_df["Stability"])
    y = y.astype('int')

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    random_state = generator.integers (0,2**32 - 1)
    model = RandomForestClassifier(random_state=random_state)
    model.fit(x_scaled, y)

    importances = model.feature_importances_
    for _, dim in dimensions.items():
        dim.divs = 1
    splits_per_cell = int(np.round(np.log2(divs_per_cell)))

    for _ in range(splits_per_cell):
        # plot_importances_and_divisions(dimensions, importances)
        index_max_importance = np.argmax(importances)
        label_max_importance = list(labels)[index_max_importance]
        dim_max_importance = get_dimension(label_max_importance, dimensions)

        if ((dim_max_importance.borders[1] - dim_max_importance.borders[0]) /
                dim_max_importance.divs < dim_max_importance.tolerance):
            importances[index_max_importance] = 0

        if importances[index_max_importance] != 0:
            dim_max_importance.divs *= 2
            importances[index_max_importance] /= 2
    # plot_importances_and_divisions(dimensions, importances)
    return dimensions


@task(returns=1)
def eval_stability(case, f):
    """Call objective function and return its result.

    :param case: Involved cases
    :param f: Objective function
    :return: Result of the evaluation
    """
    return f(case)


def gen_grid(dims):
    """
    Generate grid. Every cell is made out of a list of the Dimension objects
    involved in the problem, with the only difference that the lower and upper
    bounds change for each cell.

    :param dims: Involved dimensions
    :return: Grid
    """
    n_dims = len(dims)
    ini = tuple(dim.borders[0] for dim in dims.values())
    fin = tuple(dim.borders[1] for dim in dims.values())
    div = tuple(dim.divs for dim in dims.values())
    total_div = np.prod(div)
    grid = []
    for i in range(total_div):
        div_indices = np.unravel_index(i, div)
        lower = [
            ini[j] + (fin[j] - ini[j]) / div[j] * div_indices[j]
            for j in range(n_dims)]
        upper = [
            ini[j] + (fin[j] - ini[j]) / div[j] * (div_indices[j] + 1)
            for j in range(n_dims)
        ]
        dimensions = {}
        for j, (label, dim) in enumerate(dims.items()):
            dimensions[label] = Dimension(variables=dim.variables,
                                          n_cases=dim.n_cases, divs=dim.divs,
                                          borders=(lower[j], upper[j]),
                                          is_true_dimension=dim.is_true_dimension,
                                          tolerance=dim.tolerance)
        grid.append(Cell(dimensions))
    return grid


def calculate_entropy(freqs):
    """Obtain cell entropy from stability and non-stability frequencies.

    :param freqs: two-element list with the frequency (1-based) of stable and
    non-stable cases, respectively
    :return: Entropy
    """
    cell_entropy = 0
    for i in range(len(freqs)):
        if freqs[i] != 0:
            cell_entropy = cell_entropy - freqs[i] * np.log(freqs[i])
    return cell_entropy


def eval_entropy(stabilities, entropy_parent):
    """Calculate entropy of the cell using its list of stabilities.

    :param stabilities: List of stabilities (result of the evaluation of every
    case)
    :param entropy_parent: Parent entropy based on concrete cases (those which
    correspond to the cell)
    :return: Entropy and delta entropy
    """
    freqs = []
    counter = 0
    for stability in stabilities:
        if stability == 1:
            counter += 1
    freqs.append(counter / len(stabilities))
    freqs.append((len(stabilities) - counter) / len(stabilities))
    entropy = calculate_entropy(freqs)
    if entropy_parent is None:
        delta_entropy = 1
    else:
        delta_entropy = entropy - entropy_parent
    return entropy, delta_entropy


def get_children_parameters(children_grid, dims_heritage_df, cases_heritage_df):
    """Obtains dimensions, cases_df, entropy and delta_entropy of each child

    :param children_grid: Grid to obtain parameters
    :param dims_heritage_df: Samples dataframe(one for each case)
    :param cases_heritage_df: Inherited cases dataframe
    :return: List of children (with these attributes set)
    """
    total_cases = []
    total_dims = []
    total_entropies = []
    for cell in children_grid:
        dims = []
        cases = []
        for idx, row in dims_heritage_df.iterrows():
            # Cell dimensions don't include g_for and g_fol, but dims_df do
            if 'g_for' in row.index and 'g_fol' in row.index:
                if not isinstance(row, pd.Series):
                    raise TypeError("Row is not a pd.Series object")
                row = row.drop(labels=['g_for', 'g_fol'])

            # Check if every dimension in row is within cell borders
            cell_borders = [dims.borders
                            for _,dims in cell.dimensions.items()]
            belongs = all(cell_borders[t][0] <= row.iloc[t] <= cell_borders[t][1]
                          for t in range(len(cell.dimensions)))
            if belongs:
                cases.append(cases_heritage_df.iloc[[idx], :])
                dims.append(dims_heritage_df.iloc[[idx], :])

        if cases and dims:
            stabilities = [int(case["Stability"].iloc[0]) for case in cases]
            entropy, _ = eval_entropy(stabilities, None)
            cases_df = pd.concat(cases, axis=0, ignore_index=True)
            dims_df = pd.concat(dims, axis=0, ignore_index=True)
        else:
            entropy = None
            cases_df = pd.DataFrame()
            dims_df = pd.DataFrame()
        total_cases.append(cases_df)
        total_dims.append(dims_df)
        total_entropies.append(entropy)

    if cases_heritage_df is None:
        cases_heritage_df = pd.DataFrame()
    if sum([len(cases) for cases in total_cases]) != len(cases_heritage_df):
        raise Exception("Not every case was assigned to a child")
    return total_cases, total_dims, total_entropies
