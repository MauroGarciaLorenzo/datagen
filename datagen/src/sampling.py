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

"""
Data generator based on the entropy of different regions of space

Provides the functions needed in the algorithm
"""

import random

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from scipy.stats import qmc

from .utils import check_dims, flatten_list
from .classes import Cell, Dimension

try:
    from pycompss.api.task import task
    from pycompss.api.api import compss_wait_on
except ImportError:
    from datagen.dummies import task
    from datagen.dummies import compss_wait_on


def getLastChildren(grid, last_children):
    for cell in grid:
        if not cell.children:
            last_children.append(cell)
        else:
            last_children.extend(getLastChildren(cell.children, []))
    return last_children


@task(returns=2)
def explore_cell(
        func, n_samples, entropy, tolerance, depth, ax, dimensions,
        cases_heritage_df):
    """Explore every cell in the algorithm while its delta entropy is positive.
    It receives a dataframe (cases_df) and an entropy from its parent, and
    calculates own delta entropy.
    If delta entropy is positive, the cell will subdivide itself according to
    the divisions assigned to each dimension.
    Otherwise, it will return the cases and logs taken until this point.

    :param func: Objective function
    :param n_samples: number of samples to produce
    :param entropy: Entropy of the father calculated from the cases that fits
    into this cell's space
    :param tolerance: Maximum length of a dimension (it won't subdivide itself
    if exceeded)
    :param depth: Maximum recursivity depth (it won't subdivide itself if
    exceeded)
    :param ax: Plottable object
    :param dimensions: Cell dimensions
    :param cases_heritage_df: Inherited cases dataframe
    :return children_total: List of children dimensions, entropy,
    delta_entropy and depth
    :return cases_df: Concatenation of inherited cases and those produced by
    the cell
    """
    # Generate samples (n_samples for each dimension)
    samples_df = gen_samples(n_samples, dimensions)

    # Generate cases (n_cases (attribute of the class Dimension) for each dim)
    cases_df, dims_df = gen_cases(samples_df, dimensions)

    # Eval each case
    stabilities = []
    for i in range(len(cases_df)):
        stabilities.append(eval_stability(cases_df.iloc[i, :], func))
    stabilities = compss_wait_on(stabilities)
    cases_df["Stability"] = stabilities
    cases_df = pd.concat([cases_df, cases_heritage_df], ignore_index=True)

    entropy, delta_entropy = eval_entropy(stabilities, entropy)

    # Finish recursivity if entropy decreases or cell become too small
    if delta_entropy < 0 or not check_dims(dimensions, tolerance):
        return (dimensions, entropy, delta_entropy, depth), cases_df
    else:
        children_grid = gen_grid(dimensions)
        cases_df, children_total = explore_grid(ax, cases_df,
                                                children_grid, depth,
                                                dims_df, func, n_samples,
                                                tolerance)
        return children_total, cases_df


def explore_grid(ax, cases_df, grid, depth, dims_df, func,
                 n_samples, tolerance):
    total_cases_df, total_entropies = get_children_parameters(
        grid, dims_df, cases_df)
    children_total_params = []
    list_cases_children_df = []
    for children_cell, cases_heritage_df, entropy_children \
            in zip(grid, total_cases_df, total_entropies):
        dim = children_cell.dimensions
        child_total_params, cases_children_df = explore_cell(
            func,
            n_samples,
            entropy_children,
            tolerance,
            depth + 1,
            ax,
            dim,
            cases_heritage_df,
        )
        children_total_params.append(child_total_params)
        list_cases_children_df.append(cases_children_df)
    # implement reduction
    children_total_params = compss_wait_on(children_total_params)
    list_cases_children_df = compss_wait_on(list_cases_children_df)
    cases_df = pd.concat(list_cases_children_df, ignore_index=True)
    children_total_params = flatten_list(children_total_params)
    return cases_df, children_total_params


def generate_columns(dim):
    """Assigns names for every variable in a dimension.

    :param dim: Involved dimension
    :return: Names of de variables
    """
    return [f"{dim.label}_Var{v}" for v in range(len(dim.variables))]


def process_p_cig_dimension(samples_df, p_cig):
    """Gives value to g_for and g_fol dimension. p_cig samples values must be
    distributed between g_for and g_fol assigning a random value between 0 and
    1 as coefficient, giving g_for plus g_fol equals p_cig. g_for variables are
    calculated in a normal way, while g_fol variables are defined as:
        g_fol_i = p_cig_i - g_for_i

    :param samples_df: Involved samples
    :param p_cig: p_cig dimension
    :return:
    """
    cases_df = pd.DataFrame()
    dims_df = pd.DataFrame()

    for _, sample in samples_df.iterrows():
        # Obtain cases_p_cig
        cases_p_cig_df = pd.DataFrame(p_cig.get_cases(sample[p_cig.label]),
                                      columns=generate_columns(p_cig)).dropna()
        n_rows = cases_p_cig_df.shape[0]
        dims_p_cig_df = pd.DataFrame(
            {p_cig.label: np.repeat(sample[p_cig.label], n_rows)})

        # Obtain cases_g_for and g_fol
        grid_forming_perc = random.random()
        g_for_sample = sample["p_cig"] * grid_forming_perc
        g_fol_sample = sample["p_cig"] * (1 - grid_forming_perc)

        cases_g_for = []
        cases_g_fol = []

        dims_g_for = []
        dims_g_fol = []
        for i in range(len(cases_p_cig_df)):
            # create g_for_case
            g_for_variables = np.array([
                (p_cig.variables[x, 0], cases_p_cig_df.iloc[i, x])
                for x in range(len(p_cig.variables))])
            g_for = Dimension(
                variables=g_for_variables,
                n_cases=1,
                divs=1,
                lower=p_cig.borders[0],
                upper=sample[p_cig.label],
                label="g_for")
            case_g_for = (g_for.get_cases(g_for_sample))[0]
            if all(x is not None for x in case_g_for) and \
                    all(x is not np.nan for x in case_g_for):
                dims_g_for.append(g_for_sample)
                cases_g_for.append(case_g_for)
                dims_g_fol.append(g_fol_sample)
                cases_g_fol.append(
                    [cases_p_cig_df.iloc[i, x] - cases_g_for[i][x]
                     for x in range(len(p_cig.variables))])

        cases_g_for_df = pd.DataFrame(
            cases_g_for,
            columns=[f"g_for_Var{v}" for v in range(len(p_cig.variables))])
        dims_g_for_df = pd.DataFrame(dims_g_for, columns=["g_fol"])

        cases_g_fol_df = pd.DataFrame(
            cases_g_fol,
            columns=[f"g_fol_Var{v}" for v in range(len(p_cig.variables))])
        dims_g_fol_df = pd.DataFrame(dims_g_fol, columns=["g_fol"])
        # concat a concrete sample of p_cig, g_for and g_fol
        sample_cases_df = pd.concat(
            [cases_p_cig_df, cases_g_for_df, cases_g_fol_df], axis=1)
        sample_dims_df = pd.concat(
            [dims_p_cig_df, dims_g_for_df, dims_g_fol_df], axis=1)

        # concat samples of p_cig, g_for and g_fol
        cases_df = pd.concat([cases_df, sample_cases_df], axis=0,
                             ignore_index=True)
        dims_df = pd.concat([dims_df, sample_dims_df], axis=0,
                            ignore_index=True)

        not_na_index = cases_df.notna().all(axis=1)
        cases_df = cases_df.loc[not_na_index].reset_index(drop=True)
        dims_df = dims_df.loc[not_na_index].reset_index(drop=True)

    return cases_df, dims_df


def process_other_dimensions(samples_df, dim):
    total_cases = []
    total_dim = []
    for _, sample in samples_df.iterrows():
        cases = dim.get_cases(sample[dim.label])
        for case in cases:
            if (all(x is not
                    None for x in case) and
                    all(x is not np.nan for x in case)):
                total_cases.append(case)
                total_dim.append(sample[dim.label])

    dims_df = pd.DataFrame(total_dim, columns=[dim.label])
    cases_df = pd.DataFrame(total_cases, columns=generate_columns(dim))
    return cases_df, dims_df


def gen_cases(samples_df, dimensions):
    """Produces sum combinations of the samples given.

    :param samples_df: Involved samples (dataframe)
    :param dimensions: Involved dimensions
    :return cases_df: Samples-driven produced cases dataframe
    :return dims_df: Samples dataframe(one for each case)
    """
    total_cases_df = pd.DataFrame()
    total_dims_df = pd.DataFrame()

    for dim in dimensions:
        if dim.label == "p_cig":
            partial_cases_df, partial_dims_df = process_p_cig_dimension(
                samples_df, dim)
        else:
            partial_cases_df, partial_dims_df = process_other_dimensions(
                samples_df, dim)

        total_cases_df = pd.concat(
            [total_cases_df, partial_cases_df], axis=1)

        total_dims_df = pd.concat(
            [total_dims_df, partial_dims_df], axis=1)

    return total_cases_df, total_dims_df


def gen_samples(n_samples, dimensions):
    """Generates n_samples samples, which represent total sum of the variables
    within a dimension.

    :param n_samples: Number of samples to produce
    :param dimensions: Involved dimensions
    :return: DataFrame containing these samples with columns named after
    dimension labels
    """
    sampler = qmc.LatinHypercube(d=len(dimensions))
    samples = sampler.random(n=n_samples)
    samples_scaled = np.zeros([n_samples, len(dimensions)])

    for s in range(n_samples):
        samples_s = samples[s, :]
        for d in range(len(dimensions)):
            dimension = dimensions[d]
            lower, upper = dimension.borders
            sample = samples_s[d]
            samples_scaled[s, d] = lower + sample * (upper - lower)

    df_samples = pd.DataFrame(samples_scaled,
                              columns=[dim.label for dim in dimensions])
    return df_samples


def sensitivity(cases):
    """Sensitivity analysis. Decides which dimensions are more important to the
     decision.

    :param cases: Involved cases
    :return: Divisions for each dimension
    """
    x = []
    y = []
    for s in cases:
        x.append(s.case_dim)
        y.append(s.stability)
    x = np.array(x)
    y = np.array(y)
    x_avg = np.mean(x, axis=0)
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    model = LogisticRegression()
    model.fit(x, y)
    y_test = np.zeros([2, 1])
    std = np.zeros([len(x_avg), 1])
    for i in range(len(x_min)):
        x_test = np.copy(x_avg).reshape(1, -1)
        x_test[0, i] = x_min[i]
        y_test[0, 0] = model.predict(x_test)
        x_test[0, i] = x_max[i]
        y_test[1, 0] = model.predict(x_test)
        std[i] = np.std(y_test)

    dim_max_std = np.argmax(std)
    divs = [1, 1, 1]
    divs[dim_max_std] = 2
    return divs


# task defined (every case is going to be evaluated in parallel)
# funtion is received as a parameter
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
    ini = tuple(dim.borders[0] for dim in dims)
    fin = tuple(dim.borders[1] for dim in dims)
    div = tuple(dim.divs for dim in dims)
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
        dimensions = []
        for j in range(len(dims)):
            dimensions.append(
                Dimension(
                    dims[j].variables,
                    dims[j].n_cases,
                    dims[j].divs,
                    lower[j],
                    upper[j],
                    dims[j].label
                )
            )
        grid.append(Cell(dimensions))
    return grid


def calculate_entropy(freqs):
    """Obtain cell entropy from stability and non-stability frequencies.

    :param freqs: two-element list with the frequency (1-based) of stable and
    non-stable cases, respectively.
    :return: Entropy.
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
        stability = compss_wait_on(stability)
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


def get_children_parameters(children_grid, dims_df, cases_heritage_df):
    """Obtains dimensions, cases_df, entropy and delta_entropy of each child

    :param children_grid: Grid to obtain parameters
    :param dims_df: Samples dataframe(one for each case)
    :param cases_heritage_df: Inherited cases dataframe
    :return: List of children (with these attributes set)
    """
    total_cases_df = []
    total_entropies = []
    for cell in children_grid:
        if cases_heritage_df is not None:
            cases_df = pd.DataFrame(columns=cases_heritage_df.columns)
        else:
            cases_df = pd.DataFrame()

        for k in range(len(dims_df)):
            row = dims_df.iloc[k, :]
            if (all([row[t] >= cell.dimensions[t].borders[0]
                    for t in range(len(cell.dimensions))]) and
                    all([row[t] <= cell.dimensions[t].borders[0]
                         for t in range(len(cell.dimensions))])):
                cases_df = pd.concat(
                    [cases_df, cases_heritage_df.iloc[[k], :]],
                    ignore_index=True)

        entropy = None
        if len(cases_df) > 0:
            entropy, _ = eval_entropy(
                cases_df["Stability"], None
            )

        total_cases_df.append(cases_df)
        total_entropies.append(entropy)

    return total_cases_df, total_entropies
