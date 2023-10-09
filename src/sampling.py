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


from sklearn.linear_model import LogisticRegression
from utils import check_dims, flatten_list
from classes import Cell, Dimension
from pycompss.api.api import compss_wait_on
from pycompss.api.task import task
from scipy.stats import qmc
import numpy as np
import pandas as pd
import random

"""Data generator based on the entropy of different regions of space

Provides the functions needed in the algorithm
"""


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
    samples_df = gen_samples(
        n_samples, dimensions
    )  # generate first samples (n_samples for each dimension)

    # generate cases (n_cases(attribute of the class Dimension) for each dim)
    cases_df, dims_df = gen_cases(samples_df, dimensions)

    # eval each case
    stabilities = []
    for row in range(len(cases_df)):
        stabilities.append(eval_stability(cases_df.iloc[row, :], func))
    stabilities = compss_wait_on(stabilities)
    cases_df["Stability"] = stabilities
    cases_df = pd.concat([cases_df, cases_heritage_df], ignore_index=True)

    entropy, delta_entropy = eval_entropy(stabilities, entropy)

    if delta_entropy < 0 or not check_dims(dimensions, tolerance):
        return (dimensions, entropy, delta_entropy, depth), cases_df
    else:
        children = gen_grid_children(dimensions, dims_df, cases_df)
        children_total = [None] * len(children)
        list_cases_children_df = []
        for cell_child in range(len(children_total)):
            dim = children[cell_child][0]
            cases_heritage_df = children[cell_child][1]
            ent = children[cell_child][2]
            children_total[cell_child], cases_children_df = explore_cell(
                func,
                n_samples,
                ent,
                tolerance,
                depth + 1,
                ax,
                dim,
                cases_heritage_df,
            )
            list_cases_children_df.append(cases_children_df)

        # implement reduction
        children_total = compss_wait_on(children_total)
        list_cases_children_df = compss_wait_on(list_cases_children_df)
        cases_df = pd.concat(list_cases_children_df, ignore_index=True)

        children_total = flatten_list(children_total)
        return children_total, cases_df


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
    :param dim: g_for dimension
    :return:
    """
    results = []
    for _, sample in samples_df.iterrows():
        # Obtain cases_p_cig
        cases_p_cig = pd.DataFrame(p_cig.get_cases(sample[p_cig.label]))
        cases_p_cig.columns = generate_columns(p_cig)
        cases_p_cig[p_cig.label] = sample[p_cig.label]
        results.append(cases_p_cig)

        # Obtain cases_g_for
        coefficient = random.random()
        g_for_sample = sample["p_cig"] * coefficient
        cases_g_for = pd.DataFrame()
        
        for i in range(len(cases_p_cig)):
            g_for_variables = [
                (p_cig.variables[x][0], cases_p_cig.iloc[i, x])
                for x in range(len(p_cig.variables))]
            g_for = Dimension(
                variables=g_for_variables,
                n_cases=1,
                divs=1,
                lower=p_cig.borders[0],
                upper=sample[p_cig.label],
                label="g_for")
            case_g_for = pd.DataFrame(g_for.get_cases(g_for_sample))
            case_g_for.columns = generate_columns(g_for)
            case_g_for[g_for.label] = g_for_sample
            cases_g_for = pd.concat([cases_g_for, case_g_for], axis=1)

    return pd.concat(results).reset_index(drop=True)
"""
        coefficient = random.random()
        g_for_sample = sample["p_cig"] * coefficient
        g_fol_sample = sample["p_cig"] * (1 - coefficient)

        cases_dim_g_for_df = pd.DataFrame(dim.get_cases(g_for_sample))
        cases_dim_g_for_df.columns = generate_columns(dim)
        cases_dim_g_for_df[dim.label] = g_for_sample

        cases_dim_g_fol = [
            cases_dim_p_cig_df.iloc[row_idx, x] - cases_dim_g_for_df.iloc[0, x]
            for x in range(cases_dim_g_for_df.shape[1])]
        cases_dim_g_fol_df = pd.DataFrame([cases_dim_g_fol])
        cases_dim_g_fol_df.columns = [
            f"g_fol_Var{v}" for v in range(cases_dim_g_fol_df.shape[1])]

        cases_dim_g_fol_df["g_fol"] = g_fol_sample

        combined_df = pd.concat([cases_dim_g_for_df, cases_dim_g_fol_df],
                                axis=1)
        results.append(combined_df)
"""


def process_other_dimensions(samples_df, dim):
    results = []
    for _, sample in samples_df.iterrows():
        cases_dim = pd.DataFrame(dim.get_cases(sample[dim.label]))
        cases_dim.columns = generate_columns(dim)
        cases_dim[dim.label] = sample[dim.label]
        results.append(cases_dim)

    return pd.concat(results).reset_index(drop=True)


def gen_cases(samples_df, dimensions):
    """Produces sum combinations of the samples given.

    Assume that if g_for exists, p_cig must be before g_for

    :param samples_df: Involved samples (dataframe)
    :param dimensions: Involved dimensions
    :return cases_df: Samples-driven produced cases dataframe
    :return dims_df: Samples dataframe(one for each case)
    """
    total_samples_df = pd.DataFrame()

    for dim in dimensions:
        if dim.label == "p_cig":
            total_samples_dim_df = process_p_cig_dimension(
                samples_df, dim)

        else:
            total_samples_dim_df = process_other_dimensions(samples_df, dim)

        total_samples_df = pd.concat(
            [total_samples_df, total_samples_dim_df], axis=1)

    column_names_dims = [dim.label for dim in dimensions]
    if 'g_fol' in total_samples_df:
        column_names_dims.append('g_fol')
    cases_df = total_samples_df.drop(column_names_dims, axis=1)
    dims_df = total_samples_df[column_names_dims]

    return cases_df, dims_df


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
    """Generate initial grid

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
    """Obtain cell entropy from stability and non-stability frequencies

    :param freqs: Stable and non-stable cases
    :return: Entropy
    """
    e = 0
    for ii in range(len(freqs)):
        if freqs[ii] != 0:
            e = e - freqs[ii] * np.log(freqs[ii])
    return e


def eval_entropy(stabilities, entropy):
    """Calculate entropy of the cell using its list of stabilities.

    :param stabilities: List of stabilities (result of the evaluation of every
    case)
    :param entropy: Parent entropy based on concrete cases (those which
    correspond to the cell)
    :return: Entropy and delta entropy
    """
    freqs = []
    cont = 0
    for stability in stabilities:
        stability = compss_wait_on(stability)
        if stability == 1:
            cont += 1
    freqs.append(cont / len(stabilities))
    freqs.append((len(stabilities) - cont) / len(stabilities))
    e = calculate_entropy(freqs)
    if entropy is None:
        delta_entropy = 1
    else:
        delta_entropy = e - entropy
    return e, delta_entropy


def gen_grid_children(dims, dims_df, cases_heritage_df):
    """Obtains dimensions, cases_df, entropy and delta_entropy of each child

    :param dims: Cell dimensions
    :param dims_df: Samples dataframe(one for each case)
    :param cases_heritage_df: Inherited cases dataframe
    :return: List of children (with these attributes set)
    """
    n_dims = len(dims)
    ini = tuple(dim.borders[0] for dim in dims)
    fin = tuple(dim.borders[1] for dim in dims)
    div = tuple(dim.divs for dim in dims)
    total_div = np.prod(div)
    grid_children = []

    for i in range(total_div):
        div_indices = np.unravel_index(i, div)
        lower = [
            ini[j] + (fin[j] - ini[j]) / div[j] * div_indices[j] for j in
            range(n_dims)
        ]
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

        cases_df = pd.DataFrame(columns=cases_heritage_df.columns)
        for k in range(len(dims_df)):
            row = dims_df.iloc[k, :]
            if all([row[t] >= lower[t] for t in range(n_dims)]) and all(
                    [row[t] <= upper[t] for t in range(n_dims)]
            ):
                cases_df = pd.concat(
                    [cases_df, cases_heritage_df.iloc[[k], :]],
                    ignore_index=True)

        entropy = None
        delta_entropy = None
        if len(cases_df) > 0:
            entropy, delta_entropy = eval_entropy(
                cases_df["Stability"], None
            )

        grid_children.append((dimensions, cases_df, entropy, delta_entropy))

    return grid_children
