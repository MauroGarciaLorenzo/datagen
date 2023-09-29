from utils import check_dims, flatten_list
from classes import Cell, Dimension, Case
import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.task import task
from scipy.stats import qmc
import pandas as pd

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def getLastChildren(grid, last_children):
    for cell in grid:
        if cell.children == []:
            last_children.append(cell)
        else:
            last_children.extend(getLastChildren(cell.children, []))
    return last_children


@task(returns=1)
def explore_cell(
    func, n_samples, entropy, tolerance, depth, cell, ax, dimensions, cases
):
    # f = compss_wait_on(f)
    # cases_df_total = pd.DataFrame()
    samples = gen_samples(
        n_samples, dimensions
    )  # generate first samples (n_samples for each dimension)
    # plot_sample(ax, samples[:,0], samples[:,1], samples[:,2])

    # generate cases (n_cases(attribute of the class Dimension) for each dim)
    cases_df, dims_df = gen_cases(samples, n_samples, dimensions)
    # cases_df_total=pd.concat([cases_df_total,case_df],axis=0)

    # eval each case
    stabilities = []
    for row in range(len(cases_df)):
        stabilities.append(eval_stability(cases_df.iloc[row, :], func))
    cases_df["Stability"] = stabilities

    # eval entropy. Save entropy and delta_entropy as an attribute of the class
    #  Cell
    entropy, delta_entropy = eval_entropy(stabilities, entropy)  

    ###############################################################################
    if delta_entropy < 0 or not check_dims(dimensions, tolerance):
        return dimensions, cases, entropy, delta_entropy, depth
    else:
        # new_divs = sensitivity(cases)

        # for i in range(len(new_divs)):
        #    dimensions[i].divs = new_divs[i]

        children = gen_children(n_samples, dimensions, entropy, cases)
        div = tuple(dim.divs for dim in dimensions)
        total_div = np.prod(div)
        children_total = [None] * len(children)
        for cell_child in range(len(children_total)):
            dim = children[cell_child][0]
            sub = children[cell_child][1]
            ent = children[cell_child][2]

            children_total[cell_child] = explore_cell(
                func, n_samples, ent, tolerance, depth + 1, cell, ax, dim, sub
            )

        for cell_child in range(len(children_total)):
            children_total[cell_child] = compss_wait_on(children_total[cell_child])

        children_total = flatten_list(children_total)
        return children_total


# generate n_samples samples for each dim
def gen_samples(n_samples, dimensions):
    samples = []

    sampler = qmc.LatinHypercube(d=len(dimensions))
    samples = sampler.random(n=n_samples)

    # for _ in range(n_samples):
    # sample = []
    samples_scaled = np.zeros([n_samples, len(dimensions)])
    samples_scaled_s = np.zeros([1, len(dimensions)])

    for s in range(n_samples):
        samples_s = samples[s, :]

        for d in range(len(dimensions)):
            dimension = dimensions[d]
            lower, upper = dimension.borders

            sample = samples_s[d]
            samples_scaled_s[0, d] = lower + sample * (upper - lower)
        # samples_scaled.append(samples_scaled_s)
        samples_scaled[s, :] = samples_scaled_s
    return samples_scaled


# receives first samples
def gen_cases(samples, n_samples, dimensions):
    samples_d = list(
        zip(*samples)
    )  # gets a list with samples split by dimension (one list for each dim)
    total_samples = pd.DataFrame()

    # for each dim, get cases and re join cases ([Dim1Vars, Dim2Vars, Dim3Vars, ... DimNVars])
    for d in range(len(dimensions)):
        total_samples_d = pd.DataFrame()

        for i in range(n_samples):
            cases_dim = dimensions[d].get_cases(samples_d[d][i])
            cases_dim_df = pd.DataFrame(cases_dim)
            columns = []
            for v in range(len(dimensions[d].variables)):
                columns.append("Dim" + str(d) + "_Var" + str(v))

            cases_dim_df.columns = columns
            cases_dim_df["Dim" + str(d)] = samples_d[d][i]

            total_samples_d = pd.concat([total_samples_d, cases_dim_df], axis=0)
        total_samples_d = total_samples_d.reset_index(drop=True)
        total_samples = pd.concat([total_samples, total_samples_d], axis=1)
    
    column_names_dims = ["Dim" + str(d) for d in range(len(dimensions))]
    cases_df = total_samples.drop(column_names_dims, axis=1)
    dims_df = total_samples[column_names_dims]
    
    return cases_df, dims_df


def sensitivity(cases):
    X = []
    y = []
    for s in cases:
        X.append(s.case_dim)
        y.append(s.stability)

    X = np.array(X)
    y = np.array(y)

    x_avg = np.mean(X, axis=0)
    x_min = np.min(X, axis=0)
    x_max = np.max(X, axis=0)

    model = LogisticRegression()
    model.fit(X, y)

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
    return f(case)


# Generates grid from Dimensions received
def gen_grid(dims):
    n_dims = len(dims)
    ini = tuple(dim.borders[0] for dim in dims)
    fin = tuple(dim.borders[1] for dim in dims)
    div = tuple(dim.divs for dim in dims)
    total_div = np.prod(div)
    grid = []
    for i in range(total_div):
        div_indices = np.unravel_index(i, div)
        lower = [
            ini[j] + (fin[j] - ini[j]) / div[j] * div_indices[j] for j in range(n_dims)
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
                )
            )
        grid.append(Cell(dimensions))
    return grid


def calculate_entropy(freqs):
    E = 0
    for ii in range(len(freqs)):
        if freqs[ii] != 0:
            E = E - freqs[ii] * np.log(freqs[ii])
    return E


# Gets entropy and delta_entropy.
# Saves entropy in entropy and delta_entropy in delta_entropy
def eval_entropy(stabilities, entropy):
    freqs = []
    cont = 0
    for stability in stabilities:
        stability = compss_wait_on(stability)
        if stability == 1:
            cont += 1
    freqs.append(cont / len(stabilities))
    freqs.append((len(stabilities) - cont) / len(stabilities))
    e = calculate_entropy(freqs)
    if entropy == None:
        delta_entropy = 1
    else:
        delta_entropy = e - entropy
    return e, delta_entropy


def gen_children(
    n_samples, dims, entropy, cases
):  # cases = grid[cell].cases
    n_dims = len(dims)
    ini = tuple(dim.borders[0] for dim in dims)
    fin = tuple(dim.borders[1] for dim in dims)
    div = tuple(dim.divs for dim in dims)
    total_div = np.prod(div)
    grid_children = []

    for i in range(total_div):
        div_indices = np.unravel_index(i, div)
        lower = [
            ini[j] + (fin[j] - ini[j]) / div[j] * div_indices[j] for j in range(n_dims)
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
                )
            )

        cases_list = []

        for s in cases:
            if all([s.case_dim[t] >= lower[t] for t in range(n_dims)]) and all(
                [s.case_dim[t] <= upper[t] for t in range(n_dims)]
            ):
                cases_list.append(s)

        entropy = None
        delta_entropy = None
        if len(cases_list) > 0:
            entropy, delta_entropy = eval_entropy(
                cases_list, None
            )  # eval entropy. Save entropy and delta_entropy as an attribute of the class Cell

        grid_children.append((dimensions, cases_list, entropy, delta_entropy))

    return grid_children
