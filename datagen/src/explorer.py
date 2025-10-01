import os

import pandas as pd
import logging

from datagen.src.dimensions import Cell
from datagen.src.file_io import log_cell_info, save_df

logger = logging.getLogger(__name__)

try:
    from pycompss.api.task import task
    from pycompss.api.api import compss_wait_on
    from pycompss.api.constraint import constraint
except ImportError:
    from datagen.dummies.task import task
    from datagen.dummies.api import compss_wait_on
    from datagen.dummies.constraint import constraint

from datagen.src.utils import check_dims
from datagen.src.data_ops import flatten_list, concat_df_dict
from datagen.src.viz import plot_stabilities, plot_divs
from datagen.src.grid import gen_grid
from datagen.src.sensitivity_analysis import sensitivity
from datagen.src.case_generation import gen_samples, gen_cases
from datagen.src.evaluator import eval_entropy, eval_stability


@constraint(is_local=True)
@task(returns=1, on_failure='FAIL', priority=True)
def explore_cell(func, n_samples, parent_entropy, depth, ax, dimensions,
                 use_sensitivity, max_depth, divs_per_cell, generator,
                 feasible_rate, func_params, entropy_threshold, chunk_size,
                 delta_entropy_threshold, dst_dir=None, cell_name="",
                 ):
    """Explore every cell in the algorithm while its delta entropy is positive.
    It receives a dataframe (cases_df) and an entropy from its parent, and
    calculates own delta entropy.
    If delta entropy is positive, the cell will subdivide itself according to
    the divisions assigned to each dimension.
    Otherwise, it will return the cases and logs taken until this point.


    :param func: Objective function
    :param n_samples: Number of samples to produce
    :param parent_entropy: Entropy of the parent calculated from the cases that
    fits into this cell's space
    :param depth: Recursivity depth
    :param ax: Plottable object
    :param dimensions: Cell dimensions
    :param use_sensitivity: Boolean indicating whether sensitivity analysis is
    used or not
    :param max_depth: Maximum recursivity depth (it won't subdivide itself if
    exceeded)
    :param dims_heritage_df: Inherited independent_dims dataframe
    :param divs_per_cell: Number of resultant cells from each recursive call
    :param generator: Numpy generator for random values
    :param cell_name: Name of the current cell for logging purposes
    :return children_total: List of children dimensions, entropy,
    delta_entropy and depth
    """
    if not cell_name:
        cell_name = "0"
    logger.info(f"Entering cell {cell_name}")
    # Generate samples (n_samples for each dimension)
    samples_df = gen_samples(n_samples, dimensions, generator)
    # Generate cases (n_cases (attribute of the class Dimension) for each dim)
    cases_df, dims_df = gen_cases(samples_df, dimensions, generator)

    cases_df['cell_name'] = cell_name

    stabilities = []
    feasible_cases = 0
    stabilities_chunk = []
    output_dataframes_chunk = []
    initial_index = 0
    index = 0

    for _, case in cases_df.iterrows():
        stability, output_dfs = eval_stability(
            case=case,
            f=func,
            func_params=func_params,
            dimensions=dimensions,
            generator=generator
        )

        stabilities_chunk.append(stability)
        output_dataframes_chunk.append(output_dfs)
        index += 1

        if index % chunk_size == 0 or index == len(cases_df):
            stabilities_chunk = compss_wait_on(stabilities_chunk)
            output_dataframes_chunk = compss_wait_on(output_dataframes_chunk)

            # update feasible cases
            for stability in stabilities_chunk:
                if stability >= 0:
                    feasible_cases += 1

            # assign chunk results back to cases_df
            cases_df.loc[initial_index:index - 1,
            "Stability"] = stabilities_chunk

            # save cases_df incrementally
            save_df(cases_df.iloc[initial_index:index], dst_dir, cell_name,
                    "cases_df")

            # build and save total_dataframes incrementally
            total_dataframes = None
            for output_dfs in output_dataframes_chunk:
                if total_dataframes:
                    total_dataframes = concat_df_dict(total_dataframes,
                                                      output_dfs)
                else:
                    total_dataframes = output_dfs

            if total_dataframes:
                labels_to_remove = []
                for label, df in total_dataframes.items():
                    if df is not None and type(df) is not pd.DataFrame:
                        labels_to_remove.append(label)
                for label in labels_to_remove:
                    total_dataframes.pop(label)

                for df_name, df in total_dataframes.items():
                    save_df(df, dst_dir, cell_name, df_name)

            # reset chunk buffers
            initial_index = index
            stabilities_chunk = []
            output_dataframes_chunk = []

    # dims_df is static, save once
    save_df(dims_df, dst_dir, cell_name, "dims_df")

    # Add rectangle to plot axes representing cell borders
    if ax is not None and len(dimensions) == 2:
        plot_stabilities(ax, cases_df, dims_df, dst_dir)

    parent_entropy, delta_entropy = eval_entropy(stabilities, parent_entropy) #(cases_df, cases_heritage_df)#

    total_cases = n_samples * dimensions[0].n_cases
    message = f"Depth={depth}, Entropy={parent_entropy}, Delta_entropy={delta_entropy}"
    log_cell_info(cell_name, depth, parent_entropy, delta_entropy, feasible_cases / total_cases,
                  1, dst_dir)
    logger.info(message)

    check_entropy = False
    if delta_entropy > delta_entropy_threshold or parent_entropy > entropy_threshold:
        check_entropy = True

    # Finish recursivity if entropy decreases or cell become too small
    if (not check_entropy or not check_dims(
            dimensions) or depth >= max_depth or
            feasible_cases / total_cases < feasible_rate):

        logger.info("Stopped cell:")
        logger.info(f"    Entropy: {parent_entropy}")
        logger.info(f"    Delta entropy: {delta_entropy}")
        logger.info(f"    Depth: {depth}")

        log_cell_info(cell_name, depth, parent_entropy, delta_entropy,
                      feasible_cases / total_cases,
                      0, dst_dir)

        children_info = [(dimensions, parent_entropy, delta_entropy, depth)]
        return children_info
    else:
        if use_sensitivity:
            if total_cases < 100:
                cell = Cell(dimensions)
                cases_heritage_df, dims_heritage_df = get_parent_samples(
                    dst_dir, cell_name)

                cases_heritage_df, dims_heritage_df = get_children_samples(
                    cases_heritage_df, cell, dims_heritage_df)
                cases_df = pd.concat([cases_df, cases_heritage_df],
                                     ignore_index=True)
            dimensions = sensitivity(cases_df, dimensions, divs_per_cell,
                                     generator)
        children_grid = gen_grid(dimensions)

        if ax is not None and len(dimensions) == 2:
            plot_divs(ax, children_grid, dst_dir)

        # Recursive case returns children info
        children_info = explore_grid(ax=ax, cases_df=cases_df, grid=children_grid,
                              depth=depth, dims_df=dims_df, func=func,
                              n_samples=n_samples,
                              use_sensitivity=use_sensitivity,
                              max_depth=max_depth, divs_per_cell=divs_per_cell,
                              generator=generator, feasible_rate=feasible_rate,
                              func_params=func_params,
                              parent_entropy=parent_entropy,
                              parent_name=cell_name,
                              dst_dir=dst_dir, chunk_size=chunk_size,
                              entropy_threshold=entropy_threshold,
                              delta_entropy_threshold=delta_entropy_threshold
                              )

        return children_info


def explore_grid(ax, cases_df, grid, depth, dims_df, func, n_samples,
                 use_sensitivity, max_depth, divs_per_cell, generator,
                 feasible_rate, func_params, parent_entropy,
                 parent_name, dst_dir, entropy_threshold, delta_entropy_threshold,
                 chunk_size):
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
    :param parent_entropy: Entropy of the parent cell
    :param parent_name: Name of the parent cell
    :return: Children parameters
    """
    children_info_all = []
    i = 0

    for children_cell in grid:
        i += 1
        cell_name = f"{parent_name}.{i}"

        cases_children_df, dims_children_df = get_children_samples(
            cases_df, children_cell, dims_df)

        if not cases_children_df.empty and "Stability" not in cases_children_df.columns:
            parent_entropy, _ = eval_entropy(cases_children_df["Stability"], None)
        else:
            parent_entropy = None

        # Get children_info from child cell
        children_info = explore_cell(
            func=func, n_samples=n_samples,
            parent_entropy=parent_entropy, depth=depth + 1,
            ax=ax, dimensions=children_cell.dimensions,
            use_sensitivity=use_sensitivity, max_depth=max_depth,
            divs_per_cell=divs_per_cell, generator=generator,
            feasible_rate=feasible_rate, func_params=func_params,
            cell_name=cell_name, dst_dir=dst_dir,
            entropy_threshold=entropy_threshold, chunk_size=chunk_size,
            delta_entropy_threshold=delta_entropy_threshold
        )

        children_info = compss_wait_on(children_info)
        children_info_all.extend(children_info)

    # Wait for all results
    for i in range(len(children_info_all)):
        result = children_info_all[i]
        children_info_all[i] = compss_wait_on(result)

    return children_info_all


def get_children_samples(cases_heritage_df, cell, dims_heritage_df):
    dims = []
    cases = []
    for idx, row in dims_heritage_df.iterrows():
        case_id = row["case_id"]
        # Cell dimensions don't include g_for and g_fol, but dims_df do
        if 'p_g_for' in row.index and 'p_g_fol' in row.index:
            if not isinstance(row, pd.Series):
                message = "Row is not a pd.Series object"
                logger.error(message)
                raise TypeError(message)
            columns_to_drop = ['p_g_for', 'p_g_fol', 'p_load']
            q_columns = row.filter(regex=r'^q_')
            columns_to_drop.extend(q_columns.index)
            row = row.drop(labels=columns_to_drop)

        # Check if every dimension in row is within cell borders
        cell_independent_dims = [dim for dim in cell.dimensions
                                 if dim.independent_dimension]
        cell_borders = {
            cell_independent_dims[t].label: cell_independent_dims[t].borders
            for t in range(len(cell_independent_dims))
        }
        belongs = all(cell_borders[label][0] <= value <= cell_borders[label][1]
                      for label, value in row.items()
                      if label in cell_borders)
        if all(
                value == cell_borders[label][0]
                for label, value in row.items()
                if label in cell_borders
        ) and idx != 0:
            belongs = False
        if all(
                value == cell_borders[label][1]
                for label, value in row.items()
                if label in cell_borders
        ) and idx != len(dims_heritage_df) - 1:
            belongs = False
        if belongs:
            matching_case = cases_heritage_df[
                cases_heritage_df["case_id"] == case_id]
            matching_dim = dims_heritage_df[
                dims_heritage_df["case_id"] == case_id]
            cases.append(matching_case)
            dims.append(matching_dim)

    if cases and dims:
        cases_df = pd.concat(cases, axis=0, ignore_index=True)
        dims_df = pd.concat(dims, axis=0, ignore_index=True)
    else:
        cases_df = pd.DataFrame()
        dims_df = pd.DataFrame()
    return cases_df, dims_df


def get_parent_samples(dst_dir, cell_name):
    """
    Retrieve parent cases_df and dims_df CSVs for a given cell_name,
    and return them as concatenated pandas DataFrames.

    Example: if cell_name = "0.1.4.2",
      -> combines [cases_df_0.1.4.csv, cases_df_0.1.csv, cases_df_0.csv]
         into one DataFrame
      -> combines [dims_df_0.1.4.csv, dims_df_0.1.csv, dims_df_0.csv]
         into another DataFrame
    """
    parts = cell_name.split(".")
    parent_names = [".".join(parts[:i]) for i in range(len(parts)-1, 0, -1)]

    cases_dfs, dims_dfs = [], []

    for parent in parent_names:
        cases_path = os.path.join(dst_dir, f"cases_df_{parent}.csv")
        dims_path  = os.path.join(dst_dir, f"dims_df_{parent}.csv")

        if os.path.exists(cases_path):
            cases_dfs.append(pd.read_csv(cases_path))
        else:
            print(f"⚠️ Missing: {cases_path}")

        if os.path.exists(dims_path):
            dims_dfs.append(pd.read_csv(dims_path))
        else:
            print(f"⚠️ Missing: {dims_path}")

    cases_df = pd.concat(cases_dfs, ignore_index=True) if cases_dfs else pd.DataFrame()
    dims_df  = pd.concat(dims_dfs, ignore_index=True) if dims_dfs else pd.DataFrame()

    return cases_df, dims_df
