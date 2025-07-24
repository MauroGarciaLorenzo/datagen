import pandas as pd
import logging

from datagen.src.file_io import log_cell_info

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
from dataclasses import dataclass

@dataclass
class ExplorationResult:
    children_info: list
    cases_df: pd.DataFrame
    dims_df: pd.DataFrame
    output_dataframes: dict


@constraint(is_local=True)
@task(returns=1, on_failure='FAIL', priority=True)
def explore_cell(func, n_samples, parent_entropy, depth, ax, dimensions,
                 cases_heritage_df, dims_heritage_df, use_sensitivity,
                 max_depth, divs_per_cell, generator, feasible_rate,
                 func_params, total_dataframes=None, cell_name="",
                 dst_dir=None):
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
    :param cases_heritage_df: Inherited cases dataframe
    :param dims_heritage_df: Inherited independent_dims dataframe
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
    :return cases_df: Concatenation of inherited cases and those produced by
    the cell
    :return dims_df: Concatenation of inherited independent_dims and those produced by
    the cell
    """
    if not cell_name:
        cell_name = "0"
    logger.info(f"Entering cell {cell_name}")
    # Generate samples (n_samples for each dimension)
    samples_df = gen_samples(n_samples, dimensions, generator)
    # Generate cases (n_cases (attribute of the class Dimension) for each dim)
    cases_df, dims_df = gen_cases(samples_df, dimensions, generator)

    stabilities = []
    output_dataframes_list = []
    feasible_cases = 0
    stabilities_chunk = []
    output_dataframes_chunk = []

    index = 0
    for _, case in cases_df.iterrows():
        stability, output_dataframes = eval_stability(case=case, f=func,
                                                      func_params=func_params,
                                                      dimensions=dimensions,
                                                      generator=generator
                                                      )
        stabilities_chunk.append(stability)
        output_dataframes_chunk.append(output_dataframes)
        index += 1
        if index % 1000 == 0 or index == len(cases_df):
            stabilities_chunk = compss_wait_on(stabilities_chunk)
            output_dataframes_chunk = compss_wait_on(output_dataframes_chunk)
            stabilities.extend(stabilities_chunk)
            output_dataframes_list.extend(output_dataframes_chunk)
            stabilities_chunk = []
            output_dataframes_chunk = []

    for stability in stabilities:
        if stability >= 0:
            feasible_cases += 1
            
    cases_df["Stability"] = stabilities
    # Collect each cases dictionary of dataframes into total_dataframes
    for output_dfs in output_dataframes_list:
        if total_dataframes:
            total_dataframes = concat_df_dict(total_dataframes,
                                              output_dfs)
        else:
            total_dataframes = output_dfs
    # Remove elements of the dict of dataframes that are not a dataframe
    labels_to_remove = []
    if total_dataframes:
        for label, df in total_dataframes.items():
            if df is not None and type(df) is not pd.DataFrame:
                # Keep None values that work as a placeholder
                labels_to_remove.append(label)
        for label in labels_to_remove:
            total_dataframes.pop(label)

    # Add rectangle to plot axes representing cell borders
    if ax is not None and len(dimensions) == 2:
        plot_stabilities(ax, cases_df, dims_df, dst_dir)

    cases_df = pd.concat([cases_df, cases_heritage_df], ignore_index=True)
    dims_df = pd.concat([dims_df, dims_heritage_df], ignore_index=True)

    parent_entropy, delta_entropy = eval_entropy(stabilities, parent_entropy)

    total_cases = n_samples * dimensions[0].n_cases
    message = f"Depth={depth}, Entropy={parent_entropy}, Delta_entropy={delta_entropy}"
    log_cell_info(cell_name, depth, parent_entropy, delta_entropy, feasible_cases / total_cases,
                  1, dst_dir)
    logger.info(message)

    check_entropy = False
    if delta_entropy > 0 or parent_entropy > 0.2:
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

        # Return as ExplorationResult
        return ExplorationResult(
            children_info=[(dimensions, parent_entropy, delta_entropy, depth)],
            cases_df=cases_df,
            dims_df=dims_df,
            output_dataframes=total_dataframes
        )
    else:
        if use_sensitivity:
            dimensions = sensitivity(cases_df, dimensions, divs_per_cell,
                                     generator)
        children_grid = gen_grid(dimensions)

        if ax is not None and len(dimensions) == 2:
            plot_divs(ax, children_grid, dst_dir)

        # Recursive case returns an ExplorationResult
        result = explore_grid(ax=ax, cases_df=cases_df, grid=children_grid,
                              depth=depth, dims_df=dims_df, func=func,
                              n_samples=n_samples,
                              use_sensitivity=use_sensitivity,
                              max_depth=max_depth, divs_per_cell=divs_per_cell,
                              generator=generator, feasible_rate=feasible_rate,
                              func_params=func_params,
                              dataframes=total_dataframes,
                              parent_entropy=parent_entropy,
                              parent_name=cell_name,
                              dst_dir=dst_dir)

        return result


def explore_grid(ax, cases_df, grid, depth, dims_df, func, n_samples,
                 use_sensitivity, max_depth, divs_per_cell, generator,
                 feasible_rate, func_params, dataframes, parent_entropy,
                 parent_name, dst_dir):
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
    :return: Children independent_dims, cases and parameters
    """
    total_cases_df, total_dims_df, total_dataframes = (
        get_children_parameters(grid, dims_df, cases_df, dataframes))

    results = []
    i = 0

    for (children_cell, cases_heritage_df, dims_heritage_df,
         heritage_dataframes) in zip(grid, total_cases_df, total_dims_df,
                                     total_dataframes):
        i += 1
        cell_name = f"{parent_name}.{i}"

        # Get ExplorationResult from child cell
        child_result = explore_cell(
            func=func, n_samples=n_samples,
            parent_entropy=parent_entropy, depth=depth + 1,
            ax=ax, dimensions=children_cell.dimensions,
            cases_heritage_df=cases_heritage_df,
            dims_heritage_df=dims_heritage_df,
            use_sensitivity=use_sensitivity, max_depth=max_depth,
            divs_per_cell=divs_per_cell, generator=generator,
            feasible_rate=feasible_rate, func_params=func_params,
            total_dataframes=heritage_dataframes,
            cell_name=cell_name, dst_dir=dst_dir
        )

        results.append(child_result)

    # Wait for all results
    for i in range(len(results)):
        result = results[i]
        results[i] = compss_wait_on(result)
        

    # Combine all results
    combined_cases = pd.concat([r.cases_df for r in results],
                               ignore_index=True)
    combined_dims = pd.concat([r.dims_df for r in results], ignore_index=True)
    combined_dataframes = concat_df_dict(
        [r.output_dataframes for r in results])

    # Flatten children info
    children_info = []
    for r in results:
        children_info.extend(r.children_info)

    return ExplorationResult(
        children_info=children_info,
        cases_df=combined_cases,
        dims_df=combined_dims,
        output_dataframes=combined_dataframes
    )


def get_children_parameters(children_grid, dims_heritage_df, cases_heritage_df,
                            heritage_dataframes):
    """Obtains dimensions, cases_df, entropy and delta_entropy of each child

    :param children_grid: Grid to obtain parameters
    :param dims_heritage_df: Samples dataframe(one for each case)
    :param cases_heritage_df: Inherited cases dataframe
    :return: List of children (with these attributes set)
    """
    total_cases = []
    total_dims = []
    total_dataframes = []
    for cell in children_grid:
        dims = []
        cases = []
        dataframes = {}
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

            if heritage_dataframes:
                assert dims_heritage_df["case_id"].is_unique
                assert cases_heritage_df["case_id"].is_unique
                assert set(cases_heritage_df["case_id"]) == set(
                    dims_heritage_df["case_id"])

                for label in heritage_dataframes.keys():
                    df = heritage_dataframes[label]

                    # Get the row where case_id matches
                    row_df = df[df["case_id"] == case_id]

                    if not row_df.empty:
                        if label in dataframes:
                            dataframes[label] = pd.concat(
                                [dataframes[label], row_df], axis=0)
                        else:
                            dataframes[label] = row_df
                    else:
                        message = (f"Line with case_id {case_id} not found "
                                   f"in dataframe {label}")
                        logger.error(message)
                        raise Exception(message)

        if cases and dims:
            cases_df = pd.concat(cases, axis=0, ignore_index=True)
            dims_df = pd.concat(dims, axis=0, ignore_index=True)
        else:
            cases_df = pd.DataFrame()
            dims_df = pd.DataFrame()
        total_cases.append(cases_df)
        total_dims.append(dims_df)
        total_dataframes.append(dataframes)

    if cases_heritage_df is None:
        cases_heritage_df = pd.DataFrame()
    if sum([len(cases) for cases in total_cases]) != len(cases_heritage_df):
        message = "Not every case was assigned to a child"
        logger.error(message)
        raise Exception(message)
    return total_cases, total_dims, total_dataframes
