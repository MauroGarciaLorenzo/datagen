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

"""This module is the main entry point for the application. The main
goal is to explore the various cells (or combinations of dimensions) and
produce both a record of execution logs and a DataFrame containing specific
cases and their associated stability."""
import os
import random
import shutil
import traceback

from datagen.src.logger import setup_logger, logger

import numpy as np
import pandas as pd
import time

from .explorer import explore_cell
from .viz import print_results, boxplot
from .file_io import save_results, init_dst_dir, join_and_cleanup_csvs, \
    clean_incomplete_cells

try:
    from pycompss.api.task import task
    from pycompss.api.api import compss_wait_on, compss_barrier

except ImportError:
    from datagen.dummies.task import task
    from datagen.dummies.api import compss_wait_on, compss_barrier


def start(dimensions, n_samples, rel_tolerance, func, max_depth, dst_dir=None,
          seed=1, use_sensitivity=False, ax=None, sensitivity_divs=2, plot_boxplot=False,
          feasible_rate=0, func_params = {}, warmup=False, logging_level="INFO",
          working_dir=None, entropy_threshold=0.05, delta_entropy_threshold=0,
          chunk_length=5000, yaml_path=None, load_factor=0.9):
    """In this method we work with dimensions (main axes), which represent a
    list of variable_borders. For example, the value of each variable of a concrete
    dimension could represent the power supplied by a generator, while the
    value linked to that dimension should be the total sum of energy produced.

    The function explores each initial cells and obtains two objects from it:
        -execution_logs: dimensions, entropy, delta entropy and depth of each
                        cell.
        -cases_df: dataframe containing each case and associated stability
                taken during the execution.

    :param seed: Seed for creating random numbers (used at explore_cell)
    :param sensitivity_divs: Variable containing the number of divisions for
    each cell at sensitivity analysis
    :param dimensions: List of dimensions involved
    :param n_samples: Number of different values for each dimension
    :param rel_tolerance: Fraction of the dimension's range that will be used
        as the minimum size of new cells in grid generation. 
        e.g., if rel_tolerance = 0.1 the dimension's tolerance will be 10 % of
        its range
    :param ax: Plottable object
    :param use_sensitivity: Boolean indicating whether sensitivity analysis is
    used or not
    :param func: Objective function
    :param max_depth: Maximum depth for a cell to be subdivided
    :param plot_boxplot: Indicates whether a boxplot representing all variable_borders
    should be plotted
    :param feasible_rate: Minimum rate of feasible cases to continue dividing the cell
    :param func_params: Func params to pass to the objective function
    :param warmup: Boolean specifying if the node warmup has to be performed
    This will call a task for every accessible computing node and make the
    appropriate imports
    :param logging_level: Desired logging level.
    Possible values [logging.INFO|logging.WARNING|logging.ERROR],
    default [logging.ERROR]
    :param working_dir: Path where to take the data files. If not specified, it
    is set to datagen root directory.
    :param entropy_threshold: Minimum entropy to keep exploring the cell
    (create new children)
    :param delta_entropy_threshold: Minimum delta entropy to keep exploring the
    cell
    :param chunk_length: Maximum number of calls to eval_stability that will
    be executed simultaneously. Every chunk tasks, there will be a write to
    memory of the current cases, dims and dataframes.
    :param dst_dir: Path where the results will be stored. If the given path
    already have expored cells, these cells will be skipped.
    """

    print(f"\n{''.join(['='] * 30)}\n"
                f"Running application with the following parameters:"
                f"\n{''.join(['='] * 30)}", flush=True)

    # Gather arguments as a dictionary
    args_dict = locals()

    # Print arguments nicely
    for key, value in list(args_dict.items()):
        if key != "func_params" and key != "args_dict":
            print(f"{key}: {value}", flush=True)
    print(f"{''.join(['='] * 30)}\n", flush=True)

    # Set working dir to datagen root directory
    if working_dir is None:
        working_dir = os.path.join(os.path.dirname(__file__), "..", "..")

    # if dst_dir is provided, apply checkpointing. Otherwise, init dst_dir
    if dst_dir is None:
        calling_module = get_calling_module()
        n_cases = dimensions[0].n_cases
        dst_dir = init_dst_dir(calling_module, seed, n_cases, n_samples,
                               max_depth, working_dir, ax, dimensions)

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        print(f"Created results directory: {os.path.abspath(dst_dir)}")
    else:
        clean_incomplete_cells(dst_dir)
        print(f"Using existing results directory: {os.path.abspath(dst_dir)}")

    if not os.path.abspath(dst_dir):
        datagen_root = os.path.join(os.path.dirname(__file__), "..", "..")
        dst_dir = os.path.join(datagen_root, dst_dir)

    # Set up the logging level for the execution
    setup_logger(logging_level, dst_dir)

    # Load imports in every executor before execution
    logger.info(f"DESTINATION DIR: {dst_dir}")

    #Write yaml_path
    if yaml_path is not None:
        shutil.copy(yaml_path, dst_dir)

    logging_level = logger.get_logging_level()
    print(f"Current logging level: {logging_level}")

    if warmup:
        for _ in range(200):
            warmup_nodes()
        compss_barrier()

    t0 = time.time()

    dim_labels = set()
    for dim in dimensions:
        if dim.label in dim_labels:
            message = f"The label {dim.label} is already in use"
            logger.error(message)
            raise Exception(message)

        dim_labels.add(dim.label)
        if dim.independent_dimension:
            dim.tolerance = (dim.borders[1] - dim.borders[0]) * rel_tolerance

    if ax is not None and len(dimensions) == 2:
        x_lims = (dimensions[0].borders[0], dimensions[0].borders[1])
        y_lims = (dimensions[1].borders[0], dimensions[1].borders[1])
        ax.set_xlim(left=x_lims[0], right=x_lims[1])
        ax.set_ylim(bottom=y_lims[0], top=y_lims[1])

    if seed is None:
        seed = random.randint(1,100)

    generator = np.random.default_rng(seed)
    execution_logs = (
        explore_cell(func=func, n_samples=n_samples, parent_entropy=None,
                     depth=0, ax=ax, dimensions=dimensions,
                     use_sensitivity=use_sensitivity, max_depth=max_depth,
                     sensitivity_divs=sensitivity_divs, generator=generator,
                     feasible_rate=feasible_rate, func_params=func_params,
                     dst_dir=dst_dir, chunk_length=chunk_length,
                     entropy_threshold=entropy_threshold,
                     delta_entropy_threshold=delta_entropy_threshold,
                     load_factor=load_factor))

    execution_logs = compss_wait_on(execution_logs)

    if not isinstance(execution_logs, list):
        execution_logs = [execution_logs]

    print_results(execution_logs)
    save_results(execution_logs, dst_dir, time.time()-t0)
    join_and_cleanup_csvs(dst_dir)

    if plot_boxplot:
        cases_df = pd.read_csv(f"{dst_dir}/cases_df_join.csv")
        boxplot(cases_df, dst_dir)

    return execution_logs, dst_dir


@task(is_replicated=True)
def warmup_nodes():
    from . import sampling, utils, objective_function_ACOPF
    time.sleep(1)


def get_calling_module():
    # Extract the raw traceback stack (no inspect)
    stack = traceback.extract_stack()

    if len(stack) >= 5:
        calling_frame = stack[len(stack) - 3]
        return os.path.basename(calling_frame.filename)
    else:
        return "unknown"
