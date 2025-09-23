import numpy as np
import pandas as pd
from scipy.stats import qmc

from datagen.src.dimension_processing import process_p_cig_dimension, \
    process_p_load_dimension, process_control_dimension, \
    process_other_dimensions, process_dimension
from datagen.src.utils import generate_unique_id


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

    for dim in dimensions:
        partial_cases, partial_dims = process_dimension(samples_df, dim, [dimension for dimension in dimensions if dimension.label=='perc_g_for'][0], generator)

        # Add a new column for each p_ column in partial_cases
        for col in partial_cases.columns:
            if col.startswith("p_"):
                new_col_name = col.replace("p_", "q_")
                multiplier = np.sqrt(1 - dim.cosphi ** 2) / dim.cosphi
                partial_cases[new_col_name] = partial_cases[col] * multiplier

        sum_columns_by_prefix = partial_cases.T.groupby(
            lambda x: x.split("_Var")[0]).sum().T
        q_columns = sum_columns_by_prefix.filter(regex=r'^q_', axis=1)
        partial_dims = pd.concat([partial_dims, q_columns], axis=1)

        total_cases.append(partial_cases)
        total_dims.append(partial_dims)

    total_cases_df = pd.concat(total_cases, axis=1)
    total_dims_df = pd.concat(total_dims, axis=1)

    # Generate unique ID per case
    id_df = generate_unique_id(len(total_cases_df))
    total_cases_df = pd.concat((total_cases_df, id_df), axis=1)
    total_dims_df = pd.concat((total_dims_df, id_df), axis=1)
    return total_cases_df, total_dims_df


def gen_samples(n_samples, dimensions, generator):
    """Generates n_samples samples, which represent total sum of the variable_borders
    within a dimension.

    :param generator:
    :param n_samples: Number of samples to produce
    :param dimensions: Involved dimensions
    :return: DataFrame containing these samples with columns named after
    dimension labels
    """
    indepedent_dims = [dim for dim in dimensions if dim.independent_dimension]
    sampler = qmc.LatinHypercube(d=len(indepedent_dims), seed=generator)
    samples = sampler.random(n=n_samples)

    lower_bounds = np.array([dim.borders[0] for dim in indepedent_dims])
    upper_bounds = np.array([dim.borders[1] for dim in indepedent_dims])

    samples_scaled = lower_bounds + samples * (upper_bounds - lower_bounds)

    df_samples = pd.DataFrame(samples_scaled,
                              columns=[dim.label for dim in indepedent_dims])

    return df_samples
