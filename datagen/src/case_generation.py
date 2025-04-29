import numpy as np
import pandas as pd
from scipy.stats import qmc

from datagen.src.sampling import generate_columns
from datagen.src.dimensions import Dimension
from datagen.src.utils import generate_unique_id

def process_p_cig_dimension(samples_df, p_cig, generator):
    """ Assigns values to g_for and g_fol dimensions.

    p_cig samples values must be distributed between g_for and g_fol assigning
    a random value between 0 and 1 as a one-to-one percentage, resulting in
    g_for plus g_fol equalling p_cig. g_for variable_borders are calculated as usual,
    while g_fol variable_borders are complimentary to g_for to sum g_fol:
        g_fol_i = p_cig_i - g_for_i

    :param generator: Random number generator
    :param samples_df: Involved samples
    :param p_cig: p_cig dimension
    :return: Cases obtained and samples extended (one sample for each case)
    """
    cases = []
    dims = []

    for _, sample in samples_df.iterrows():
        # Obtain p_cig cases
        cases_p_cig_df = pd.DataFrame(
            p_cig.get_cases_extreme(sample[p_cig.label], generator),
            columns=generate_columns(p_cig)).dropna()
        n_rows = len(cases_p_cig_df)
        dims_p_cig_df = pd.DataFrame(
            np.repeat(sample[p_cig.label], n_rows), columns=[p_cig.label])

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

            # g_for_variables = np.array([
            #     (p_cig.variable_borders[x, 0], cases_p_cig_df.iloc[i, x])
            #     for x in range(len(p_cig.variable_borders))])
            g_for_variables = np.array([
                (0, cases_p_cig_df.iloc[i, x])
                for x in range(len(p_cig.variable_borders))])
            g_for = Dimension(variable_borders=g_for_variables, n_cases=1, divs=1,
                              borders=(p_cig.borders[0], sample[p_cig.label]),
                              label="p_g_for", tolerance=p_cig.tolerance)
            # Create g_for case
            case_g_for = (g_for.get_cases_extreme(g_for_sample, generator))[0]
            if not np.isnan(case_g_for).any():
                # dims_g_for.append(g_for_sample)
                # cases_g_for.append(case_g_for)
                # dims_g_fol.append(g_fol_sample)
                # Compose g_fol subtracting p_cig from g_for case variable_borders
                # cases_g_fol.append(
                #     [cases_p_cig_df.iloc[i, x] - case_g_for[x] if cases_p_cig_df.iloc[i, x] - case_g_for[x] >1e-3 else 0
                #      for x in range(len(p_cig.variable_borders))])
                case_g_fol=[]
                for x in range(len(p_cig.variable_borders)):
                    if cases_p_cig_df.iloc[i, x] - case_g_for[x] >1e-3:
                        case_g_fol.append(cases_p_cig_df.iloc[i, x] - case_g_for[x])
                    else:
                        case_g_fol.append(0)
                        case_g_for[x]=case_g_for[x]+cases_p_cig_df.iloc[i, x] - case_g_for[x]
                dims_g_for.append(g_for_sample)
                cases_g_for.append(case_g_for)
                dims_g_fol.append(g_fol_sample)
                cases_g_fol.append(np.array(case_g_fol).ravel())





        cases_g_for_df = pd.DataFrame(
            cases_g_for,
            columns=[f"p_g_for_Var{v}" for v in range(len(p_cig.variable_borders))])
        dims_g_for_df = pd.DataFrame(dims_g_for, columns=["p_g_for"])
        cases_g_fol_df = pd.DataFrame(
            cases_g_fol,
            columns=[f"p_g_fol_Var{v}" for v in range(len(p_cig.variable_borders))])
        dims_g_fol_df = pd.DataFrame(dims_g_fol, columns=["p_g_fol"])

        # Error check
        check_sum = (cases_g_fol_df.sum(axis=1) + cases_g_for_df.sum(axis=1)
                     - cases_p_cig_df.sum(axis=1)).to_numpy()
        if not np.isclose(check_sum, 0).all():
            raise ValueError("Sum of p_g_for and p_g_fol must equal p_cig")

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


def process_p_load_dimension(samples_df, dim):
    total_cases = []
    total_dim = []
    for _, sample in samples_df.iterrows():
        # TODO: poner el factor de escala de la carga en el setup file
        new_sample = (sample["p_sg"] + sample["p_cig"])*0.9
        cases = [[new_sample * value for value in dim.values]
                 for _ in range(dim.n_cases)]
        for case in cases:
            if not np.isnan(case).any():
                total_cases.append(case)
                total_dim.append(new_sample)

    dims_df = pd.DataFrame(total_dim, columns=[dim.label])
    cases_df = pd.DataFrame(total_cases, columns=generate_columns(dim))
    return cases_df, dims_df


def process_control_dimension(samples_df, dim):
    total_cases = []
    total_dim = []
    for _, sample in samples_df.iterrows():
        cases = [[sample[dim.label]]
                 for _ in range(dim.n_cases)]
        for case in cases:
            if not np.isnan(case).any():
                total_cases.append(case)
                total_dim.append(case)

    dims_df = pd.DataFrame(total_dim, columns=[dim.label])
    cases_df = pd.DataFrame(total_cases, columns=[dim.label])
    return cases_df, dims_df


def process_other_dimensions(samples_df, dim, generator):
    """
    This method assigns values to the variable_borders within a generic dimension.

    :param generator:
    :param samples_df: Dataframe containing every sample in this cell
    :param dim: Involved dimension
    :return: Cases obtained and samples extended (one sample for each case)
    """
    total_cases = []
    total_dim = []
    for _, sample in samples_df.iterrows():
        cases = dim.get_cases_extreme(sample[dim.label], generator)
        for case in cases:
            if not np.isnan(case).any():
                total_cases.append(case)
                total_dim.append(sample[dim.label])

    dims_df = pd.DataFrame(total_dim, columns=[dim.label])
    cases_df = pd.DataFrame(total_cases, columns=generate_columns(dim))
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

    for dim in dimensions:
        if dim.label == "p_cig":
            partial_cases, partial_dims = process_p_cig_dimension(samples_df,
                                                                  dim, generator)
        elif dim.label == "p_load":
            partial_cases, partial_dims = process_p_load_dimension(samples_df,
                                                                   dim)
        elif dim.label.startswith("tau"):
            partial_cases, partial_dims = process_control_dimension(samples_df,
                                                                   dim)
        else:
            partial_cases, partial_dims = process_other_dimensions(samples_df,
                                                                   dim, generator)

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
