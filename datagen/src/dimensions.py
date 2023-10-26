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

"""This module defines two primary classes:

    -Cell: Represents a distinct space within a grid containing a list of
Dimension objects.

    -Dimension: Describes a particular dimension of a specific
cell. Each dimension is made up of variables that can represent different
things, like power supplied by a generator. The Dimension class can produce
various cases based on given samples, considering their tolerance and
maintaining the integrity of the sum of variables.
"""

import random
import numpy as np


class Cell:
    def __init__(self, dimensions):
        self.dimensions = dimensions


class Dimension:
    """Dimension class

    An object from this class represents a concrete dimension of a specific
    cell in our problem. It is defined by:
        -variables: list of variables represented by tuples containing its
                lower and upper borders.
        -n_cases: number of cases taken for each sample (each sample represents
                the total sum of a dimension). A case is a combination of
                variables where all summed together equals the sample.
        -divs: number of divisions in that dimension. It will be the growth
                order of the number of cells
        -borders: bounds of the dimension (maximum and minimum values of a
                sample)
        -label: dimension identifier
    """
    def __init__(self, variables, n_cases, divs, borders, label,
                 tolerance=None):
        self.variables = np.array(variables, dtype='float')
        self.n_cases = n_cases
        self.divs = divs
        self.borders = borders
        self.label = label
        self.tolerance = tolerance

    def get_cases_normal(self, sample, iter_limit_factor=1000):
        """
        Generate `n_cases` number of random cases for the given sample.

        The cases are generated using normal distribution with means obtained
        from distributing the sample value over the different values in a
        proportional way with respect to their ranges (lower/upper bounds).

        The standard deviation of each variable is selected so that there is a
        probability of 99 % of a new point lying inside the variable range.

        :param sample: A random input value representing the dimension, with
            the requirement that the different variables of the dimension
            must collectively sum up to it.
        :param iter_limit_factor: Factor to multiply for the maximum number of
            iterations
        :return cases: Array of the generated cases
        """
        cases = []
        # Perform scaling
        var_avg = ((self.variables[:, 1] - self.variables[:, 0]) / 2
                   + self.variables[:, 0])
        avg_sum = var_avg.sum()
        alpha = sample / avg_sum
        scaled_avgs = var_avg * alpha
        stds = []
        # Perform scaling
        for i in range(len(self.variables)):
            d_min = min(abs(self.variables[i][0] - scaled_avgs[i]),
                        abs(self.variables[i][1] - scaled_avgs[i]))
            # Initialize standard deviations
            stds.append(d_min / 3)
        iters = 0
        iter_limit = len(self.variables) * self.n_cases * iter_limit_factor
        max_val = sum([v[1] for v in self.variables])
        min_val = sum([v[0] for v in self.variables])

        if not (max_val >= sample >= min_val):
            raise ValueError(f"Sample {sample} cannot be reached by "
                             f"dimension {self.label}, with variables borders "
                             f"{self.variables}")

        while len(cases) < self.n_cases and iters < iter_limit:
            case = np.random.normal(scaled_avgs, stds)
            lower_bounds = self.variables[:, 0]
            upper_bounds = self.variables[:, 1]
            case = np.clip(case, lower_bounds, upper_bounds)
            case_sum = case.sum()
            if self.borders[0] < case_sum < self.borders[1]:
                cases.append(case)
            else:
                print(f"get_cases_normal: Iteration {iters + 1}")
                print(f"Warning: (label {self.label}) Case sum {case_sum} out "
                      f"of dimension borders {self.borders} in {case} for "
                      f"sample {sample}. Retrying...")
            iters += 1
        print(f"Dim {self.label}: get_cases_normal run {iters} iterations.")

        while len(cases) < self.n_cases:
            print(f"Warning: Dim {self.label} - get_cases_normal exhausted "
                  f"iterations: {iters} iterations.")
            print("Adding NaN cases")
            cases.append([np.nan] * len(self.variables))

        return cases

    def get_cases_extreme(self, sample, iter_limit=5000,
                          iter_limit_variables=500):
        """This case generator aims to reach more variance between cases within
        a sample. Here, we assign random values to de variables in the range
        lower bound of this variable - minimum between upper bound of the
        variable and remaining sum, so that we never exceed sample.

        Once every variable has value, we will add to it a random value
        between this value and the maximum possible value (explained above)
        until error is less than defined (dimension tolerance).

        :param sample: Target sum
        :param iter_limit: Maximum number of iterations. Useful to avoid
            infinite loops
        :param iter_limit_variables: Maximum number of iterations to go over
            all variables again and distribute the remaining sum
        :return: Combinations of n_cases variables that, when summed together,
            equal sample. If the combination cannot not be found with the
            defined iter_limit, this case will be filled with NaN values.
        """
        # Distribute remaining sum within variables
        # Shuffle variables
        cases = []
        iters_cases = 0
        max_val = sum([v[1] for v in self.variables])
        min_val = sum([v[0] for v in self.variables])

        if not (max_val >= sample >= min_val):
            raise ValueError(f"Sample {sample} cannot be reached by "
                             f"dimension {self.label}, with variables borders "
                             f"{self.variables}")

        while len(cases) < self.n_cases and iters_cases < iter_limit:
            iters_cases += 1
            initial_case = self.variables[:, 0]
            case = initial_case.copy()
            total_sum = sum(case)
            iters_variables = 0
            while (not np.isclose(total_sum, sample) and
                   iters_variables < iter_limit):
                indexes = list(range(len(self.variables)))
                random.shuffle(indexes)

                iters_variables += 1
                for i in indexes:
                    if np.isclose(total_sum, sample):
                        break
                    new_var = random.uniform(case[i],
                                             self.variables[i, 1])
                    new_var = np.clip(new_var, case[i],
                                      case[i] + sample - total_sum)
                    case[i] = new_var
                    total_sum = sum(case)

            if iters_variables >= iter_limit_variables:
                print(f"Warning: sample {sample} couldn't be reached"
                      f" by total sum {total_sum}) in case {case}")
                continue
            if np.isclose(total_sum, sample):
                cases.append(case)
        if iters_cases >= iter_limit:
            print("Warning: Iterations count exceeded. "
                  "Retrying with normal sampling")
            return self.get_cases_normal(sample)

        return cases
