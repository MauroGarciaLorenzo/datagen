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
        self.variables = variables
        self.n_cases = n_cases
        self.divs = divs
        self.borders = borders
        self.label = label
        self.tolerance = tolerance

    def get_cases_normal(self, sample):
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
            # Initialize standard deviations.
            stds.append(d_min / 3)
        iters = 0
        iter_limit = len(self.variables) * self.n_cases * 1000

        while len(cases) < self.n_cases and iters < iter_limit:
            case = np.random.normal(scaled_avgs, stds)
            lower_bounds = self.variables[:, 0]
            upper_bounds = self.variables[:, 1]
            case = np.clip(case, lower_bounds, upper_bounds)
            if self.borders[0] < case.sum() < self.borders[1]:
                cases.append(case)
            else:
                print(f"Warning: (label {self.label}) Case sum out of "
                      f"dimension borders {self.borders} in {case} for sample "
                      f"{sample}. Retrying...")
            iters += 1
        print(f"Dim {self.label}: get_cases_normal run {iters} iterations.")

        while len(cases) < self.n_cases:
            cases.append([np.nan] * len(self.variables))

        return cases

    def get_cases_extreme(self, sample, iter_limit=5000, 
                          iter_limit_reloop=500):
        """This case generator aims to reach more variance between cases within
         a sample. Here, we assign random values to de variables in the range
         lower bound of this variable - minimum between upper bound of the
         variable and remaining sum, so that we never exceed sample.

         Once every variable has value, we will add to it a random value
         between this value and the maximum possible value (explained above)
         until error is less than defined.

        :param sample: Target sum
        :param iter_limit: Iterations limit. Useful to avoid an infinite loop
        :return: Combinations of N_cases variables that, when summed together,
        equal sample. If the combination could not be found with the defined
        iter_limit, this case will be none.
        """
        cases = []
        iters_case = 0
        while len(cases) < self.n_cases and iters_case < iter_limit:
            # Assign random value between variables minimum and remaining sum
            iters_case += 1
            total_sum = 0
            case = []
            valid_case = True
            for i in range(len(self.variables)):
                limits = (self.variables[i, 0],
                          min(self.variables[i, 1], abs(sample - total_sum)))
                if limits[1] <= limits[0]:
                    print(f"Lower bound for variable {i} in dimension "
                          f"{self.label} ({limits[0]}) exceeds the remaining "
                          f"sum {sample - total_sum} to reach the sample value"
                          f" {sample}.")
                    valid_case = False
                    break
                var = random.random() * (limits[1] - limits[0]) + limits[0]
                case.append(var)
                total_sum += var
            if not valid_case:
                continue
            # Distribute remaining sum within variables
            # Shuffle variables
            indexes = list(range(len(self.variables)))
            random.shuffle(indexes)
            variables_shuffled = self.variables[indexes]
            case = [case[i] for i in indexes]

            iters_reloop = 0
            remaining_sum = sample - total_sum
            while (abs(remaining_sum) > self.tolerance and
                   iters_reloop < iter_limit_reloop):
                iters_reloop += 1
                for i in range(len(case)):
                    if abs(remaining_sum) <= self.tolerance:  # TODO: toler??
                        break
                    new_sum_range = (
                        0,
                        min(remaining_sum, variables_shuffled[i, 1] - case[i]))
                    var_sum = random.random() * new_sum_range[1]
                    case[i] += var_sum
                    remaining_sum -= var_sum

            if iters_reloop >= iter_limit_reloop:
                print(f"Warning: sample {sample} couldn't be reached"
                      f" by total sum {total_sum}) in case {case}")
                continue
            # Restore variables order
            restore_order = np.argsort(indexes)
            case = [case[i] for i in restore_order]
            if abs(remaining_sum) <= self.tolerance:
                cases.append(case)

        if iters_case >= iter_limit:
            print(f"Warning: Iterations count exceeded. Retrying")

        while len(cases) < self.n_cases:
            cases.append([np.nan] * len(self.variables))

        return cases
