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

        while len(cases) < self.n_cases:
            cases.append([None] * len(self.variables))

        return cases

    def get_cases_extreme(self, sample, iter_limit=5000):
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
        new_case_count = 0
        while len(cases) < self.n_cases and new_case_count < iter_limit:
            # Assign random value between variables minimum and remaining sum
            new_case_count += 1
            total_sum = 0
            case = []
            valid_case = True
            for i in range(len(self.variables)):
                limits = (self.variables[i, 0],
                          min(self.variables[i, 1], abs(sample - total_sum)))
                if limits[1] <= limits[0]:
                    print(f"Warning: sample {sample} exceeded by total sum "
                          f"({total_sum}) in case {case}")
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

            distribute_sum_count = 0
            remaining_sum = sample - total_sum
            while (abs(remaining_sum) > self.tolerance and
                   distribute_sum_count < 500):
                distribute_sum_count += 1
                for i in range(len(case)):
                    if abs(remaining_sum) <= self.tolerance:  # TODO: toler??
                        break
                    new_sum_range = (
                        0,
                        min(remaining_sum, variables_shuffled[i, 1] - case[i]))
                    var_sum = random.random() * new_sum_range[1]
                    case[i] += var_sum
                    remaining_sum -= var_sum

            if distribute_sum_count >= 500:
                print(f"Warning: sample {sample} couldn't be reached"
                      f" by total sum {total_sum}) in case {case}")
                continue
            # Restore variables order
            restore_order = np.argsort(indexes)
            case = [case[i] for i in restore_order]
            if abs(remaining_sum) <= self.tolerance:
                cases.append(case)

        if new_case_count >= 5000:
            print(f"Warning: Iterations count exceeded. Retrying")

        while len(cases) < self.n_cases:
            cases.append([None] * len(self.variables))

        return cases
