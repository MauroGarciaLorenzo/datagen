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
    def __init__(self, variables, n_cases, divs, lower, upper, label="None",
                 tolerance=0):
        self.variables = variables
        self.n_cases = n_cases
        self.divs = divs
        self.borders = (lower, upper)
        self.label = label
        self.tolerance = tolerance

    def get_cases(self, sample):
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
