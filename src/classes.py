import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.task import task
from scipy.stats import qmc


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

    def __init__(self, variables, n_cases, divs, lower, upper, label="None"):
        self.variables = variables
        self.n_cases = n_cases
        self.divs = divs
        self.borders = (lower, upper)
        self.label = label

    def get_cases(self, sample):
        tolerance = 0.1
        valid_cases = []

        while len(valid_cases) < self.n_cases:
            case = np.array(
                [np.random.uniform(lb, ub) for lb, ub in self.variables])

            if abs(np.sum(case) - sample) <= tolerance:
                valid_cases.append(case)

        return np.array(valid_cases)

    """
        def get_cases(self, sample):
        # for ii in range(len(self.samples)):
        sampler = qmc.LatinHypercube(d=len(self.variables))
        current_cases = []
        while len(current_cases) < self.n_cases:
            samples_lhs = sampler.random(n=self.n_cases)
            lb = []
            ub = []
            for v in range(len(self.variables)):
                lb.append(self.variables[v][0])
                ub.append(self.variables[v][1])

            # scale to upper and lower bounds
            new_samples = qmc.scale(samples_lhs, lb, ub)
            # scale to comply the total sum per row (sample)
            sum_new_samples = np.sum(new_samples, axis=1)
            alpha = sum_new_samples / sample
            norm_samples = np.zeros([self.n_cases, len(self.variables)])

            for kk in range(self.n_cases):
                for jj in range(len(self.variables)):
                    norm_samples[kk, jj] = new_samples[kk, jj] / alpha[kk]

            for i in range(len(norm_samples)):
                if all(norm_samples[i] < ub) and all(norm_samples[i] > lb):
                    current_cases.append(norm_samples[i])
        return norm_samples
"""
