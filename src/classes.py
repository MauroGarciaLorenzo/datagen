import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.task import task
from scipy.stats import qmc


class Cell:
    def __init__(self, dimensions):
        self.dimensions = dimensions


class Dimension:
    def __init__(self, variables, n_cases, divs, lower, upper):  # ,sample):
        self.variables = variables
        self.n_cases = n_cases
        self.divs = divs
        self.borders = (lower, upper)

    #        self.sample=sample

    def get_cases(self, sample):
        # for ii in range(len(self.samples)):
        sampler = qmc.LatinHypercube(d=len(self.variables))
        samples_lhs = sampler.random(n=self.n_cases)

        lb = []
        ub = []
        for v in range(len(self.variables)):

            lb.append(self.variables[v][0])
            ub.append(self.variables[v][1])

        new_samples = qmc.scale(samples_lhs, lb, ub)

        sum_new_samples = np.sum(new_samples, axis=1)

        alpha = sum_new_samples / sample

        norm_samples = np.zeros([self.n_cases, len(self.variables)])

        for kk in range(self.n_cases):
            for jj in range(len(self.variables)):
                norm_samples[kk, jj] = new_samples[kk, jj] / alpha[kk]

        return norm_samples
