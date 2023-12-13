from utils import f
import random
from classes3_2_nocompss_2 import exec, Dimension
#from pycompss.api.task import task
from utils import f
import matplotlib.pyplot as plt

#@task(returns=1)
# def main():
VariablesD1 = [(0,2), (0,1.5), (0,1.5)] 
VariablesD2 = [(0,1), (0,1.5), (0,1.5), (1,2)] 
VariablesD3 = [(1,3.5), (1,3.5)]
dim_min = [0,1,2] 
dim_max = [5,6,7]
n_samples = 5
n_subsamples = 2
error = 0.1
max_depth = 5
divs = [2,1,1]
#ax = plt.figure().add_subplot(projection='3d')
Dims = []
Dims.append(Dimension(VariablesD1, n_subsamples, divs[0], dim_min[0], dim_max[0]))
Dims.append(Dimension(VariablesD2, n_subsamples, divs[1], dim_min[1], dim_max[1]))
Dims.append(Dimension(VariablesD3, n_subsamples, divs[2], dim_min[2], dim_max[2]))
exec(n_samples, Dims, f, error, None)


def print_grid(grid):
    print("")
    for i in range(len(grid)):
        print ("------","casilla",i,"------")
        print("samples casilla", grid[i].n_samples)
        for j in grid[i].dimensions:
            print ("        variables:", j.variables)
            print ("        subsamples", j.n_subsamples)
            print ("        divisiones", j.divs)
            print ("        limites", j.borders)
            print("")
        print("")
        print("")    

if (__name__ == "__main__"):
    main()



"""
------ casilla 0 ------
samples casilla 10
        variables: [(0, 2), (0, 1.5), (0, 1.5)]
        subsamples 5
        divisiones 2
        limites (0.0, 2.5)

        variables: [(0, 1), (0, 1.5), (0, 1.5), (1, 2)]
        subsamples 5
        divisiones 3
        limites (1.0, 2.666666666666667)

        variables: [(1, 3.5), (1, 3.5)]
        subsamples 5
        divisiones 1
        limites (2.0, 7.0)



------ casilla 1 ------
samples casilla 10
        variables: [(0, 2), (0, 1.5), (0, 1.5)]
        subsamples 5
        divisiones 2
        limites (0.0, 2.5)

        variables: [(0, 1), (0, 1.5), (0, 1.5), (1, 2)]
        subsamples 5
        divisiones 3
        limites (2.666666666666667, 4.333333333333334)

        variables: [(1, 3.5), (1, 3.5)]
        subsamples 5
        divisiones 1
        limites (2.0, 7.0)



------ casilla 2 ------
samples casilla 10
        variables: [(0, 2), (0, 1.5), (0, 1.5)]
        subsamples 5
        divisiones 2
        limites (0.0, 2.5)

        variables: [(0, 1), (0, 1.5), (0, 1.5), (1, 2)]
        subsamples 5
        divisiones 3
        limites (4.333333333333334, 6.0)

        variables: [(1, 3.5), (1, 3.5)]
        subsamples 5
        divisiones 1
        limites (2.0, 7.0)



------ casilla 3 ------
samples casilla 10
        variables: [(0, 2), (0, 1.5), (0, 1.5)]
        subsamples 5
        divisiones 2
        limites (2.5, 5.0)

        variables: [(0, 1), (0, 1.5), (0, 1.5), (1, 2)]
        subsamples 5
        divisiones 3
        limites (1.0, 2.666666666666667)

        variables: [(1, 3.5), (1, 3.5)]
        subsamples 5
        divisiones 1
        limites (2.0, 7.0)



------ casilla 4 ------
samples casilla 10
        variables: [(0, 2), (0, 1.5), (0, 1.5)]
        subsamples 5
        divisiones 2
        limites (2.5, 5.0)

        variables: [(0, 1), (0, 1.5), (0, 1.5), (1, 2)]
        subsamples 5
        divisiones 3
        limites (2.666666666666667, 4.333333333333334)

        variables: [(1, 3.5), (1, 3.5)]
        subsamples 5
        divisiones 1
        limites (2.0, 7.0)



------ casilla 5 ------
samples casilla 10
        variables: [(0, 2), (0, 1.5), (0, 1.5)]
        subsamples 5
        divisiones 2
        limites (2.5, 5.0)

        variables: [(0, 1), (0, 1.5), (0, 1.5), (1, 2)]
        subsamples 5
        divisiones 3
        limites (4.333333333333334, 6.0)

        variables: [(1, 3.5), (1, 3.5)]
        subsamples 5
        divisiones 1
        limites (2.0, 7.0)
"""
