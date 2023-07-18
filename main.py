from classes3_2 import exec, Dimension
#from classes1_2 import exec, Dimension
import random
#from pycompss.api.task import task
from utils import f
import matplotlib.pyplot as plt

#%%
#@task(returns=1)
def main():
#%%
    VariablesD1 = [(0,2), (0,1.5), (0,1.5)] 
    VariablesD2 = [(0,1), (0,1.5), (0,1.5), (1,2)] 
    VariablesD3 = [(1,3.5), (1,3.5)]
    dim_min = [0,1,2] 
    dim_max = [5,6,7]
    n_samples = 5
    n_subsamples = 5
    error = 0.5
    max_depth = 5
    divs = [2,1,1]
    
    Dims = []
    Dims.append(Dimension(VariablesD1, n_subsamples, divs[0], dim_min[0], dim_max[0]))
    Dims.append(Dimension(VariablesD2, n_subsamples, divs[1], dim_min[1], dim_max[1]))
    Dims.append(Dimension(VariablesD3, n_subsamples, divs[2], dim_min[2], dim_max[2]))
    
    ax = plt.figure().add_subplot(projection='3d')
#%%    
    exec(n_samples, Dims, f, error,ax)
    #exec(n_samples, Dims, f, max_depth)


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


