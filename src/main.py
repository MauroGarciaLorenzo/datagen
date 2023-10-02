from classes import Dimension
from utils import flatten_list
from sampling import explore_cell, gen_grid
from objective_function import dummy
from pycompss.api.task import task
from utils import f
import time

@task(returns=1)
def main():
    VariablesD1 = [(0, 2), (0, 1.5), (0, 1.5)]
    VariablesD2 = [(0, 1), (0, 1.5), (0, 1.5), (1, 2)]
    VariablesD3 = [(1, 3.5), (1, 3.5)]
    dim_min = [0, 1, 2]
    dim_max = [5, 6, 7]
    n_samples = 5
    n_cases = 3
    tolerance = 0.1
    max_depth = 5
    divs = [2,1,1]
    #ax = plt.figure().add_subplot(projection='3d')
    Dims = []
    Dims.append(Dimension(VariablesD1, n_subsamples, divs[0], dim_min[0], dim_max[0]))
    Dims.append(Dimension(VariablesD2, n_subsamples, divs[1], dim_min[1], dim_max[1]))
    Dims.append(Dimension(VariablesD3, n_subsamples, divs[2], dim_min[2], dim_max[2]))
    t1=time.time()
    exec(n_samples, Dims, f, error, None)
    print("tiempo de execucion: ",time.time()-t1)

    # implement reduce
    for cell in range(len(grid)):
        result[cell] = compss_wait_on(result[cell])
        list_cases_df[cell] = compss_wait_on(list_cases_df[cell])
    cases_df = pd.concat(list_cases_df, ignore_index=True)

    result = flatten_list(result)
    for r in result:
        print("number of samples in cell:", len(r[1]))
        print("samples-stability:")
        for i in r[1]:
            for j in i.case:
                print(j)
            print(i.stability)
            print("-----------------------------------------------------------------")
        print("entropy:", r[2])
        print("delta entropy:", r[3])
        print("depth:", r[4])
        print("")
        print("")
        print("")


if __name__ == "__main__":
    main()
