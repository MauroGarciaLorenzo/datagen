import numpy as np
from itertools import chain
from pycompss.api.api import compss_wait_on
from pycompss.api.task import task
import random
import time
from scipy.stats import qmc
import pandas as pd

from sklearn.linear_model import LogisticRegression

#%%

# main function
def exec(n_samples, Dims, f, error):
    grid = gen_grid(Dims)
    subsamples = [None] * len(grid)
    for cell in range(len(grid)):
        subsamples[cell] = grid[cell].explore_cell(f, n_samples, None, error, 0, cell) # for each cell in grid, explore_cell
    # grid = compss_wait_on(grid)
    last_children = getLastChildren(grid, []) 
    for c in last_children:
        print(c.delta_entropy)
    return grid

def getLastChildren(grid, last_children):
    for cell in grid:
        if cell.children == []: last_children.append(cell)
        else: last_children.extend(getLastChildren(cell.children, []))
    return last_children

class Cell():
    def __init__(self, Dimensions):
        self.dimensions = Dimensions
        self.subsamples = None
        self.entropy = None
        self.delta_entropy = None
        self.children = []

    @task(returns=1)
    def explore_cell(self, f, n_samples, entropy, error, depth, cell):
        subsamples_df_total = pd.DataFrame()
        subsamples = []
        samples = self.gen_samples(n_samples) # generate first samples (n_samples for each dimension)
        
        # generate subsamples (n_subsamples(attribute of the class Dimension) for each dim) 
        subsample_df = self.gen_subsamples(samples,n_samples)
        
        subsamples_df_total = pd.concat([subsamples_df_total,subsample_df],axis=0)
        
        
        for s in range(len(subsample_df)):
            subsamples.append(Subsample(list(subsample_df.drop(['Dim'+str(d) for d in range(len(self.dimensions))],axis=1).iloc[s]),list(subsample_df[['Dim'+str(d) for d in range(len(self.dimensions))]].iloc[s])))
        # eval each subsample
        for i in subsamples:
            i.stability = eval_subsamples(i.subsample,f)

        
        entropy, delta_entropy = eval_entropy(subsamples, entropy) # eval entropy. Save entropy and delta_entropy as an attribute of the class Cell
        
        self.delta_entropy=delta_entropy
        self.entropy=entropy
        self.subsamples=subsamples
        
        if delta_entropy < 0 or not check_dims(self.dimensions, error): 
            return subsamples

        else:
            #new_divs=self.sensitivity()
            
            #for i in range(len(new_divs)):
            #    self.dimensions[i].divs=new_divs[i]

            children = gen_children(n_samples, self.dimensions, entropy, self.subsamples)
            div = tuple(dim.divs for dim in self.dimensions)
            total_div = np.prod(div)
            children_subsamples = [None] * len(children)
            for cell_child in range(len(children_subsamples)):
                children_subsamples[cell_child] = children[cell_child].explore_cell(f, n_samples, entropy, error, depth + 1, cell)
            
            self.children = children
            
            for cell in range(len(children)):
                children_subsamples[cell] = compss_wait_on(children_subsamples[cell])
            children_subsamples = list(filter(lambda x: x is not None, children_subsamples))
            l = list(chain.from_iterable(children_subsamples))
            subsamples.extend(l) 
                                    
        return subsamples


    # generate n_samples samples for each dim
    def gen_samples(self, n_samples):
        samples = []

        sampler = qmc.LatinHypercube(d=len(self.dimensions))
        samples = sampler.random(n=n_samples)

        #for _ in range(n_samples):
            #sample = []
        samples_scaled=np.zeros([n_samples,len(self.dimensions)])
        samples_scaled_s=np.zeros([1,len(self.dimensions)])

        for s in range(n_samples):
            samples_s=samples[s,:]

            for d in range(len(self.dimensions)):
                dimension = self.dimensions[d]
                lower, upper = dimension.borders

                sample=samples_s[d]
                samples_scaled_s[0,d] = lower + sample * (upper - lower)
            # samples_scaled.append(samples_scaled_s)   
            samples_scaled[s,:]=samples_scaled_s
        return samples_scaled


    #receives first samples
    def gen_subsamples(self, samples,n_samples):
        samplesD = list(zip(*samples)) # gets a list with samples split by dimension (one list for each dim)
        total_samples = pd.DataFrame()

        # for each dim, get subsamples and re join subsamples ([Dim1Vars, Dim2Vars, Dim3Vars, ... DimNVars])
        for d in range(len(self.dimensions)): 
            total_samples_d = pd.DataFrame()

            for i in range(n_samples):
                subsamples_dim = self.dimensions[d].get_subsamples(samplesD[d][i])
                subsamples_dim_df=pd.DataFrame(subsamples_dim)
                columns=[]
                for v in range(len(self.dimensions[d].variables)):
                   columns.append('Dim'+str(d)+'_Var'+str(v))
                   
                subsamples_dim_df.columns=columns
                subsamples_dim_df['Dim'+str(d)]=samplesD[d][i]
                
                total_samples_d=pd.concat([total_samples_d,subsamples_dim_df],axis=0)
            total_samples_d=total_samples_d.reset_index(drop=True)
            total_samples=pd.concat([total_samples,total_samples_d],axis=1)
        
        return total_samples       
        
    def sensitivity(self):
        X=[]
        y=[]
        for s in self.subsamples:
            X.append(s.subsample_dim)
            y.append(s.stability)
           
        X=np.array(X)
        y=np.array(y)
        
        X_avg=np.mean(X,axis=0)
        X_min=np.min(X,axis=0)
        X_max=np.max(X,axis=0)
        
        
        model=LogisticRegression()
        model.fit(X,y)
            
        y_test=np.zeros([2,1])
        
        std=np.zeros([len(X_avg),1])
        
        for i in range(len(X_min)):
            X_test=np.copy(X_avg).reshape(1,-1)
            X_test[0,i]=X_min[i]
            y_test[0,0]=model.predict(X_test)
            
            X_test[0,i]=X_max[i]
            y_test[1,0]=model.predict(X_test)
            
            std[i]=np.std(y_test)
            
        dim_max_std=np.argmax(std)
        
        divs=[1,1,1]
        divs[dim_max_std]=2
       
        return divs
       


class Dimension():
    def __init__(self, Variables, n_subsamples, divs, lower, upper):#,sample):
        self.variables = Variables
        self.n_subsamples = n_subsamples
        self.divs = divs
        self.borders = (lower,upper)
#        self.sample=sample
        
    def get_subsamples(self,sample):
        #for ii in range(len(self.samples)):
        sampler = qmc.LatinHypercube(d=len(self.variables))
        samples_lhs = sampler.random(n=self.n_subsamples)
                
        lb=[]
        ub=[]
        for v in range(len(self.variables)):
            
            lb.append(self.variables[v][0])
            ub.append(self.variables[v][1])
            
        new_samples=qmc.scale(samples_lhs, lb, ub)

        sum_new_samples=np.sum(new_samples,axis=1)
        
        alpha=sum_new_samples/sample
        
        norm_samples=np.zeros([self.n_subsamples,len(self.variables)])

        for kk in range(self.n_subsamples):
            for jj in range(len(self.variables)):
                norm_samples[kk,jj]=new_samples[kk,jj]/alpha[kk]
    
        return norm_samples



    # # Gets subsamples from dimension's samples
    # def get_subsamples(self, samples):
    #     samplesD = []
    #     for sample in samples:
    #         new_samples = []
    #         lb = self.borders[0]
    #         ub = self.borders[1]
    #         for _ in range(self.n_subsamples):
    #             s = random.uniform(0, 1)
    #             scaled_sample = lb + s * (ub - lb)
    #             new_samples.append(scaled_sample)
    #         samplesD.extend(new_samples)
    #     return samplesD
    
    

class Subsample():
    def __init__(self, subsample, subsample_dim):
        self.subsample = subsample
        self.stability = None
        self.subsample_dim=subsample_dim


# task defined (every subsample is going to be evaluated in parallel)
# funtion is received as a parameter
@task(returns=1)
def eval_subsamples(subsample, f):
    return f(subsample)


# Generates grid from Dimensions received
def gen_grid(dims):
    n_dims = len(dims)
    ini = tuple(dim.borders[0] for dim in dims)
    fin = tuple(dim.borders[1] for dim in dims)
    div = tuple(dim.divs for dim in dims)
    total_div = np.prod(div)
    grid = []
    for i in range(total_div):
        div_indices = np.unravel_index(i, div)
        lower = [ini[j] + (fin[j] - ini[j]) / div[j] * div_indices[j] for j in range(n_dims)]
        upper = [ini[j] + (fin[j] - ini[j]) / div[j] * (div_indices[j]+1) for j in range(n_dims)]
        Dimensions = []
        for j in range(len(dims)):
            Dimensions.append(Dimension(dims[j].variables, dims[j].n_subsamples, dims[j].divs, lower[j], upper[j]))
        grid.append(Cell(Dimensions))
    return grid


def calculate_entropy(freqs):
    E = 0
    for ii in range(len(freqs)):
        E = E - freqs[ii] * np.log(freqs[ii])
    return E


# Gets entropy and delta_entropy. 
# Saves entropy in self.entropy and delta_entropy in self.delta_entropy
def eval_entropy(subsamples, entropy):
    freqs = []
    cont = 0
    for i in subsamples:
        i.stability = compss_wait_on(i.stability)
        if i.stability == 1: cont += 1
    freqs.append(cont/len(subsamples) + 0.00001)
    freqs.append((len(subsamples) - cont)/len(subsamples)+0.00001)
    E = calculate_entropy(freqs)
    if entropy == None: delta_entropy = 1
    else: delta_entropy = E - entropy
    return E, delta_entropy


def gen_children(n_samples, dims, entropy,subsamples): #subsamples = grid[cell].subamples
    n_dims = len(dims)
    ini = tuple(dim.borders[0] for dim in dims)
    fin = tuple(dim.borders[1] for dim in dims)
    div = tuple(dim.divs for dim in dims)
    total_div = np.prod(div)
    grid_children = []
    for i in range(total_div):
        div_indices = np.unravel_index(i, div)
        lower = [ini[j] + (fin[j] - ini[j]) / div[j] * div_indices[j] for j in range(n_dims)]
        upper = [ini[j] + (fin[j] - ini[j]) / div[j] * (div_indices[j]+1) for j in range(n_dims)]
        Dimensions = []
        for j in range(len(dims)):
            Dimensions.append(Dimension(dims[j].variables, dims[j].n_subsamples, dims[j].divs, lower[j], upper[j]))
        grid_children.append(Cell(Dimensions))
        
        subsamples_list=[]

        for s in subsamples:
            if all([s.subsample_dim [t] >=lower[t] for t in range(n_dims)]) and all([s.subsample_dim [t] <=upper[t] for t in range(n_dims)]):
                subsamples_list.append(s)
                
        grid_children[i].subsamples=subsamples_list
        entropy, delta_entropy = eval_entropy(grid_children[i].subsamples, grid_children[i].entropy) # eval entropy. Save entropy and delta_entropy as an attribute of the class Cell

        grid_children[i].delta_entropy=delta_entropy
        grid_children[i].entropy=entropy
    return grid_children



def check_dims(dims, error):
    for i in dims:
        if (i.borders[1] - i.borders[0]) < error: return False

    return True

