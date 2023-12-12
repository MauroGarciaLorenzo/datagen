import pandas as pd
import numpy as np
from numpy import linalg
import control

def FEIG(ss_sys, plot):
    
    # Compute state-space system eigenvalues
    eig, eigv = linalg.eig(ss_sys.A)
    
    # Compute real, imaginary, damping and frequency
    real = np.real(eig)
    imag = np.imag(eig)
    damp = -real/np.absolute(eig)
    freq = np.absolute(imag)/(2*np.pi)
    
    # Generate table    
    T_EIG = pd.DataFrame({'real':real, 'imag':imag, 'freq':freq, 'damp':damp})
    T_EIG = T_EIG.sort_values(by='real', ascending = False)
    
    # Add mode ID
    mode = list(range(1,len(T_EIG)+1))
    T_EIG.insert(0,"mode",mode)
    
    return T_EIG
