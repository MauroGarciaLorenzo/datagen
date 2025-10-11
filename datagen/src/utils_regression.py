import numpy as np

def rgr_inputs(byp, ayp, byn, ayn, cy, cz,  bzp,azp, bzn, azn, theta):

    ## data set _4909, removed corr vars
    # y = [p1, p2, v1,..., v9, theta1]

    # # V1
    # byn[2] = -0.0315442546471506
    # ayn[2] = 1.08609116512098
    # byp[2] = -0.0377954179176665
    # ayp[2] = 1.08609116512098
    # #theta 1
    # byn[11] = -0.0155145364124157
    # ayn[11] = 0.280713084794927
    # byp[11] = 0.00246455502475496
    # ayp[11] = 0.280713084794927
    # # P GFOR 1
    # byn[0] = 7.62429687000212e-5
    # ayn[0] = 4.16138295634869
    # byp[0] = 0.00515434969172181
    # ayp[0] = 4.16138295634869
    # # P SG 2
    # byn[1] = -0.000544874376064117
    # ayn[1] = 1.21518076182698
    # byp[1] = -0.0002663633801353
    # ayp[1] = 1.21518076182698
    # # No loads
    # # z = np.zeros(3)
    # # bzp = np.copy(z)
    # # azp = np.copy(z)
    # # bzn = np.copy(z)
    # # azn = np.copy(z)
    # # Terms
    # k = 1.00696204466922 - theta
    
    ## data set _4909, NO removed corr vars
    # ## y = [p1, p2, v1,..., v9, theta1, theta6]
    # # V1
    # byn[2] = - 0.0249127407765362
    # ayn[2] = 1.08609116512098
    # byp[2] = - 0.0388709208365658
    # ayp[2] = 1.08609116512098
    # # V6
    # byn[7] = 0.00580120592938459
    # ayn[7] = 1.02343704764492
    # byp[7] = 0.043387098164852
    # ayp[7] = 1.02343704764492
    # # #theta 1
    # byn[11] = -0.00364816881583431
    # ayn[11] = 0.280713084794927
    # byp[11] = 0.00405272346324446
    # ayp[11] = 0.280713084794927
    # ## theta 6
    # byn[12] = - 0.0138642042507058
    # ayn[12] = - 0.12682417701087
    # byp[12] = - 0.00468399169866162
    # ayp[12] = - 0.12682417701087
    # # # P GFOR 1
    # byn[0] = - 0.00152878002245204
    # ayn[0] = 4.16138295634869
    # byp[0] = 0.00590474432172466
    # ayp[0] = 4.16138295634869
    # # # P SG 2
    # # byn[1] = -0.000544874376064117
    # # ayn[1] = 1.21518076182698
    # # byp[1] = -0.0002663633801353
    # # ayp[1] = 1.21518076182698
    # # # No loads
    # # z = np.zeros(3)
    # # bzp = np.copy(z)
    # # azp = np.copy(z)
    # # bzn = np.copy(z)
    # # azn = np.copy(z)
    # # # Terms
    # k =  1.0067936133136- theta
    
    ## data set _3663, removed corr vars
    ## y = [p1, p2, v1,..., v9]
    # V1
    byn[2] = - 0.02954888980811
    ayn[2] = 0.975719379898085
    byp[2] = 0.0289789644947749
    ayp[2] = 0.975719379898085
    # V6
    byn[7] = 0.0035517789440024
    ayn[7] = 0.926898159423878
    byp[7] = - 0.00712075582772141
    ayp[7] = 0.926898159423878
    
    k = 0.999066436863123- theta
    
    return byp, ayp, byn, ayn, cy, cz, bzp,azp,bzn, azn, k

def small_signal_constraint_gradient(y_r, z, byp, ayp, byn, ayn, cy, cz, bzp, azp, bzn, azn, k):
    
    #t_0 - t_7 should be equal for any case
    t_0 = y_r - ayp
    t_1 = byp * np.maximum(t_0, 0)
    t_2 = ayn - y_r
    t_3 = byn * np.maximum(t_2, 0)
    t_4 = bzp * np.maximum(z - azp, 0)
    t_5 = bzn * np.maximum(azn - z, 0)
    t_6 = cy * y_r
    t_7 = cz * z
    
    # # penalty term max(0,g()-theta)^3 
    # t_8 = k + np.sum(t_1) + np.sum(t_3) + np.sum(t_4) + np.sum(t_5) + np.sum(t_6) + np.sum(t_7)
    # t_9 = np.maximum(t_8, 0)**2
    # t_10 = np.maximum(np.sign(t_8), 0)
    # t_11 = 3*t_9*t_10
    # # functionValue = (np.maximum(((((((k + np.sum(t_1)) + np.sum(t_3)) + np.sum(t_4)) + np.sum(t_5)) + np.sum(t_6)) + np.sum(t_7)), 0) ** 3)
    # gradient = (((t_11 * (byp * np.maximum(np.sign(t_0), 0))) - ((3 * (t_9 * t_10)) * (byn * np.maximum(np.sign(t_2), 0)))) + (t_11 * cy))

    # penalty term max(0,g()-theta)
    t_8 = np.maximum(np.sign(((((((k + np.sum(t_1)) + np.sum(t_3)) + np.sum(t_4)) + np.sum(t_5)) + np.sum(t_6)) + np.sum(t_7))), 0)
    # functionValue = np.maximum(((((((k + np.sum(t_1)) + np.sum(t_3)) + np.sum(t_4)) + np.sum(t_5)) + np.sum(t_6)) + np.sum(t_7)), 0)
    gradient = (((t_8 * (byp * np.maximum(np.sign(t_0), 0))) - (t_8 * (byn * np.maximum(np.sign(t_2), 0)))) + (t_8 * cy))

    return gradient
    

def DI_rgr(y_r,z, byp, ayp, byn, ayn, cy, cz, bzp, azp, bzn, azn, k, theta):
    
    #t_0 - t_7 should be equal for any case
    t_0 = y_r - ayp
    t_1 = byp * np.maximum(t_0, 0)
    t_2 = ayn - y_r
    t_3 = byn * np.maximum(t_2, 0)
    t_4 = bzp * np.maximum(z - azp, 0)
    t_5 = bzn * np.maximum(azn - z, 0)
    t_6 = cy * y_r
    t_7 = cz * z
    
    # # penalty term max(0,g()-theta)^3 
    # t_8 = k + np.sum(t_1) + np.sum(t_3) + np.sum(t_4) + np.sum(t_5) + np.sum(t_6) + np.sum(t_7)
    # t_9 = np.maximum(t_8, 0)**2
    # t_10 = np.maximum(np.sign(t_8), 0)
    # t_11 = 3*t_9*t_10
    # DI = ((((((k + np.sum(t_1)) + np.sum(t_3)) + np.sum(t_4)) + np.sum(t_5)) + np.sum(t_6)) + np.sum(t_7))
    # penalty = max(0,DI-self._theta)**3

    # penalty term max(0,g()-theta)
    t_8 = np.maximum(np.sign(((((((k + np.sum(t_1)) + np.sum(t_3)) + np.sum(t_4)) + np.sum(t_5)) + np.sum(t_6)) + np.sum(t_7))), 0)
    DI = ((((((k + np.sum(t_1)) + np.sum(t_3)) + np.sum(t_4)) + np.sum(t_5)) + np.sum(t_6)) + np.sum(t_7))
    penalty = max(0,DI - theta)
    return DI, penalty

def initialize_rgr_inputs(ofo):
    byp=np.zeros(ofo._ny)
    ayp=np.zeros(ofo._ny)
    byn=np.zeros(ofo._ny)
    ayn=np.zeros(ofo._ny)
    cy = np.zeros(ofo._ny)
    cz = np.zeros(3)
    z = np.zeros(3)
    bzp = np.copy(z)
    azp = np.copy(z)
    bzn = np.copy(z)
    azn = np.copy(z)    
    y_r=np.zeros(ofo._ny)

    return y_r,z, byp, ayp, byn, ayn, cy, cz, bzp, azp, bzn, azn