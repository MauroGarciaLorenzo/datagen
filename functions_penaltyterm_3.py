"""
Sample code automatically generated on 2025-09-19 10:16:21

by www.matrixcalculus.org

from input

d/dy relu(sum(byp.*relu(y-ayp))+sum(byn.*relu(ayn-y))+sum(bzp.*relu(z-azp))+sum(bzn.*relu(azn-z))+sum(cy.*y)+sum(cz.*z)+k)^3 = 3*relu(k+sum(byp.*relu(y-ayp))+sum(byn.*relu(ayn-y))+sum(bzp.*relu(z-azp))+sum(bzn.*relu(azn-z))+sum(cy.*y)+sum(cz.*z)).^2*relu(sign(k+sum(byp.*relu(y-ayp))+sum(byn.*relu(ayn-y))+sum(bzp.*relu(z-azp))+sum(bzn.*relu(azn-z))+sum(cy.*y)+sum(cz.*z)))*byp.*relu(sign(y-ayp))-3*relu(k+sum(byp.*relu(y-ayp))+sum(byn.*relu(ayn-y))+sum(bzp.*relu(z-azp))+sum(bzn.*relu(azn-z))+sum(cy.*y)+sum(cz.*z)).^2*relu(sign(k+sum(byp.*relu(y-ayp))+sum(byn.*relu(ayn-y))+sum(bzp.*relu(z-azp))+sum(bzn.*relu(azn-z))+sum(cy.*y)+sum(cz.*z)))*byn.*relu(sign(ayn-y))+3*relu(k+sum(byp.*relu(y-ayp))+sum(byn.*relu(ayn-y))+sum(bzp.*relu(z-azp))+sum(bzn.*relu(azn-z))+sum(cy.*y)+sum(cz.*z)).^2*relu(sign(k+sum(byp.*relu(y-ayp))+sum(byn.*relu(ayn-y))+sum(bzp.*relu(z-azp))+sum(bzn.*relu(azn-z))+sum(cy.*y)+sum(cz.*z)))*cy

where

ayn is a vector
ayp is a vector
azn is a vector
azp is a vector
byn is a vector
byp is a vector
bzn is a vector
bzp is a vector
cy is a vector
cz is a vector
k is a scalar
y is a vector
z is a vector

The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

def fAndG(ayn, ayp, azn, azp, byn, byp, bzn, bzp, cy, cz, k, y, z):
    assert isinstance(ayn, np.ndarray)
    dim = ayn.shape
    assert len(dim) == 1
    ayn_rows = dim[0]
    assert isinstance(ayp, np.ndarray)
    dim = ayp.shape
    assert len(dim) == 1
    ayp_rows = dim[0]
    assert isinstance(azn, np.ndarray)
    dim = azn.shape
    assert len(dim) == 1
    azn_rows = dim[0]
    assert isinstance(azp, np.ndarray)
    dim = azp.shape
    assert len(dim) == 1
    azp_rows = dim[0]
    assert isinstance(byn, np.ndarray)
    dim = byn.shape
    assert len(dim) == 1
    byn_rows = dim[0]
    assert isinstance(byp, np.ndarray)
    dim = byp.shape
    assert len(dim) == 1
    byp_rows = dim[0]
    assert isinstance(bzn, np.ndarray)
    dim = bzn.shape
    assert len(dim) == 1
    bzn_rows = dim[0]
    assert isinstance(bzp, np.ndarray)
    dim = bzp.shape
    assert len(dim) == 1
    bzp_rows = dim[0]
    assert isinstance(cy, np.ndarray)
    dim = cy.shape
    assert len(dim) == 1
    cy_rows = dim[0]
    assert isinstance(cz, np.ndarray)
    dim = cz.shape
    assert len(dim) == 1
    cz_rows = dim[0]
    if isinstance(k, np.ndarray):
        dim = k.shape
        assert dim == (1, )
    assert isinstance(y, np.ndarray)
    dim = y.shape
    assert len(dim) == 1
    y_rows = dim[0]
    assert isinstance(z, np.ndarray)
    dim = z.shape
    assert len(dim) == 1
    z_rows = dim[0]
    assert byp_rows == ayn_rows == cy_rows == y_rows == byn_rows == ayp_rows
    assert bzp_rows == azp_rows == bzn_rows == cz_rows == azn_rows == z_rows

    t_0 = (y - ayp)
    t_1 = (byp * np.maximum(t_0, 0))
    t_2 = (ayn - y)
    t_3 = (byn * np.maximum(t_2, 0))
    t_4 = (bzp * np.maximum((z - azp), 0))
    t_5 = (bzn * np.maximum((azn - z), 0))
    t_6 = (cy * y)
    t_7 = (cz * z)
    t_8 = ((((((k + np.sum(t_1)) + np.sum(t_3)) + np.sum(t_4)) + np.sum(t_5)) + np.sum(t_6)) + np.sum(t_7))
    t_9 = (np.maximum(t_8, 0) ** 2)
    t_10 = np.maximum(np.sign(t_8), 0)
    t_11 = ((3 * t_9) * t_10)
    functionValue = (np.maximum(((((((k + np.sum(t_1)) + np.sum(t_3)) + np.sum(t_4)) + np.sum(t_5)) + np.sum(t_6)) + np.sum(t_7)), 0) ** 3)
    gradient = (((t_11 * (byp * np.maximum(np.sign(t_0), 0))) - ((3 * (t_9 * t_10)) * (byn * np.maximum(np.sign(t_2), 0)))) + (t_11 * cy))

    return functionValue, gradient

def checkGradient(ayn, ayp, azn, azp, byn, byp, bzn, bzp, cy, cz, k, y, z):
    # numerical gradient checking
    # f(x + t * delta) - f(x - t * delta) / (2t)
    # should be roughly equal to inner product <g, delta>
    t = 1E-6
    delta = np.random.randn(3)
    f1, _ = fAndG(ayn, ayp, azn, azp, byn, byp, bzn, bzp, cy, cz, k, y + t * delta, z)
    f2, _ = fAndG(ayn, ayp, azn, azp, byn, byp, bzn, bzp, cy, cz, k, y - t * delta, z)
    f, g = fAndG(ayn, ayp, azn, azp, byn, byp, bzn, bzp, cy, cz, k, y, z)
    print('approximation error',
          np.linalg.norm((f1 - f2) / (2*t) - np.tensordot(g, delta, axes=1)))

def generateRandomData():
    ayn = np.random.randn(3)
    ayp = np.random.randn(3)
    azn = np.random.randn(3)
    azp = np.random.randn(3)
    byn = np.random.randn(3)
    byp = np.random.randn(3)
    bzn = np.random.randn(3)
    bzp = np.random.randn(3)
    cy = np.random.randn(3)
    cz = np.random.randn(3)
    k = np.random.randn(1)
    y = np.random.randn(3)
    z = np.random.randn(3)

    return ayn, ayp, azn, azp, byn, byp, bzn, bzp, cy, cz, k, y, z

if __name__ == '__main__':
    ayn, ayp, azn, azp, byn, byp, bzn, bzp, cy, cz, k, y, z = generateRandomData()
    functionValue, gradient = fAndG(ayn, ayp, azn, azp, byn, byp, bzn, bzp, cy, cz, k, y, z)
    print('functionValue = ', functionValue)
    print('gradient = ', gradient)

    print('numerical gradient checking ...')
    checkGradient(ayn, ayp, azn, azp, byn, byp, bzn, bzp, cy, cz, k, y, z)
