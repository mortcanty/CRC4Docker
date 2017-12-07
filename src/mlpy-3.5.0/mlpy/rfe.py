## This code is written by Davide Albanese, <albanese@fbk.eu>.
## (C) 2011 mlpy Developers.

## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

__all__ = ['rfe_kfda', 'rfe_w2']

import numpy as np
from kernel_class import *
from da import KFDAC


# used in rfe_kfda
def rayleigh(x, kernel, lmb, alpha, idx1, idx2):
    R = np.empty(x.shape[1], dtype=np.float)
    idx = np.ones(x.shape[1], dtype=bool)
    for i in range(x.shape[1]):
        idx[i] = False
        xi = x[:, idx]
        n = xi.shape[0]
        n1, n2 = idx1.shape[0], idx2.shape[0]
        K = kernel.kernel(xi, xi)
        K1, K2 = K[:, idx1], K[:, idx2]
        N1 = np.dot(np.dot(K1, np.eye(n1) - \
                 (1 / float(n1))), K1.T)
        N2 = np.dot(np.dot(K2, np.eye(n2) - \
                 (1 / float(n2))), K2.T)
        N = N1 + N2 + np.diag(np.repeat(lmb, n))
        m1 = np.sum(K1, axis=1) / float(n1)
        m2 = np.sum(K2, axis=1) / float(n2)
        d = m1 - m2
        M = np.dot(d.reshape(-1, 1), d.reshape(1, -1))
        R[i] = np.dot(np.dot(alpha, M), alpha.reshape(-1, 1)) / \
            np.dot(np.dot(alpha, N), alpha.reshape(-1, 1))
        idx[i] = True
        
    return R


def rfe_kfda(x, y, p, lmb, kernel):
    """KFDA-RFE algorithm based on the Rayleigh coefficient
    proposed in [Louw06]_. The algorithm works with only two
    classes.
    
    .. [Louw06] N Louw and S J Steel. Variable selection in kernel Fisher discriminant analysis by means of recursive feature elimination. Journal Computational Statistics & Data Analysis, 2006.
    
    :Parameters:
       x: 2d array_like object (N,P)
          training data
       y : 1d array_like object integer (N)
          class labels (only two classes)
       p : float [0.0, 1.0]
          percentage  of features (upper rounded) to remove
          at each iteration (p=0 one variable)
       lmb : float (>= 0.0)
          regularization parameter
       kernel : mlpy.Kernel object.
          kernel.

    :Returns:
       ranking : 1d numpy array int
          feature ranking. ranking[i] contains the feature index ranked
          in i-th position.
    """

    if (p < 0.0) or (p > 1.0):
        raise ValueError("parameter p must be in [0.0, 1.0]")

    xarr = np.asarray(x, dtype=np.float)
    yarr = np.asarray(y, dtype=np.int)

    if xarr.ndim != 2:
        raise ValueError("x must be a 2d array_like object")

    if yarr.ndim != 1:
        raise ValueError("y must be an 1d array_like object")

    if xarr.shape[0] != yarr.shape[0]:
        raise ValueError("x, y shape mismatch")

    if not isinstance(kernel, Kernel):
        raise ValueError("kernel must be None or a mlpy.Kernel object")
    
    labels = np.unique(yarr)
    if labels.shape[0] != 2:
        raise ValueError("number of classes must be = 2")
    
    idx1 = np.where(yarr==labels[0])[0]
    idx2 = np.where(yarr==labels[1])[0]

    kfda = KFDAC(lmb=lmb, kernel=kernel)
    idxglobal = np.arange(xarr.shape[1], dtype=np.int)
    ranking = []

    while True:
        nelim = np.max((int(np.ceil(idxglobal.shape[0] * p)), 1))
        xi = xarr[:, idxglobal]
        # compute alpha
        kfda.learn(xi, yarr)
        alpha = kfda.alpha()
        # compute the rayleigh coeffs
        R = rayleigh(xi, kernel, lmb, alpha, idx1, idx2)
        # sorted indexes (descending order)
        idxsorted = np.argsort(R)[::-1]
        # indexes to remove
        idxelim = idxglobal[idxsorted[:nelim]][::-1]
        ranking.insert(0, idxelim)
        # update idxglobal
        idxglobal = idxglobal[idxsorted[nelim:]]
        idxglobal.sort()
        
        if len(idxglobal) <= 1:
            ranking.insert(0, idxglobal)
            break
    
    return np.concatenate(ranking)


def rfe_w2(x, y, p, classifier):
    """RFE algorithm, where the ranking criteria is w^2,
    described in [Guyon02]_. `classifier` must be an linear classifier 
    with learn() and w() methods.
        
    .. [Guyon02] I Guyon, J Weston, S Barnhill and V Vapnik. Gene Selection for Cancer Classification using Support Vector Machines. Machine Learning, 2002.
    
    :Parameters:
       x: 2d array_like object (N,P)
          training data
       y : 1d array_like object integer (N)
          class labels (only two classes)
       p : float [0.0, 1.0]
          percentage  of features (upper rounded) to remove
          at each iteration (p=0 one variable)
       classifier : object with learn() and w() methods
          object

    :Returns:
       ranking : 1d numpy array int
          feature ranking. ranking[i] contains the feature index ranked
          in i-th position.
    """

    
    if (p < 0.0) or (p > 1.0):
        raise ValueError("parameter p must be in [0.0, 1.0]")

    if not (hasattr(classifier, 'learn') and hasattr(classifier, 'w')):
        raise ValueError("parameter classifier must have learn() and w() methods")

    xarr = np.asarray(x, dtype=np.float)
    yarr = np.asarray(y, dtype=np.int)

    if xarr.ndim != 2:
        raise ValueError("x must be a 2d array_like object")

    if yarr.ndim != 1:
        raise ValueError("y must be an 1d array_like object")

    if xarr.shape[0] != yarr.shape[0]:
        raise ValueError("x, y shape mismatch")
    
    labels = np.unique(yarr)
    if labels.shape[0] != 2:
        raise ValueError("number of classes must be = 2")
    
    idxglobal = np.arange(xarr.shape[1], dtype=np.int)
    ranking = []

    while True:
        nelim = np.max((int(np.ceil(idxglobal.shape[0] * p)), 1))
        xi = xarr[:, idxglobal]
        classifier.learn(xi, yarr)
        w = classifier.w()
        idxsorted = np.argsort(w**2)
        # indexes to remove
        idxelim = idxglobal[idxsorted[:nelim]][::-1]
        ranking.insert(0, idxelim)
        # update idxglobal
        idxglobal = idxglobal[idxsorted[nelim:]]
        idxglobal.sort()
        
        if len(idxglobal) <= 1:
            ranking.insert(0, idxglobal)
            break
    
    return np.concatenate(ranking)
