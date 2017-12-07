## Ridge Regression and Kernel Ridge Regression

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

__all__ = ["ridge_base", "Ridge", "KernelRidge"]

import numpy as np
from kernel_class import *


def ridge_base(x, y, lmb):
    """Solves the equation X beta = y by computing a vector beta that
    minimize ||y - X beta||^2 + ||lambda beta||^2 where ||.|| is the L^2
    norm (X is a NxP matrix). When if N >= P the function solves 
    the normal equation (primal solution), when N < P the function 
    solves the dual solution.

    X must be centered by columns.

    :Parameters:
       x : 2d array_like object
          training data (N x P)
       y : 1d array_like object (N)
          target values
       lmb : float (> 0.0)
          lambda, regularization parameter

    :Returns:
       beta : 1d numpy array
          beta
    """

    xarr = np.asarray(x)
    yarr = np.asarray(y)
    n, p = xarr.shape

    if n >= p:
        # primal solution
        # beta = (X'X + lambda I)^-1 X'Y
        beta = np.linalg.solve(np.dot(x.T, x) + 
            lmb * np.eye(x.shape[1]), np.dot(x.T, y))
    else:
        # dual solution 
        # solve two linear equations systems:
        # (XX' + lambda I) alpha = y
        # beta = x^T alpha
        alpha = np.linalg.solve(np.dot(x, x.T) + lmb *
            np.eye(x.shape[0]), y)
        beta = np.dot(x.T, alpha)
        
        # # SVD method
        # # beta = V'(R'R + lambda I)R'Y
        # # where X = UDV and R = UD
        # u, d, v = np.linalg.svd(xarr, full_matrices=False)
        # r = np.dot(u, np.diag(d))
        # tmp = np.linalg.inv(np.dot(r.T, r) + lmb *
        #     np.eye(r.shape[0]))
        # beta = np.dot(np.dot(np.dot(v.T, tmp), r.T), y)
        
    return beta


class Ridge:
    """Ridge Regression.
    
    Solves the equation X beta = y by computing a vector beta that
    minimize ||y - X beta||^2 + ||lambda beta||^2 where ||.|| is the L^2
    norm (X is a NxP matrix). When if N >= P the function solves 
    the normal equation (primal solution), when N < P the function 
    solves the dual solution. 
    """

    def __init__(self, lmb=1.0):
        """Initialization.

        :Parameters:
           lmb : float (>= 0.0)
              regularization parameter
        """

        self._lmb = float(lmb)
        
        if self._lmb < 0:
            raise ValueError("lmb must be >= 0")

        self._beta = None
        self._beta0 = None

    def learn(self, x, y):
        """Compute the regression coefficients.

        Parameters:
           x : 2d array_like object
              training data (N, P)
           y : 1d array_like object (N)
              target values
        """

        xarr = np.array(x, dtype=np.float, copy=True)
        yarr = np.asarray(y, dtype=np.float)

        if xarr.ndim != 2:
            raise ValueError("x must be a 2d array_like object")

        if yarr.ndim != 1:
            raise ValueError("y must be an 1d array_like object")

        if xarr.shape[0] != yarr.shape[0]:
            raise ValueError("x, y shape mismatch")

        xmean =  np.mean(xarr, axis=0)
        xarr = xarr - xmean
                
        self._beta = ridge_base(xarr, yarr, self._lmb)
        self._beta0 = np.mean(y) - np.dot(xmean, self._beta)
                
    def pred(self, t):
        """Compute the predicted response.

        :Parameters:
           t : 1d or 2d array_like object ([M,] P)
              test data

        :Returns:
           p : integer or 1d numpy darray
              predicted response
        """

        if self._beta is None:
            raise ValueError("no model computed; run learn()")

        tarr = np.asarray(t, dtype=np.float)

        if tarr.ndim > 2 or tarr.ndim < 1:
            raise ValueError("t must be an 1d or a 2d array_like object")

        try:
            p = np.dot(tarr, self._beta) + self._beta0
        except ValueError:
            raise ValueError("t, beta: shape mismatch")

        return p

    def beta(self):
        """Return b1, ..., bp.
        """

        return self._beta

    def beta0(self):
        """Return b0.
        """

        return self._beta0


class KernelRidge:
    """Kernel Ridge Regression (dual).
    """

    def __init__(self, lmb=1.0, kernel=None):
        """Initialization.

        :Parameters:
           lmb : float (>= 0.0)
              regularization parameter
           kernel : None or mlpy.Kernel object.
              if kernel is None, K and Kt in .learn()
              and in .pred() methods must be precomputed kernel 
              matricies, else K and Kt must be training (resp. 
              test) data in input space.
        """

        if lmb < 0:
            raise ValueError("lmb must be >= 0")

        if kernel is not None:
            if not isinstance(kernel, Kernel):
                raise ValueError("kernel must be None or a mlpy.Kernel object")

        self._lmb = float(lmb)
        self._kernel = kernel
        self._alpha = None
        self._b = None
        self._x = None
                                
    def learn(self, K, y):
        """Compute the regression coefficients.

        Parameters:
           K: 2d array_like object
              precomputed training kernel matrix (if kernel=None);
              training data in input space (if kernel is a Kernel object)
           y : 1d array_like object (N)
              target values
        """

        K_arr = np.asarray(K, dtype=np.float)
        y_arr = np.asarray(y, dtype=np.float)

        if K_arr.ndim != 2:
            raise ValueError("K must be a 2d array_like object")

        if y_arr.ndim != 1:
            raise ValueError("y must be an 1d array_like object")
        
        if K_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("K, y shape mismatch")

        if self._kernel is None:
            if K_arr.shape[0] != K_arr.shape[1]:
                raise ValueError("K must be a square matrix")
        else:
            self._x = K_arr.copy()
            K_arr = self._kernel.kernel(K_arr, K_arr)

        n = K_arr.shape[0]

        # dual solution 
        # (K + lambda I) alpha = y
        
        # Solve |K+lambdaI 1| |alpha| = |y|
        #       |    1     0| |  b  |   |0|
        # as in G. C. Cawley, N. L. C. Nicola and O. Chapelle.
        # Estimating Predictive Variances with Kernel Ridge 
        # Regression.
        A = np.empty((n+1, n+1), dtype=np.float)
        A[:n, :n] = K_arr + self._lmb * np.eye(n)
        A[n, :n], A[:n, n], A[n, n] = 1., 1., 0.
        g = np.linalg.solve(A, np.append(y_arr, 0))
        self._alpha, self._b = g[:-1], g[-1]
                
    def pred(self, Kt):
        """Compute the predicted response.

        :Parameters:
           Kt : 1d or 2d array_like object
              precomputed test kernel matrix. (if kernel=None);
              test data in input space (if kernel is a Kernel object).

        :Returns:
           p : integer or 1d numpy darray
              predicted response
        """

        if self._alpha is None:
            raise ValueError("no model computed; run learn()")

        Kt_arr = np.asarray(Kt, dtype=np.float)
        if self._kernel is not None:
            Kt_arr = self._kernel.kernel(Kt_arr, self._x)
        
        try:
            p = np.dot(self._alpha, Kt_arr.T) + self._b
        except ValueError:
            raise ValueError("Kt, alpha: shape mismatch")
    
        return p

    def alpha(self):
        """Return alpha.
        """

        return self._alpha
    
    def b(self):
        """Return b.
        """

        return self._b
