## This code is written by Davide Albanese, <albanese@fbk.eu>
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


import numpy as np
cimport numpy as np

cdef extern from "c_adatron.h":
    int adatron(long *y, double *K, int n, double C, int maxsteps, 
                double eps,double *alpha, double *margin)
    
np.import_array()
    

class KernelAdatron:
    """Kernel Adatron algorithm without-bias-term (binary classifier).
    
    The algoritm handles a version of the 1-norm soft margin
    support vector machine. If C is very high the algoritm 
    handles a version of the hard margin SVM.

    Use positive definite kernels (such as Gaussian
    and Polynomial kernels)
    """
    
    def __init__(self, C=1000, maxsteps=1000, eps=0.01):
        """Initialization.

        :Parameters:
           C : float
              upper bound on the value of alpha
           maxsteps : integer (> 0)
              maximum number of steps
           eps : float (>=0)
              the algoritm stops when abs(1 - margin) < eps
        """

        self._C = float(C)
        self._maxsteps = int(maxsteps)
        self._eps = float(eps)
        self._alpha = None
        self._margin = None
        self._steps = None
        self._labels = None
        self._y = None

    def learn(self, K, y):
        """Learn.

        Parameters:
           K: 2d array_like object (N, N)
              precomputed kernel matrix
           y : 1d array_like object (N)
              target values
        """
        
        cdef np.ndarray[np.int_t, ndim=1] ynew
        cdef np.ndarray[np.float_t, ndim=2] K_arr
        cdef np.ndarray[np.float_t, ndim=1] alpha_arr
        cdef double margin

        K_arr = np.ascontiguousarray(K, dtype=np.float)
        y_arr = np.asarray(y, dtype=np.int)

        if K_arr.ndim != 2:
            raise ValueError("K must be a 2d array_like object")

        if y_arr.ndim != 1:
            raise ValueError("y must be an 1d array_like object")
        
        if K_arr.shape[0] != K_arr.shape[1]:
            raise ValueError("K must be a square matrix")

        if K_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("K, y shape mismatch")
        
        self._labels = np.unique(y_arr)
        if self._labels.shape[0] != 2:
            raise ValueError("number of classes != 2")

        ynew = np.where(y_arr==self._labels[0], -1, 1)
        n = K_arr.shape[0]
        alpha_arr = np.zeros(n, dtype=np.float)

        steps = adatron(<long*> ynew.data, <double *> K_arr.data,
            <int> n, <double> self._C, <int> self._maxsteps,
            <double> self._eps, <double *> alpha_arr.data, &margin)

        self._alpha = alpha_arr
        self._margin = margin
        self._steps = steps
        self._y = ynew

    def pred(self, Kt):
        """Compute the predicted class.

        :Parameters:
           Kt : 1d or 2d array_like object ([M], N)
              test kernel matrix. Precomputed inner products 
              (in feature space) between M testing and N 
              training points.

        :Returns:
           p : integer or 1d numpy array
              predicted class
        """

        if self._alpha is None:
            raise ValueError("no model computed; run learn() first")

        Kt_arr = np.asarray(Kt, dtype=np.float)

        try:
            s = np.sign(np.dot(self._alpha * self._y, Kt_arr.T))
        except ValueError:
            raise ValueError("Kt, alpha: shape mismatch")
        
        return np.where(s==-1, self._labels[0], self._labels[1]) \
            .astype(np.int)

    def margin(self):
        """Return the margin.
        """

        return self._margin
    
    def steps(self):
        """Return the number of steps performed.
        """

        return self._steps
    
    def alpha(self):
        """Return alpha
        """
        
        return self._alpha
