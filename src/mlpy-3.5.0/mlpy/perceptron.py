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

import numpy as np

__all__ = ["Perceptron"]

class Perceptron:
    """Perceptron binary classifier.
    """

    def __init__(self, alpha=0.1, thr=0.0, maxiters=1000):
        """The algorithm stops when the iteration error is less
        or equal than `thr`, or a predetermined number of 
        iterations (`maxiters`) have been completed.

        :Parameters:
           alpha : float, in range (0.0, 1]
              learning rate
           thr : float, in range [0.0, 1.0]
              iteration error (e.g. thr=0.13 for error=13%) 
           maxiters : integer (>0)
              maximum number of iterations
        """


        self._alpha = alpha # learning rate, where 0.0 < alpha <= 1
        self._maxiters = maxiters
        self._thr = float(thr) # error threshold

        self._labels = None
        self._w = None
        self._bias = None # bias term
        self._err = None
        self._iters = None

    def learn(self, x, y):
        """Learning method.

        :Parameters:
           x : 2d array_like object
              training data (N, P)
           y : 1d array_like object integer (only two classes)
              target values (N)
        """

        xarr = np.asarray(x, dtype=np.float)
        yarr = np.asarray(y, dtype=np.int)
        
        if xarr.ndim != 2:
            raise ValueError("x must be a 2d array_like object")
        
        if yarr.ndim != 1:
            raise ValueError("y must be an 1d array_like object")
        
        if xarr.shape[0] != yarr.shape[0]:
            raise ValueError("x, y: shape mismatch")

        self._labels = np.unique(yarr)
        k = self._labels.shape[0]
        
        if k != 2:
            raise ValueError("number of classes must be = 2")
        
        ynew = np.where(yarr == self._labels[0], 0, 1)
        
        self._w = np.zeros(xarr.shape[1], dtype=np.float)
        self._bias = 0.0
        n = ynew.shape[0]
        
        for i in range(self._maxiters):
            tmp = np.where((np.dot(xarr, self._w)+self._bias)>0, 1, 0)
            err = np.sum(ynew != tmp) / float(n)
            
            if err <= self._thr:
                i = i - 1
                break
     
            diff = ynew - tmp
            self._w += self._alpha * np.dot(xarr.T, diff)
            self._bias += self._alpha * np.sum(diff)
                 
        tmp = np.where((np.dot(xarr, self._w)+self._bias)>0, 1, 0)
        err = np.sum(ynew != tmp) / float(n)

        self._err = err
        self._iters = i + 1

    def pred(self, t):
        """Prediction method.

        :Parameters:
           t : 1d or 2d array_like object
              testing data ([M,], P)
        """

        if self._w is None:
            raise ValueError("no model computed")

        tarr = np.asarray(t, dtype=np.float)
        if tarr.ndim > 2:
            raise ValueError("t must be an 1d or a 2d array_like object")
        
        try:
            tmp = np.dot(tarr, self._w) + self._bias
        except ValueError:
            raise ValueError("t, model: shape mismatch")

        return np.where(tmp>0, self._labels[1], self._labels[0]) \
            .astype(np.int)
        
    def w(self):
        """Returns the coefficients.
        """

        if self._w is None:
            raise ValueError("no model computed")

        return self._w

    def labels(self):
        """Outputs the name of labels.
        """
        
        return self._labels

    def bias(self):
        """Returns the bias."""
        
        if self._w is None:
            raise ValueError("no model computed.")

        return self._bias

    def err(self):
        """Returns the iteration error"""
        
        return self._err
    
    def iters(self):
        """Returns the number of iterations"""

        return self._iters

    
