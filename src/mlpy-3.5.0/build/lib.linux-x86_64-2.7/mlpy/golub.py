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

__all__ = ["Golub"]


class Golub:
    """Golub binary classifier described in [Golub99]_.

    Decision function is D(x) = w (x-mu), where w is defined
    as w_i = (mu_i(+) - mu_i(-)) / (std_i(+) + std_i(-)) and 
    mu id defined as (mu(+) + mu(-)) / 2.

    .. [Golub99] T R Golub et al. Molecular classification of cancer: Class discovery and class prediction by gene expression monitoring. Science, 1999.
    """
    
    def __init__(self):
        """Initialization.
        """
        
        self._labels = None
        self._w = None
        self._mean = None
        
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
        
        idxn = yarr == self._labels[0]
        idxp = yarr == self._labels[1]
        meann = np.mean(xarr[idxn], axis=0)
        meanp = np.mean(xarr[idxp], axis=0)
        stdn = np.std(xarr[idxn], axis=0, ddof=1)
        stdp = np.std(xarr[idxp], axis=0, ddof=1)
        self._w = (meanp - meann) / (stdp + stdn)
        self._mean = 0.5 * (meanp + meann)
        
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
            tmp = np.dot(tarr-self._mean, self._w)
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
