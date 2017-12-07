## PLS

## This code is written by Davide Albanese, <davide.albanese@gmail.com>.
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

__all__ = ['PLS']

import numpy as np


class PLS:
    """Multivariate primal Partial Least Squares (PLS) 
    algorithm as described in [Taylor04]_.

    .. [Taylor04] J Shawe-Taylor and N Cristianini. Kernel Methods for Pattern Analysis.
    """

    def __init__(self, iters):
        """Initialization.

        :Parameters:
           iters : int (>= 1)
              number of iterations. iters should be <= min(N-1, P)
        """

        self._iters = iters
        self._xmean = None
        self._beta0 = None
        self._beta = None

    def learn(self, x, y):
        """Compute the regression coefficients.

        Parameters:
           x : 2d array_like object
              training data (N, P)
           y : 1d array_like object (N [,M])
              target values
        """

        xarr = np.array(x, dtype=np.float, copy=True)
        yarr = np.array(y, dtype=np.float, copy=True)

        if xarr.ndim != 2:
            raise ValueError("x must be a 2d array_like object")

        if yarr.ndim > 2:
            raise ValueError("y must be an 1d array_like object")
    
        if xarr.shape[0] != yarr.shape[0]:
            raise ValueError("x, y shape mismatch")

        if yarr.ndim == 1:
            yarr = yarr.reshape(-1, 1)
        
        self._xmean =  np.mean(xarr, axis=0)
        self._beta0 =  np.mean(yarr, axis=0)
        xarr = xarr - self._xmean
        yarr = yarr - self._beta0
        
        u = np.empty((xarr.shape[1], self._iters), dtype=np.float)
        c = np.empty((yarr.shape[1], self._iters), dtype=np.float)
        p = np.empty((xarr.shape[1], self._iters), dtype=np.float)
        for i in range(self._iters):
            YX = np.dot(yarr.T, xarr)
            u[:, i] = YX[0] / np.linalg.norm(YX[0])

            if yarr.shape[1] > 1:
                uold = u[:, i] + 1
                while np.linalg.norm(u[:, i] - uold) > 0.001:
                    uold = u[:, i]
                    tu = np.dot(np.dot(YX.T, YX), u[:, i])
                    u[:, i] = tu / np.linalg.norm(tu)
            
            t = np.dot(xarr, u[:, i].reshape(-1, 1))
            tt = np.dot(t.T, t)
            c[:, i] = np.ravel(np.dot(yarr.T, t) / tt)
            p[:, i] = np.ravel(np.dot(xarr.T, t) / tt)
            xarr = xarr - np.dot(t, p[:, i].reshape(1, -1))         
            
        self._beta = np.dot(u, np.linalg.solve(np.dot(p.T, u), c.T))
        
        if yarr.shape[1] == 1:
            self._beta = np.ravel(self._beta)

    def pred(self, t):
        """Compute the predicted response(s).

        :Parameters:
           t : 1d or 2d array_like object ([M,] P)
              test data

        :Returns:
           p : integer or 1d numpy darray
              predicted response(s)
        """

        if self._beta is None:
            raise ValueError("no model computed; run learn()")

        tarr = np.asarray(t, dtype=np.float)
        
        if tarr.ndim > 2 or tarr.ndim < 1:
            raise ValueError("t must be an 1d or a 2d array_like object")
        
        try:
            p = np.dot(tarr - self._xmean, self._beta) + self._beta0
        except ValueError:
            raise ValueError("t, beta: shape mismatch")

        return p
    
    def beta(self):
        """Returns the regression coefficients. 
        
        beta is a (P) vector in the univariate case
        and a (P, M) matrix in the multivariate case,
        where M is the number of target outputs.
        """

        return self._beta

    def beta0(self):
        """Returns offset(s). 
        
        beta is a float in the univariate case,
        and a (M) vector in the multivariate case,
        where M is the number of target outputs.
        """

        return self._beta0
