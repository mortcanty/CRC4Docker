## This code is written by Davide Albanese, <albanese@fbk.eu>.
## (C) 2010 mlpy Developers.

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

__all__ = ["ols_base", "OLS"]

import numpy as np

def ols_base(x, y, tol):
    """Ordinary (Linear) Least Squares.

    Solves the equation X beta = y by computing a vector beta that
    minimize ||y - X beta||^2 where ||.|| is the L^2 norm
    This function uses numpy.linalg.lstsq().

    X must be centered by columns.

    :Parameters:
       x : 2d array_like object
          training data (samples x features)
       y : 1d array_like object integer (two classes)
          target values
       tol : float
          Cut-off ratio for small singular values of x.
          Singular values are set to zero if they are smaller
          than `tol` times the largest singular value of x.
          If `tol` < 0, machine precision is used instead.

    :Returns:
       beta, rank = 1d numpy array, float
          beta, rank of matrix `x`.
    """

    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=tol)
    return beta, rank


class OLS:
    """Ordinary (Linear) Least Squares Regression (OLS).
    """

    def __init__(self, tol=-1):
        """Initialization.

        :Parameters:
           tol : float
              Cut-off ratio for small singular values of x.
              Singular values are set to zero if they are smaller
              than `tol` times the largest singular value of x.
              If `tol` < 0, machine precision is used instead.
        """

        self._beta = None
        self._beta0 = None
        self._rank = None
        self._tol = tol

    def learn(self, x, y):
        """Learning method.

        :Parameters:
           x : 2d array_like object
              training data (samples x features)
           y : 1d array_like object integer (two classes)
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

        xarr = np.concatenate((np.ones((xarr.shape[0], 1),
                   dtype=np.float), xarr), axis=1)

        beta, self._rank = ols_base(xarr, yarr, self._tol)
        self._beta = beta[1:]
        self._beta0 = beta[0]

    def pred(self, t):
        """Compute the predicted response.

        :Parameters:
           t : 1d or 2d array_like object
              test data

        :Returns:
           p : integer or 1d numpy darray
              predicted response
        """
        if not self._beta or not self._beta0:
            raise ValueError('no mode computed; run learn() first')

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

    def rank(self):
        """Rank of matrix `x`.
        """

        return self._rank

