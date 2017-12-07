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

__all__ = ["lars_base", "LARS"]

import numpy as np


def lars_base(x, y, maxsteps=None):
    """Least Angle Regression.

    `x` should be centered and normalized by columns, and `y`
    should be centered.

     :Parameters:
        x : 2d array_like object (N x P)
           matrix of regressors
        y : 1d array_like object (N)
           response
        maxsteps : int (> 0) or None
           maximum number of steps. If `maxsteps` is None,
           the maximum number of steps is min(N-1, P),
           where N is the number of variables and P is the
           number of features.

     :Returns:
        active, est, steps : 1d numpy array, 2d numpy array, int
           active features, all LARS estimates, number of
           steps performed
    """

    xarr = np.asarray(x)
    yarr = np.asarray(y)

    ms = np.min((xarr.shape[0]-1, xarr.shape[1]))
    if (maxsteps == None) or (maxsteps > ms):
        maxsteps = ms

    mu = np.ones(xarr.shape[0])
    active = []
    inactive = range(xarr.shape[1])
    beta = np.zeros(xarr.shape[1], dtype=np.float)
    est = np.zeros((maxsteps+1, xarr.shape[1]), dtype=np.float)

    for i in range(maxsteps):

        # equation 2.8
        c = np.dot(xarr.T, (yarr - mu))

        # equation 2.9
        ct = c.copy()
        ct[active] = 0.0 # avoid re-selections
        ct_abs = np.abs(ct)
        j = np.argmax(ct_abs)

        C = ct_abs[j]
        active.append(j)
        inactive.remove(j)

        # equation 2.10
        s = np.sign(c[active])

        # equation 2.4
        xa = xarr[:, active] * s

        # equation 2.5
        G = np.dot(xa.T, xa)
        Gi = np.linalg.inv(G)
        A = np.sum(Gi)**(-0.5)

        # equation 2.6
        w = np.sum(A * Gi, axis=1)
        u = np.dot(xa, w)

        # equation 2.11
        a = np.dot(xarr.T, u)

        # equation 2.13
        g1 = (C - c[inactive]) / (A - a[inactive])
        g2 = (C + c[inactive]) / (A + a[inactive])
        g = np.concatenate((g1, g2))
        g = g[g > 0.0]

        if g.shape[0] == 0:
            gammahat = C / A # page 9
        else:
            gammahat = np.min(g)

        beta[active] = beta[active] + gammahat * w
        est[i+1, active] = beta[active] * s
        mu = mu + (gammahat * u) # equation 2.12

    return np.asarray(active), est, maxsteps


class LARS():
    """Least Angle Regression.
    """

    def __init__(self, maxsteps=None):
        """Initialization.

        :Parameters:
          maxsteps : int (> 0) or None
            maximum number of steps.
        """

        if maxsteps is not None:
            if maxsteps <= 0:
                raise ValueError("maxsteps must be > 0")

        self._maxsteps = maxsteps

        self._beta = None
        self._beta0 = None
        self._est = None
        self._active = None
        self._steps = None

    def learn(self, x, y):
        """Compute the regression coefficients.

        :Parameters:
          x : 2d array_like object (N x P)
            matrix of regressors
          y : 1d array_like object (N)
            response
        """

        xarr = np.array(x, dtype=np.float, copy=True)
        yarr = np.array(y, dtype=np.float, copy=True)

        if xarr.ndim != 2:
            raise ValueError("x must be a 2d array_like object")

        if yarr.ndim != 1:
            raise ValueError("y must be an 1d array_like object")

        if xarr.shape[0] != yarr.shape[0]:
            raise ValueError("x, y shape mismatch")

        # center x
        xmean = np.mean(xarr, axis=0)
        xarr -= xmean

        # normalize x
        xnorm = np.sqrt(np.sum((xarr)**2, axis=0))
        xarr /= xnorm

        # center y
        ymean = np.mean(yarr)

        self._active, self._est, self._steps = \
            lars_base(xarr, yarr, self._maxsteps)
        self._est /= xnorm
        self._beta = np.copy(self._est[-1])
        self._beta0 = ymean - np.dot(xmean, self._beta)

    def pred(self, t):
        """Compute the predicted response.

        :Parameters:
           t : 1d or 2d array_like object ([M,] P)
              test data

        :Returns:
           p : float or 1d numpy array
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

    def est(self):
        """Returns all LARS estimates.
        """

        return self._est

    def active(self):
        """Returns the active features.
        """

        return self._active

    def beta(self):
        """Return b_1, ..., b_p.
        """

        return self._beta

    def beta0(self):
        """Return b_0.
        """

        return self._beta0

    def steps(self):
        """Return the number of steps performed.
        """

        return self._steps
