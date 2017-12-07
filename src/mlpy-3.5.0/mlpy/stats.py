## This code is written by Davide Albanese, <albanese@fbk.eu>.
##(C) 2011 mlpy Developers.

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

__all__ = ['bootstrap_ci', 'quantile']

import numpy as np

import sys
if sys.version >= '3':
    from . import gsl
else:
    import gsl


def bootstrap_ci(x, B=1000, alpha=0.05, seed=0):
    """Computes the (1-alpha) Bootstrap confidence interval
    from empirical bootstrap distribution of sample mean. 
        
    The lower and upper confidence bounds are the (B*alpha/2)-th 
    and B * (1-alpha/2)-th ordered means, respectively.
    For B = 1000 and alpha = 0.05 these are the 25th and 975th
    ordered means.
    """
    
    x_arr = np.ravel(x)

    if B < 2:
        raise ValueError("B must be >= 2")
    
    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be in [0, 1]")

    np.random.seed(seed)
    
    bmean = np.empty(B, dtype=np.float)
    for b in range(B):
        idx = np.random.random_integers(0, x_arr.shape[0]-1, x_arr.shape[0])
        bmean[b] = np.mean(x_arr[idx])
        
    bmean.sort()
    lower = int(B * (alpha * 0.5))
    upper = int(B * (1 - (alpha * 0.5)))
    
    return (bmean[lower], bmean[upper])


def quantile(x, f):
    """Returns a quantile value of `x`.

    The quantile is determined by the `f`, a fraction between 
    0 and 1. For example, to compute the value of the 75th
    percentile `f` should have the value 0.75.
    """

    xarr = np.array(x, dtype=np.float, copy=True)
    xarr = np.ravel(x)
    xarr.sort()
    
    return gsl.stats_quantile_from_sorted_data(xarr, f)
