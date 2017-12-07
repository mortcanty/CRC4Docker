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

__all__ = ['borda_count']

import numpy as np

import sys
if sys.version >= '3':
    from . import cborda
else:
    import cborda


def borda_count(x, k=None):
    """Given N ranked ids lists of length P compute the number of
    extractions on top-k positions and the mean position for each id.
    Sort the element ids with decreasing number of extractions, and
    element ids with equal number of extractions will be sorted with
    increasing mean positions.
    
    :Parameters:
       x : 2d array_like object integer (N, P)
          ranked ids lists. For each list ids must be unique
          in [0, P-1].
       k : None or integer
          compute borda on top-k position (None -> k = P)
          
    :Returns:
       borda : 1d numpy array objects
          sorted-ids, number of extractions, mean positions

    Example:
    
    >>> import numpy as np
    >>> import mlpy
    >>> x = [[2,4,1,3,0], # first ranked list
    ...      [3,4,1,2,0], # second ranked list
    ...      [2,4,3,0,1], # third ranked list
    ...      [0,1,4,2,3]] # fourth ranked list
    >>> mlpy.borda_count(x=x, k=3)
    (array([4, 1, 2, 3, 0]), array([4, 3, 2, 2, 1]), array([ 1.25      ,  1.66666667,  0.        ,  1.        ,  0.        ]))

      * Id 4 is in the first position with 4 extractions and mean position 1.25.
      * Id 1 is in the first position with 3 extractions and mean position 1.67.
      * ...
    """
    
    x_arr = np.asarray(x, dtype=np.int)
    n, p = x_arr.shape

    if k == None:
        k = p
        
    if k < 1 or k > p:
        raise ValueError('k must be in [1, %d]' % p)

    ext, pos = cborda.core(x_arr, k)
        
    invpos = (pos + 1)**(-1) # avoid zero division
    idx = np.lexsort(keys=(invpos, ext))[::-1] 
    
    return idx, ext[idx], pos[idx]

