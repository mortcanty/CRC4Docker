## Findpeaks submodule

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

__all__ = ["findpeaks_dist", "findpeaks_win"]

import numpy as np
cimport numpy as np
cimport cython
from stdlib cimport free

cdef extern from "c_findpeaks.h":
   int *fp_win(double *x, int n, int span, int *m)


@cython.boundscheck(False)
@cython.wraparound(False)
def findpeaks_win(x, span):
    """Find peaks with a sliding window of
    width `span`.

    :Parameters:
        x : 1d array_like object
            input data
        span : odd integer (>=3)
            span
    
    :Returns:
        idx : 1d numpy array int
            peaks indexes

    Example:
    
    >>> import mlpy
    >>> x = [6,2,2,1,3,4,1,3,1,1,1,6,2,2,7,1]
    >>> mlpy.findpeaks_win(x, span=3)
    array([ 0,  5,  7, 11, 14])
    """

    cdef np.ndarray[np.float_t, ndim=1] xarr
    cdef np.ndarray[np.int_t, ndim=1] ret
    cdef int *retc
    cdef int i, m
    
    span = int(span)

    if (span % 2 == 0) or (span < 3):
        raise ValueError("span must be >= 3 and odd")
    
    xarr = np.ascontiguousarray(x, dtype=np.float)
    retc = fp_win(<double *> xarr.data, <int> xarr.shape[0], <int> span, &m)
    ret = np.empty(m, dtype=np.int)
    
    for i in range(m):
        ret[i] = retc[i]
            
    free(retc)
    return ret


@cython.boundscheck(False)
@cython.wraparound(False)
def findpeaks_dist(x, mindist=2):
    """Find peaks. With `mindist` parameter 
    the algorithm ignore small peaks that occur
    in the neighborhood of a larger peak.

    :Parameters:
        x : 1d array_like object
            input data
        mindist : integer (>=2)
            minimum peak distance (minimum separation
            between peaks)
    
    :Returns:
        idx : 1d numpy array int
            peaks indexes

    Example:
    
    >>> import mlpy
    >>> x = [6,2,2,1,3,4,1,3,1,1,1,6,2,2,7,1]
    >>> mlpy.findpeaks_dist(x, mindist=3)
    array([ 0,  5, 11, 14])
    """

    cdef long i, j, mi
    cdef double mm
    cdef np.ndarray[np.float_t, ndim=1] _x
    cdef np.ndarray[np.int_t, ndim=1] idx
    cdef np.ndarray[np.int_t, ndim=1] tmp
    
    if mindist < 2:
        raise ValueError("mindist must be >= 2")

    _x = np.ascontiguousarray(x, dtype=np.float)
    idx = findpeaks_win(_x, 3)
    tmp = np.empty_like(idx)
 
    j = 0
    mm = _x[idx[0]]
    mi = idx[0]
    
    for i in range(1, idx.shape[0]):
        if (idx[i] - idx[i-1]) < mindist:
            if _x[idx[i]] > mm:
                mm = _x[idx[i]]
                mi = idx[i]
        else:
            tmp[j] = mi
            j += 1
            mm = _x[idx[i]]
            mi = idx[i]
    
    tmp[j] = mi
    return tmp[:j+1]
