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
from libc.stdlib cimport *

from clcs cimport *

np.import_array()


def lcs_std(x, y):
    """Standard Longest Common Subsequence (LCS)
    algorithm as described in [Cormen01]_.
    
    The elements of sequences must be coded as integers.
    
    :Parameters:
       x : 1d integer array_like object (N)
          first sequence
       y : 1d integer array_like object (M)
          second sequence

    :Returns:
       length : integer
          length of the LCS of x and y
       path : tuple of two 1d numpy array (path_x, path_y)
          path of the LCS

    Example
    
    Reproducing the example in figure 15.6 of [Cormen01]_,
    where sequence X = (A, B, C, B, D, A, B) and Y = (B, D, C, A, B, A).

    >>> import mlpy
    >>> x = [0,1,2,1,3,0,1] # (A, B, C, B, D, A, B)
    >>> y = [1,3,2,0,1,0] # (B, D, C, A, B, A)
    >>> length, path = mlpy.lcs_std(x, y)
    >>> length
    4
    >>> path
    (array([1, 2, 3, 5]), array([0, 2, 4, 5]))   

    .. [Cormen01] H Cormen et al.. Introduction to Algorithms, Second Edition. The MIT Press, 2001.
    """

    cdef np.ndarray[np.int_t, ndim=1] x_arr
    cdef np.ndarray[np.int_t, ndim=1] y_arr
    cdef np.ndarray[np.int_t, ndim=1] px_arr
    cdef np.ndarray[np.int_t, ndim=1] py_arr
    cdef char **b
    cdef int i
    cdef Path p
    cdef int length

    x_arr = np.ascontiguousarray(x, dtype=np.int)
    y_arr = np.ascontiguousarray(y, dtype=np.int)

    b = <char **> malloc ((x_arr.shape[0]+1) * sizeof(char *))
    for i in range(x_arr.shape[0]+1):
        b[i] = <char *> malloc ((y_arr.shape[0]+1) * sizeof(char))    

    length = std(<long *> x_arr.data, <long *> y_arr.data, b,
                  <int> x_arr.shape[0], <int> y_arr.shape[0])

    trace(b, <int> x_arr.shape[0], <int> y_arr.shape[0], &p)
    
    for i in range(x_arr.shape[0]+1):
        free (b[i])
    free(b)

    px_arr = np.empty(p.k, dtype=np.int)
    py_arr = np.empty(p.k, dtype=np.int)
    
    for i in range(p.k):
         px_arr[i] = p.px[i]
         py_arr[i] = p.py[i]

    free (p.px)
    free (p.py)
    
    return length, (px_arr, py_arr)


def lcs_real(x, y, eps, delta):
    """Longest Common Subsequence (LCS) for series
    composed by real numbers as described in 
    [Vlachos02]_.
       
    :Parameters:
       x : 1d integer array_like object (N)
          first sequence
       y : 1d integer array_like object (M)
          second sequence
       eps : float (>=0)
          matching threshold
       delta : int (>=0)
          controls how far in time we can go in order to
          match a given point from one series to a 
          point in another series

    :Returns:
       length : integer
          length of the LCS of x and y
       path : tuple of two 1d numpy array (path_x, path_y)
          path of the LCS

    .. [Vlachos02] M Vlachos et al.. Discovering Similar Multidimensional Trajectories. In Proceedings of the 18th international conference on data engineering, 2002
    """

    cdef np.ndarray[np.float_t, ndim=1] x_arr
    cdef np.ndarray[np.float_t, ndim=1] y_arr
    cdef np.ndarray[np.int_t, ndim=1] px_arr
    cdef np.ndarray[np.int_t, ndim=1] py_arr
    cdef char **b
    cdef int i
    cdef Path p
    cdef int length

    x_arr = np.ascontiguousarray(x, dtype=np.float)
    y_arr = np.ascontiguousarray(y, dtype=np.float)

    b = <char **> malloc ((x_arr.shape[0]+1) * sizeof(char *))
    for i in range(x_arr.shape[0]+1):
        b[i] = <char *> malloc ((y_arr.shape[0]+1) * sizeof(char))    

    length = real(<double *> x_arr.data, <double *> y_arr.data, b,
                  <int> x_arr.shape[0], <int> y_arr.shape[0],
                   <double> eps, <int> delta)

    trace(b, <int> x_arr.shape[0], <int> y_arr.shape[0], &p)
    
    for i in range(x_arr.shape[0]+1):
        free (b[i])
    free(b)

    px_arr = np.empty(p.k, dtype=np.int)
    py_arr = np.empty(p.k, dtype=np.int)
    
    for i in range(p.k):
         px_arr[i] = p.px[i]
         py_arr[i] = p.py[i]

    free (p.px)
    free (p.py)
    
    return length, (px_arr, py_arr)
