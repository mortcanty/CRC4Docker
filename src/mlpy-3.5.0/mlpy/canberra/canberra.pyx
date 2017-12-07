import numpy as np
cimport numpy as np
cimport cython

cdef extern from "c_canberra.h":
    double c_canberra(double *x, double *y, long n)
    double c_canberra_location(long *x, long *y, long n, long k)
    double c_canberra_stability(long *x, long n, long p, long k)
    double c_canberra_expected(long n, long k)

def canberra(x, y):
    """Returns the Canberra distance between two P-vectors x and y:
    sum_i(abs(x_i - y_i) / (abs(x_i) + abs(y_i))).
    """

    cdef np.ndarray[np.float64_t, ndim=1] x_arr
    cdef np.ndarray[np.float64_t, ndim=1] y_arr
    
    x_arr = np.ascontiguousarray(x, dtype=np.float)
    y_arr = np.ascontiguousarray(y, dtype=np.float)

    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("x, y: shape mismatch")
    
    return c_canberra(<double *> x_arr.data, <double *> y_arr.data,
                       <long> x_arr.shape[0])


def canberra_location(x, y, k=None):
    """Returns the Canberra distance between two position lists,
    `x` and `y`. A position list of length P contains the position 
    (from 0 to P-1) of P elements. k is the location parameter,
    if k=None will be set to P.
    """

    cdef np.ndarray[np.int64_t, ndim=1] x_arr
    cdef np.ndarray[np.int64_t, ndim=1] y_arr
    
    x_arr = np.ascontiguousarray(x, dtype=np.int)
    y_arr = np.ascontiguousarray(y, dtype=np.int)

    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("x, y: shape mismatch")
    
    if k == None:
        k = x_arr.shape[0]

    if k <= 0 or k > x_arr.shape[0]:
        raise ValueError("k must be in [1, %i]" % x_arr.shape[0])   

    return c_canberra_location(<long *> x_arr.data,
               <long *> y_arr.data, <long> x_arr.shape[0], <long> k)


def canberra_stability(x, k=None):
    """Returns the Canberra stability indicator between N position
    lists, where `x` is an (N, P) matrix. A position list of length 
    P contains the position (from 0 to P-1) of P elements. k is 
    the location parameter, if k=None will be set to P. The lower 
    the indicator value, the higher the stability of the lists.

    The stability is computed by the mean distance of all the 
    (N(N-1))/2 non trivial values of the distance matrix (computed
    by canberra_location()) scaled by the expected (average) 
    value of the Canberra metric.

    Example:

    >>> import numpy as np
    >>> import mlpy
    >>> x = np.array([[2,4,1,3,0], [3,4,1,2,0], [2,4,3,0,1]])  # 3 position lists
    >>> mlpy.canberra_stability(x, 3) # stability indicator
    0.74862979571499755
    """

    cdef np.ndarray[np.int64_t, ndim=2] x_arr
        
    x_arr = np.ascontiguousarray(x, dtype=np.int)
    
    if k == None:
        k = x_arr.shape[1]

    if k <= 0 or k > x_arr.shape[1]:
        raise ValueError("k must be in [1, %i]" % x_arr.shape[1])
    
    return c_canberra_stability(<long *> x_arr.data, 
               <long> x_arr.shape[0], <long> x_arr.shape[1], <long> k)


def canberra_location_expected(p, k=None):
    """Returns the expected value of the Canberra location distance,
    where `p` is the number of elements and `k` is the number of 
    positions to consider.
    """

    if k == None:
        k = p

    if k <= 0 or k > p:
        raise ValueError("k must be in [1, %i]" % p)

    return c_canberra_expected(p, k)
