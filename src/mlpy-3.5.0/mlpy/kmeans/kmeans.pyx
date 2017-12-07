import numpy as np
cimport numpy as np

cdef extern from "c_kmeans.h":

    void init_std(double *data, double *means, int nn, int pp, int kk, unsigned long seed)
    void init_plus(double *data, double *means, int nn, int pp, int kk, unsigned long seed)
    int km(double *data, double *means, int *cls, int nn, int pp, int kk)


def kmeans(x, k, plus=False, seed=0):
    """k-means clustering.

    :Parameters:
       x : 2d array_like object (N, P)
          data
       k : int (1<k<N)
          number of clusters
       plus : bool
          k-means++ algorithm for initialization
       seed : int
          random seed for initialization

    :Returns:
       clusters, means, steps: 1d array, 2d array, int
          cluster membership in 0,...,K-1, means (K,P), number of steps
          
    """

    cdef np.ndarray[np.float64_t, ndim=2] x_arr
    cdef np.ndarray[np.float64_t, ndim=2] means_arr
    cdef np.ndarray[np.int32_t, ndim=1] cls_arr
  
    x_arr = np.ascontiguousarray(x, dtype=np.float)

    if k <= 1 or k >= x_arr.shape[0]:
        raise ValueError("k must be in [2, N-1]")

    means_arr = np.empty((k, x_arr.shape[1]), dtype=np.float)
    cls_arr = np.empty(x_arr.shape[0], dtype=np.int32)
    
    if plus:
        init_plus(<double *> x_arr.data, <double *> means_arr.data,
                <int> x_arr.shape[0], <int> x_arr.shape[1], <int> k,
                <unsigned long> seed)
    else:
        init_std(<double *> x_arr.data, <double *> means_arr.data,
                  <int> x_arr.shape[0], <int> x_arr.shape[1], <int> k,
                  <unsigned long> seed)
        
    steps = km(<double *> x_arr.data,  <double *> means_arr.data,
                <int *> cls_arr.data, <int> x_arr.shape[0], 
                <int> x_arr.shape[1], <int> k)
                
    return cls_arr.astype(np.int), means_arr, steps
