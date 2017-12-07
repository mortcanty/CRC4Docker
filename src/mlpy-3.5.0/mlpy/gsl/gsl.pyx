from cgsl cimport *
import numpy as np
cimport numpy as np

def sf_gamma (double x):
    return gsl_sf_gamma (x)

def sf_fact (unsigned int n):
    return gsl_sf_fact (n)

def stats_quantile_from_sorted_data (sorted_data, f):
    cdef np.ndarray[np.float_t, ndim=1] sorted_data_arr

    try:
        sorted_data_arr = np.array(sorted_data, dtype=np.float, order='C')
    except ValueError:
        raise ValueError("sorted_data must be a 2d array_like object")

    return gsl_stats_quantile_from_sorted_data (<double *> 
        sorted_data_arr.data, 1, <size_t> sorted_data_arr.shape[0], 
        <double> f)
