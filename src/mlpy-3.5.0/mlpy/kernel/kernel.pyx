## This code is written by Davide Albanese, <albanese@fbk.eu>.
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
cimport cython

cdef extern from "c_kernel.h":
    double linear(double *x, double *y, int n)
    double polynomial(double *x, double *y, int n, double gamma, double b, double d)
    double gaussian(double *x, double *y, int n, double sigma)
    double exponential(double *x, double *y, int n, double sigma)
    double sigmoid(double *x, double *y, int n, double gamma, double b)


@cython.boundscheck(False)
def kernel_linear(t, x):
    cdef np.ndarray[np.float64_t, ndim=2] t_arr
    cdef np.ndarray[np.float64_t, ndim=2] x_arr
    cdef np.ndarray[np.float64_t, ndim=2] k_arr
    cdef int i, j, nt, nx, pt, px
    
    t_arr = np.array(t, dtype=np.float, order='C', ndmin=2)
    x_arr = np.array(x, dtype=np.float, order='C', ndmin=2)
    nt, pt = t_arr.shape[0], t_arr.shape[1]
    nx, px = x_arr.shape[0], x_arr.shape[1]

    if pt != px:
        raise ValueError("t, x: shape mismatch")
        
    k_arr = np.empty((nt, nx), dtype=np.float)
    for i in range(nt):
        for j in range(nx):
            k_arr[i, j] = linear(<double *> t_arr.data + (i * pt),
                <double *> x_arr.data + (j * px), pt)
    
    if nt == 1:
        return k_arr[0]
    else:
        return k_arr


@cython.boundscheck(False)
def kernel_polynomial(t, x, gamma=1.0, b=1.0, d=2.0):
    cdef np.ndarray[np.float64_t, ndim=2] t_arr
    cdef np.ndarray[np.float64_t, ndim=2] x_arr
    cdef np.ndarray[np.float64_t, ndim=2] k_arr
    cdef int i, j, nt, nx, pt, px
    
    t_arr = np.array(t, dtype=np.float, order='C', ndmin=2)
    x_arr = np.array(x, dtype=np.float, order='C', ndmin=2)
    nt, pt = t_arr.shape[0], t_arr.shape[1]
    nx, px = x_arr.shape[0], x_arr.shape[1]

    if pt != px:
        raise ValueError("t, x: shape mismatch")
        
    k_arr = np.empty((nt, nx), dtype=np.float)
    for i in range(nt):
        for j in range(nx):
            k_arr[i, j] = polynomial(<double *> t_arr.data + (i * pt),
                <double *> x_arr.data + (j * px), pt, gamma, b, d)
    
    if nt == 1:
        return k_arr[0]
    else:
        return k_arr


@cython.boundscheck(False)
def kernel_gaussian(t, x, sigma=1.0):
    cdef np.ndarray[np.float64_t, ndim=2] t_arr
    cdef np.ndarray[np.float64_t, ndim=2] x_arr
    cdef np.ndarray[np.float64_t, ndim=2] k_arr
    cdef int i, j, nt, nx, pt, px
    
    t_arr = np.array(t, dtype=np.float, order='C', ndmin=2)
    x_arr = np.array(x, dtype=np.float, order='C', ndmin=2)
    nt, pt = t_arr.shape[0], t_arr.shape[1]
    nx, px = x_arr.shape[0], x_arr.shape[1]

    if pt != px:
        raise ValueError("t, x: shape mismatch")
        
    k_arr = np.empty((nt, nx), dtype=np.float)
    for i in range(nt):
        for j in range(nx):
            k_arr[i, j] = gaussian(<double *> t_arr.data + (i * pt),
                <double *> x_arr.data + (j * px), pt, sigma)

    if nt == 1:
        return k_arr[0]
    else:
        return k_arr
    

@cython.boundscheck(False)
def kernel_exponential(t, x, sigma=1.0):
    cdef np.ndarray[np.float64_t, ndim=2] t_arr
    cdef np.ndarray[np.float64_t, ndim=2] x_arr
    cdef np.ndarray[np.float64_t, ndim=2] k_arr
    cdef int i, j, nt, nx, pt, px
    
    t_arr = np.array(t, dtype=np.float, order='C', ndmin=2)
    x_arr = np.array(x, dtxpe=np.float, order='C', ndmin=2)
    nt, pt = t_arr.shape[0], t_arr.shape[1]
    nx, px = x_arr.shape[0], x_arr.shape[1]

    if pt != px:
        raise ValueError("t, x: shape mismatch")
        
    k_arr = np.empty((nt, nx), dtype=np.float)
    for i in range(nt):
        for j in range(nx):
            k_arr[i, j] = exponential(<double *> t_arr.data + (i * pt),
                <double *> x_arr.data + (j * px), pt, sigma)
    
    if nt == 1:
        return k_arr[0]
    else:
        return k_arr


@cython.boundscheck(False)
def kernel_sigmoid(t, x, gamma=1.0, b=1.0):
    cdef np.ndarray[np.float64_t, ndim=2] t_arr
    cdef np.ndarray[np.float64_t, ndim=2] x_arr
    cdef np.ndarray[np.float64_t, ndim=2] k_arr
    cdef int i, j, nt, nx, pt, px
    
    t_arr = np.array(t, dtype=np.float, order='C', ndmin=2)
    x_arr = np.array(x, dtype=np.float, order='C', ndmin=2)
    nt, pt = t_arr.shape[0], t_arr.shape[1]
    nx, px = x_arr.shape[0], x_arr.shape[1]

    if pt != px:
        raise ValueError("t, x: shape mismatch")
        
    k_arr = np.empty((nt, nx), dtype=np.float)
    for i in range(nt):
        for j in range(nx):
            k_arr[i, j] = sigmoid(<double *> t_arr.data + (i * pt),
                <double *> x_arr.data + (j * px), pt, gamma, b)
    
    if nt == 1:
        return k_arr[0]
    else:
        return k_arr
   

def kernel_center(Kt, K):
    """Kernel matrix centering.
    """
    
    Kt_arr = np.array(Kt, dtype=np.float)
    K_arr = np.array(K, dtype=np.float)

    if Kt_arr.ndim == 1:
        J1 = np.mean(Kt_arr)
    else:
        J1 = np.mean(Kt_arr, axis=1).reshape(-1, 1)

    J2 = np.mean(K_arr, axis=0)

    return Kt_arr - J1 - J2 + np.mean(K_arr)


