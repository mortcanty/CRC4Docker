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
import scipy.linalg as spla
from ridge import ridge_base
from ols import ols_base
from kernel_class import *

import sys
if sys.version >= '3':
    from . import kernel
else:
    import kernel

__all__ = ['LDA', 'SRDA', 'KFDA', 'PCA', 'PCAFast', 'KPCA']


def proj(u, v):
    """(<v, u> / <u, u>) u
    """

    return (np.dot(v, u) / np.dot(u, u)) * u


def gso(v, norm=False):
    """Gram-Schmidt orthogonalization.
    Vectors v_1, ..., v_k are stored by rows.
    """
    
    for j in range(v.shape[0]):
        for i in range(j):
            v[j] = v[j] - proj(v[i], v[j])
        
        if norm:
            v[j] /= np.linalg.norm(v[j])


def lda(xarr, yarr):
    """Linear Discriminant Analysis.
    
    Returns the transformation matrix `coeff` (P, C-1),
    where `x` is a matrix (N,P) and C is the number of
    classes. Each column of `x` represents a variable, 
    while the rows contain observations. Each column of 
    `coeff` contains coefficients for one transformation
    vector.
    
    Sample(s) can be embedded into the C-1 dimensional space
    by z = x coeff (z = np.dot(x, coeff)).

    :Parameters:
       x : 2d array_like object (N, P)
          data matrix
       y : 1d array_like object integer (N)
          class labels
    
    :Returns:
       coeff: 2d numpy array (P, P)
          transformation matrix.
    """

    n, p = xarr.shape[0], xarr.shape[1]
    labels = np.unique(yarr)
    
    sw = np.zeros((p, p), dtype=np.float)   
    for i in labels:
        idx = np.where(yarr==i)[0]
        sw += np.cov(xarr[idx], rowvar=0) * \
            (idx.shape[0] - 1)
    st = np.cov(xarr, rowvar=0) * (n - 1)

    sb = st - sw
    evals, evecs = spla.eig(sb, sw, overwrite_a=True,
                            overwrite_b=True)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evecs = evecs[:, :labels.shape[0]-1]
    
    return evecs


def srda(xarr, yarr, alpha):
    """Spectral Regression Discriminant Analysis.

    Returns the (P, C-1) transformation matrix, where 
    `x` is a matrix (N,P) and C is the number of classes.
    Each column of `x` represents a variable, while the 
    rows contain observations. `x` must be centered 
    (subtracting the empirical mean vector from each column 
    of`x`).

    Sample(s) can be embedded into the C-1 dimensional space
    by z = x coeff (z = np.dot(x, coeff)).

    :Parameters:
       x : 2d array_like object
          training data (N, P)
       y : 1d array_like object integer
          target values (N)
       alpha : float (>=0)
          regularization parameter

    :Returns:
       coeff : 2d numpy array (P, C-1)
          tranformation matrix
    """

    # Point 1 in section 4.2
    yu = np.unique(yarr)
    yk = np.zeros((yu.shape[0]+1, yarr.shape[0]), dtype=np.float)
    yk[0] = 1.
    for i in range(1, yk.shape[0]):
        yk[i][yarr==yu[i-1]] = 1.
    gso(yk, norm=False) # orthogonalize yk
    yk = yk[1:-1]
    
    # Point 2 in section 4.2
    ak = np.empty((yk.shape[0], xarr.shape[1]), dtype=np.float)
    for i in range(yk.shape[0]):
        ak[i] = ridge_base(xarr, yk[i], alpha)

    return ak.T


def pca(xarr, method='svd'):
    """Principal Component Analysis.
    
    Returns the principal component coefficients `coeff`(K,K) 
    and the corresponding eigenvalues (K) of the covariance 
    matrix of `x` (N,P) sorted by decreasing eigenvalue, where 
    K=min(N,P). Each column of `x` represents a variable,  
    while the rows contain observations. Each column of `coeff` 
    contains coefficients for one principal component.
    
    Sample(s) can be embedded into the M (<=K) dimensional
    space by z = x coeff_M (z = np.dot(x, coeff[:, :M])).

    :Parameters:
       x : 2d numpy array (N, P)
          data matrix
       method : str
          'svd' or 'cov'
    
    :Returns:
       coeff, evals : 2d numpy array (K, K), 1d numpy array (K)
          principal component coefficients (eigenvectors of
          the covariance matrix of x) and eigenvalues sorted by 
          decreasing eigenvalue.
    """


    n, p = xarr.shape
    
    if method == 'svd':
        x_h = (xarr - np.mean(xarr, axis=0)) / np.sqrt(n - 1)
        u, s, v = np.linalg.svd(x_h.T, full_matrices=False)
        evecs = u
        evals = s**2
    elif method == 'cov':
        k = np.min((n, p))
        C = np.cov(xarr, rowvar=0)
        evals, evecs = np.linalg.eigh(C)
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]
        evals = evals[idx]
        evecs = evecs[:, :k]
        evals = evals[:k]
    else:
        raise ValueError("method must be 'svd' or 'cov'")

    return evecs, evals
        

def pca_fast(xarr, m, eps):
    """Fast principal component analysis using the fixed-point
    algorithm.
    
    Returns the first `m` principal component coefficients
    `coeff` (P, M). Each column of `x` represents a variable,  
    while the rows contain observations. Each column of `coeff` 
    contains coefficients for one principal component.

    Sample(s) can be embedded into the m (<=P) dimensional space 
    by z = x coeff (z = np.dot(X,  coeff)).

    :Parameters:
       x : 2d numpy array (N, P)
          data matrix
       m : integer (0 < m <= P) 
          the number of principal axes or eigenvectors required
       eps : float (> 0)
          tolerance error
    
    :Returns:
       coeff : 2d numpy array (P, H)
          principal component coefficients
    """
    
    m = int(m)

    np.random.seed(0)
    evecs = np.random.rand(m, xarr.shape[1])

    C = np.cov(xarr, rowvar=0)    
    for i in range(0, m):
        while True:
            evecs_old = np.copy(evecs[i])
            evecs[i] = np.dot(C, evecs[i])
            
            # Gram-Schmidt orthogonalization
            a = np.dot(evecs[i], evecs[:i].T).reshape(-1, 1)
            b = a  * evecs[:i]
            evecs[i] -= np.sum(b, axis=0) # if i=0 sum is 0
            
            # Normalization
            evecs[i] = evecs[i] / np.linalg.norm(evecs[i])
            
            # convergence criteria
            if np.abs(np.dot(evecs[i], evecs_old) - 1) < eps:
                break

    return evecs.T
      

def lda_fast(xarr, yarr):
    """Fast implementation of Linear Discriminant Analysis.
    
    Returns the (P, C-1) transformation matrix, where 
    `x` is a centered matrix (N,P) and C is the number of classes.
    Each column of `x` represents a variable, while the 
    rows contain observations. `x` must be centered 
    (subtracting the empirical mean vector from each column 
    of`x`).

    :Parameters:
       x : 2d array_like object
          training data (N, P)
       y : 1d array_like object integer
          target values (N)
    
    :Returns:
       A : 2d numpy array (P, C-1)
          tranformation matrix
    """

    yu = np.unique(yarr)
    yk = np.zeros((yu.shape[0]+1, yarr.shape[0]), dtype=np.float)
    yk[0] = 1.
    for i in range(1, yk.shape[0]):
        yk[i][yarr==yu[i-1]] = 1.
    gso(yk, norm=False) # orthogonalize yk
    yk = yk[1:-1]
    
    ak = np.empty((yk.shape[0], xarr.shape[1]), dtype=np.float)
    for i in range(yk.shape[0]):
        ak[i], _ = ols_base(xarr, yk[i], -1)

    return ak.T


def kpca(K):
    """Kernel Principal Component Analysis, PCA in 
    a kernel-defined feature space making use of the
    dual representation.
    
    Returns the kernel principal component coefficients 
    `coeff` (N, N) computed as :math:`\lambda^{-1/2} \mathbf{v}_j`
    where :math:`\lambda` and :math:`\mathbf{v}` are the ordered
    eigenvalues and the corresponding eigenvector of the centered 
    kernel matrix K.
    
    Sample(s) can be embedded into the G (<=N) dimensional space
    by z = K coeff_G (z = np.dot(K, coeff[:, :G])).

    :Parameters:
       K: 2d array_like object (N,N)
          precomputed centered kernel matrix
        
    :Returns:
       coeff, evals: 2d numpy array (N,N), 1d numpy array (N)
          kernel principal component coefficients, eigenvalues
          sorted by decreasing eigenvalue.
    """
    
    evals, evecs = np.linalg.eigh(K)
    idx = np.argsort(evals)
    idx = idx[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]
    
    for i in range(len(evals)):
        evecs[:, i] /= np.sqrt(evals[i])
   
    return evecs, evals


def kfda(Karr, yarr, lmb=0.001):
    """Kernel Fisher Discriminant Analysis.
    
    Returns the transformation matrix `coeff` (N,1),
    where `K` is a the kernel matrix (N,N) and y
    is the class labels (the alghoritm works only with 2
    classes).
   
    :Parameters:
       K: 2d array_like object (N, N)
          precomputed kernel matrix
       y : 1d array_like object integer (N)
          class labels
       lmb : float (>= 0.0)
          regularization parameter

    :Returns:
       coeff: 2d numpy array (N,1)
          kernel fisher coefficients.
    """  

    labels = np.unique(yarr)
    n = yarr.shape[0]

    idx1 = np.where(yarr==labels[0])[0]
    idx2 = np.where(yarr==labels[1])[0]
    n1 = idx1.shape[0]
    n2 = idx2.shape[0]
    
    K1, K2 = Karr[:, idx1], Karr[:, idx2]
    
    N1 = np.dot(np.dot(K1, np.eye(n1) - (1 / float(n1))), K1.T)
    N2 = np.dot(np.dot(K2, np.eye(n2) - (1 / float(n2))), K2.T)
    N = N1 + N2 + np.diag(np.repeat(lmb, n))

    M1 = np.sum(K1, axis=1) / float(n1)
    M2 = np.sum(K2, axis=1) / float(n2)
    M = M1 - M2
    
    coeff = np.linalg.solve(N, M).reshape(-1, 1)
            
    return coeff
 

class LDA:
    """Linear Discriminant Analysis.
    """
    
    def __init__(self, method='cov'):
        """Initialization.
        
        :Parameters:
           method : str
              'cov' or 'fast'
        """
        
        self._coeff = None
        self._mean = None

        if method not in ['cov', 'fast']:
            raise ValueError("method must be 'cov' or 'fast'")

        self._method = method

    def learn(self, x, y):
        """Computes the transformation matrix.
        `x` is a matrix (N,P) and `y` is a vector containing
        the class labels. Each column of `x` represents a 
        variable, while the rows contain observations.
        """

        xarr = np.asarray(x, dtype=np.float)
        yarr = np.asarray(y, dtype=np.int)
        
        if xarr.ndim != 2:
            raise ValueError("x must be a 2d array_like object")
        
        if yarr.ndim != 1:
            raise ValueError("y must be an 1d array_like object")
        
        if xarr.shape[0] != yarr.shape[0]:
            raise ValueError("x, y shape mismatch")
        
        self._mean = np.mean(xarr, axis=0)
        
        if self._method == 'cov':
            self._coeff = lda(xarr, yarr)
        elif self._method == 'fast':
            self._coeff = lda_fast(xarr-self._mean, yarr)

    def transform(self, t):
        """Embed `t` (M,P) into the C-1 dimensional space.
        Returns a (M,C-1) matrix.
        """
        if self._coeff is None:
            raise ValueError("no model computed")

        tarr = np.asarray(t, dtype=np.float)
        
        try:
            return np.dot(tarr-self._mean, self._coeff)
        except:
            ValueError("t, coeff: shape mismatch")

    def coeff(self):
        """Returns the tranformation matrix (P,C-1), where
        C is the number of classes. Each column contains 
        coefficients for one transformation vector.
        """

        return self._coeff


class SRDA:
    """Spectral Regression Discriminant Analysis.
    """
    
    def __init__(self, alpha=0.001):
        """Initialization.

        :Parameters:
           alpha : float (>=0)
              regularization parameter
        """
        
        self._coeff = None
        self._mean = None
        self._alpha = alpha

    def learn(self, x, y):
        """Computes the transformation matrix.
        `x` is a matrix (N,P) and `y` is a vector containing
        the class labels. Each column of `x` represents a 
        variable, while the rows contain observations.
        """

        xarr = np.asarray(x, dtype=np.float)
        yarr = np.asarray(y, dtype=np.int)
        
        if xarr.ndim != 2:
            raise ValueError("x must be a 2d array_like object")
        
        if yarr.ndim != 1:
            raise ValueError("y must be an 1d array_like object")
        
        if xarr.shape[0] != yarr.shape[0]:
            raise ValueError("x, y shape mismatch")
        
        self._mean = np.mean(xarr, axis=0)
        self._coeff = srda(xarr-self._mean, yarr, self._alpha)

    def transform(self, t):
        """Embed t (M,P) into the C-1 dimensional space.
        Returns a (M,C-1) matrix.
        """

        if self._coeff is None:
            raise ValueError("no model computed")

        tarr = np.asarray(t, dtype=np.float)
        
        try:
            return np.dot(tarr-self._mean, self._coeff)
        except:
            ValueError("t, coeff: shape mismatch")

    def coeff(self):
        """Returns the tranformation matrix (P,C-1), where
        C is the number of classes. Each column contains 
        coefficients for one transformation vector.
        """

        return self._coeff


class KFDA:
    """Kernel Fisher Discriminant Analysis.
    """
    
    def __init__(self, lmb=0.001, kernel=None):
        """Initialization.
        
        :Parameters:
           lmb : float (>= 0.0)
              regularization parameter
           kernel : None or mlpy.Kernel object.
              if kernel is None, K and Kt in .learn()
              and in .transform() methods must be precomputed kernel 
              matricies, else K and Kt must be training (resp. 
              test) data in input space.
        """

        
        if kernel is not None:
            if not isinstance(kernel, Kernel):
                raise ValueError("kernel must be None or a mlpy.Kernel object")
    
        self._kernel = kernel
        self._x = None
        self._coeff = None
        self._lmb = lmb

    def learn(self, K, y):
        """Computes the transformation vector.

        :Parameters:
           K: 2d array_like object
              precomputed training kernel matrix (if kernel=None);
              training data in input space (if kernel is a Kernel object)
           y : 1d array_like object integer (N)
              class labels (only two classes)
        """

        Karr = np.array(K, dtype=np.float)
        yarr = np.asarray(y, dtype=np.int)
        if yarr.ndim != 1:
            raise ValueError("y must be an 1d array_like object")

        if self._kernel is None:
            if Karr.shape[0] != Karr.shape[1]:
                raise ValueError("K must be a square matrix")
        else:
            self._x = Karr.copy()
            Karr = self._kernel.kernel(Karr, Karr)

        labels = np.unique(yarr)
        if labels.shape[0] != 2:
            raise ValueError("number of classes must be = 2")
        
        self._coeff = kfda(Karr, yarr, self._lmb)

    def transform(self, Kt):
        """Embed Kt into the 1d kernel fisher space.
        
        :Parameters:
           Kt : 1d or 2d array_like object
              precomputed test kernel matrix. (if kernel=None);
              test data in input space (if kernel is a Kernel object).
        """

        if self._coeff is None:
            raise ValueError("no model computed")

        Ktarr = np.asarray(Kt, dtype=np.float)
        if self._kernel is not None:
            Ktarr = self._kernel.kernel(Ktarr, self._x)

        try:
            return np.dot(Ktarr, self._coeff)
        except:
            ValueError("Kt, coeff: shape mismatch")

    def coeff(self):
        """Returns the tranformation vector (N,1).
        """

        return self._coeff


class PCA:
    """Principal Component Analysis.
    """
    
    def __init__(self, method='svd', whiten=False):
        """Initialization.
        
        :Parameters:
           method : str
              method, 'svd' or 'cov'
           whiten : bool
              whitening. The eigenvectors will be scaled
              by eigenvalues**-(1/2)
        """
        
        self._coeff = None
        self._coeff_inv = None
        self._evals = None
        self._mean = None
        self._method = method
        self._whiten = whiten        

    def learn(self, x):
        """Compute the principal component coefficients.
        `x` is a matrix (N,P). Each column of `x` represents a 
        variable, while the rows contain observations.
        """

        xarr = np.asarray(x, dtype=np.float)
        if xarr.ndim != 2:
            raise ValueError("x must be a 2d array_like object")
        
        self._mean = np.mean(xarr, axis=0)
        self._coeff, self._evals = pca(x, method=self._method)

        if self._whiten:
            self._coeff_inv = np.empty((self._coeff.shape[1], 
                self._coeff.shape[0]), dtype=np.float)
            
            for i in range(len(self._evals)):
                eval_sqrt = np.sqrt(self._evals[i])
                self._coeff_inv[i] = self._coeff[:, i] * \
                    eval_sqrt
                self._coeff[:, i] /= eval_sqrt
        else:
            self._coeff_inv = self._coeff.T

    def transform(self, t, k=None):
        """Embed `t` (M,P) into the k dimensional subspace.
        Returns a (M,K) matrix. If `k` =None will be set to 
        min(N,P)
        """

        if self._coeff is None:
            raise ValueError("no PCA computed")
        
        if k == None:
            k = self._coeff.shape[1]

        if k < 1 or k > self._coeff.shape[1]:
            raise ValueError("k must be in [1, %d] or None" % \
                                 self._coeff.shape[1])

        tarr = np.asarray(t, dtype=np.float)

        try:
            return np.dot(tarr-self._mean, self._coeff[:, :k])
        except:
            raise ValueError("t, coeff: shape mismatch")
            
    def transform_inv(self, z):
        """Transform data back to its original space,
        where `z` is a (M,K) matrix. Returns a (M,P) matrix.
        """

        if self._coeff is None:
            raise ValueError("no PCA computed")

        zarr = np.asarray(z, dtype=np.float)

        return np.dot(zarr, self._coeff_inv[:zarr.shape[1]]) +\
            self._mean
        
    def coeff(self):
        """Returns the tranformation matrix (P,L), where
        L=min(N,P), sorted by decreasing eigenvalue.
        Each column contains coefficients for one principal 
        component.
        """
        
        return self._coeff
    
    def coeff_inv(self):
        """Returns the inverse of tranformation matrix (L,P),
        where L=min(N,P), sorted by decreasing eigenvalue.
        """
        
        return self._coeff_inv

    def evals(self):
        """Returns sorted eigenvalues (L), where L=min(N,P).
        """
        
        return self._evals


class PCAFast:
    """Fast Principal Component Analysis.
    """
    
    def __init__(self, k=2, eps=0.01):
        """Initialization.
        
        :Parameters:
           k : integer
              the number of principal axes or eigenvectors required
           eps : float (> 0)
              tolerance error
        """
        
        self._coeff = None
        self._coeff_inv = None
        self._mean = None
        self._k = k
        self._eps = eps        

    def learn(self, x):
        """Compute the firsts `k` principal component coefficients.
        `x` is a matrix (N,P). Each column of `x` represents a 
        variable, while the rows contain observations.
        """

        xarr = np.asarray(x, dtype=np.float)
        if xarr.ndim != 2:
            raise ValueError("x must be a 2d array_like object")
        
        self._mean = np.mean(xarr, axis=0)
        self._coeff = pca_fast(xarr, m=self._k, eps=self._eps)
        self._coeff_inv = self._coeff.T

    def transform(self, t):
        """Embed t (M,P) into the `k` dimensional subspace.
        Returns a (M,K) matrix.
        """

        if self._coeff is None:
            raise ValueError("no PCA computed")

        tarr = np.asarray(t, dtype=np.float)

        try:
            return np.dot(tarr-self._mean, self._coeff)
        except:
            raise ValueError("t, coeff: shape mismatch")
            
    def transform_inv(self, z):
        """Transform data back to its original space,
        where `z` is a (M,K) matrix. Returns a (M,P) matrix.
        """

        if self._coeff is None:
            raise ValueError("no PCA computed")

        zarr = np.asarray(z, dtype=np.float)
        return np.dot(zarr, self._coeff_inv) + self._mean
        
    def coeff(self):
        """Returns the tranformation matrix (P,K) sorted by 
        decreasing eigenvalue.
        Each column contains coefficients for one principal 
        component.
        """
        
        return self._coeff
    
    def coeff_inv(self):
        """Returns the inverse of tranformation matrix (K,P),
        sorted by decreasing eigenvalue.
        """
        
        return self._coeff_inv


class KPCA:
    """Kernel Principal Component Analysis.
    """
    
    def __init__(self, kernel=None):
        """Initialization.
        
        :Parameters:
           kernel : None or mlpy.Kernel object.
              if kernel is None, K and Kt in .learn()
              and in .transform() methods must be precomputed kernel 
              matricies, else K and Kt must be training (resp. 
              test) data in input space.
        """

        if kernel is not None:
            if not isinstance(kernel, Kernel):
                raise ValueError("kernel must be None or a mlpy.Kernel object")
        
        self._coeff = None
        self._evals = None
        self._K = None
        self._kernel = kernel
        self._x = None

    def learn(self, K):
        """Compute the kernel principal component coefficients.

        :Parameters:
           K: 2d array_like object
              precomputed training kernel matrix (if kernel=None);
              training data in input space (if kernel is a Kernel object)
        """

        Karr = np.asarray(K, dtype=np.float)
        if Karr.ndim != 2:
            raise ValueError("K must be a 2d array_like object")
        
        if self._kernel is None:
            if Karr.shape[0] != Karr.shape[1]:
                raise ValueError("K must be a square matrix")
        else:
            self._x = Karr.copy()
            Karr = self._kernel.kernel(Karr, Karr)

        self._K = Karr.copy()
        Karr = kernel.kernel_center(Karr, Karr)
        self._coeff, self._evals = kpca(Karr)
       
    def transform(self, Kt, k=None):
        """Embed Kt into the `k` dimensional subspace.
        
        :Parameters:
           Kt : 1d or 2d array_like object
              precomputed test kernel matrix. (if kernel=None);
              test data in input space (if kernel is a Kernel object).
        """

        if self._coeff is None:
            raise ValueError("no KPCA computed")
        
        if k == None:
            k = self._coeff.shape[1]

        if k < 1 or k > self._coeff.shape[1]:
            raise ValueError("k must be in [1, %d] or None" % \
                                 self._coeff.shape[1])

        Ktarr = np.asarray(Kt, dtype=np.float)
        if self._kernel is not None:
            Ktarr = self._kernel.kernel(Ktarr, self._x)
        
        Ktarr = kernel.kernel_center(Ktarr, self._K)

        try:
            return np.dot(Ktarr, self._coeff[:, :k])
        except:
            raise ValueError("Kt, coeff: shape mismatch")
 
    def coeff(self):
        """Returns the tranformation matrix (N,N) sorted by 
        decreasing eigenvalue.
        """
        
        return self._coeff
    
    def evals(self):
        """Returns sorted eigenvalues (N).
        """
        
        return self._evals
   
