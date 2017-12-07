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
from kernel_class import *


__all__ = ['LDAC', 'DLDA', 'KFDAC']


class LDAC:
    """Linear Discriminant Analysis Classifier.
    """
    
    def __init__(self):
        """Initialization.
        """

        self._labels = None
        self._w = None
        self._bias = None
      
    def learn(self, x, y):
        """Learning method.

        :Parameters:
           x : 2d array_like object
              training data (N, P)
           y : 1d array_like object integer
              target values (N)
        """
        
        xarr = np.asarray(x, dtype=np.float)
        yarr = np.asarray(y, dtype=np.int)
        
        if xarr.ndim != 2:
            raise ValueError("x must be a 2d array_like object")
        
        if yarr.ndim != 1:
            raise ValueError("y must be an 1d array_like object")

        self._labels = np.unique(yarr)
        k = self._labels.shape[0]

        if k < 2:
            raise ValueError("number of classes must be >= 2")     
        
        p = np.empty(k, dtype=np.float)
        mu = np.empty((k, xarr.shape[1]), dtype=np.float)
        cov = np.zeros((xarr.shape[1], xarr.shape[1]), dtype=np.float)

        for i in range(k):
            wi = (yarr == self._labels[i])
            p[i] = np.sum(wi) / float(xarr.shape[0])
            mu[i] = np.mean(xarr[wi], axis=0)
            xi = xarr[wi] - mu[i]
            cov += np.dot(xi.T, xi)
        cov /= float(xarr.shape[0] - k)
        covinv = np.linalg.inv(cov)
        
        self._w = np.empty((k, xarr.shape[1]), dtype=np.float)
        self._bias = np.empty(k, dtype=np.float)

        for i in range(k):           
            self._w[i] = np.dot(covinv, mu[i])
            self._bias[i] = - 0.5 * np.dot(mu[i], self._w[i]) + \
                np.log(p[i])

    def labels(self):
        """Outputs the name of labels.
        """
        
        return self._labels
        
    def w(self):
        """Returns the coefficients.
        For multiclass classification this method returns a 2d 
        numpy array where w[i] contains the coefficients of label i.
        For binary classification an 1d numpy array (w_1 - w_0) 
        is returned.
        """
        
        if self._w is None:
            raise ValueError("no model computed.")

        if self._labels.shape[0] == 2:
            return self._w[1] - self._w[0]
        else:
            return self._w

    def bias(self):
        """Returns the bias.
        For multiclass classification this method returns a 1d 
        numpy array where b[i] contains the coefficients of label i. 
        For binary classification an float (b_1 - b_0) is returned.
        """
        
        if self._w is None:
            raise ValueError("no model computed.")
        
        if self._labels.shape[0] == 2:
            return self._bias[1] - self._bias[0]
        else:
            return self._bias

    def pred(self, t):
        """Does classification on test vector(s) `t`.
      
        :Parameters:
            t : 1d (one sample) or 2d array_like object
                test data ([M,] P)
            
        :Returns:        
            p : integer or 1d numpy array
                predicted class(es)
        """
        
        if self._w is None:
            raise ValueError("no model computed.")

        tarr = np.asarray(t, dtype=np.float)

        if tarr.ndim == 1:
            delta = np.empty(self._labels.shape[0], dtype=np.float)
            for i in range(self._labels.shape[0]):
                delta[i] = np.dot(tarr, self._w[i]) + self._bias[i]
            return self._labels[np.argmax(delta)]
        else:
            delta = np.empty((tarr.shape[0], self._labels.shape[0]),
                        dtype=np.float)
            for i in range(self._labels.shape[0]):
                delta[:, i] = np.dot(tarr, self._w[i]) + self._bias[i]
            return self._labels[np.argmax(delta, axis=1)]



class DLDA:
    """Diagonal Linear Discriminant Analysis classifier.
    The algorithm uses the procedure called Nearest Shrunken
    Centroids (NSC).
    """
    
    def __init__(self, delta):
        """Initialization.
        
        :Parameters:
           delta : float
              regularization parameter
        """

        self._delta = float(delta)
        self._xstd = None # s_j
        self._dprime = None # d'_kj
        self._xmprime = None # xbar'_kj
        self._p = None # class prior probability
        self._labels = None

    def learn(self, x, y):
        """Learning method.

        :Parameters:
           x : 2d array_like object
              training data (N, P)
           y : 1d array_like object integer
              target values (N)
        """
        
        xarr = np.asarray(x, dtype=np.float)
        yarr = np.asarray(y, dtype=np.int)
        
        if xarr.ndim != 2:
            raise ValueError("x must be a 2d array_like object")
        
        if yarr.ndim != 1:
            raise ValueError("y must be an 1d array_like object")

        self._labels = np.unique(yarr)
        k = self._labels.shape[0]

        if k < 2:
            raise ValueError("number of classes must be >= 2")
        
        xm = np.mean(xarr, axis=0)
        self._xstd = np.std(xarr, axis=0, ddof=1)
        s0 = np.median(self._xstd)
        self._dprime = np.empty((k, xarr.shape[1]), dtype=np.float)
        self._xmprime = np.empty((k, xarr.shape[1]), dtype=np.float)
        n = yarr.shape[0]
        self._p = np.empty(k, dtype=np.float)

        for i in range(k):
            yi = (yarr == self._labels[i])
            xim = np.mean(xarr[yi], axis=0)
            nk = np.sum(yi)
            mk = np.sqrt(nk**-1 - n**-1)
            d = (xim - xm) / (mk * (self._xstd + s0))
            
            # soft thresholding
            tmp = np.abs(d) - self._delta
            tmp[tmp<0] = 0.0
            self._dprime[i] = np.sign(d) * tmp
            
            self._xmprime[i] = xm + (mk * (self._xstd + s0) * self._dprime[i])
            self._p[i] = float(nk) / float(n)

    def labels(self):
        """Outputs the name of labels.
        """
        
        return self._labels
        
    def sel(self):
        """Returns the most important features (the features that 
        have a nonzero dprime for at least one of the classes).
        """

        return np.where(np.sum(self._dprime, axis=0) != 0)[0]

    def dprime(self):
        """Return the dprime d'_kj (C, P), where C is the
        number of classes.
        """
        
        return self._dprime

    def _score(self, x):
        """Return the discriminant score"""

        return - np.sum((x-self._xmprime)**2/self._xstd**2,
                        axis=1) + (2 * np.log(self._p))

    def _prob(self, x):
        """Return the probability estimates"""
        
        score = self._score(x)
        tmp = np.exp(score * 0.5)
        return tmp / np.sum(tmp)
        
    def pred(self, t):
        """Does classification on test vector(s) t.
      
        :Parameters:
           t : 1d (one sample) or 2d array_like object
              test data ([M,] P)
            
        :Returns:        
           p : int or 1d numpy array
              the predicted class(es) for t is returned.
        """
        
        if self._xmprime is None:
            raise ValueError("no model computed.")
        
        tarr = np.asarray(t, dtype=np.float)
        
        if tarr.ndim == 1:
            return self._labels[np.argmax(self._score(tarr))]
        else:
            ret = np.empty(tarr.shape[0], dtype=np.int)
            for i in range(tarr.shape[0]):
                ret[i] = self._labels[np.argmax(self._score(tarr[i]))]
            return ret
        
    def prob(self, t):
        """For each sample returns C (number of classes)
        probability estimates.
        """

        if self._xmprime is None:
            raise ValueError("no model computed.")
        
        tarr = np.asarray(t, dtype=np.float)

        if tarr.ndim == 1:
            return self._prob(tarr)
        else:
            ret = np.empty((tarr.shape[0], self._labels.shape[0]),
                dtype=np.float)
            for i in range(tarr.shape[0]):
                ret[i] = self._prob(tarr[i])
            return ret


class KFDAC:
    """Kernel Fisher Discriminant Analysis Classifier (binary classifier).

    The bias term (b) is computed as in [Gavin03]_.

    .. [Gavin03] Gavin C. et al. Efficient Cross-Validation of Kernel Fisher Discriminant Classifers. ESANN'2003 proceedings - European Symposium on Artificial Neural Networks, 2003.
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

        self._lmb = float(lmb)
        self._kernel = kernel
        self._labels = None
        self._alpha = None
        self._b = None
        self._x = None
      
    def learn(self, K, y):
        """Learning method.

        :Parameters:
           K: 2d array_like object
              precomputed training kernel matrix (if kernel=None);
              training data in input space (if kernel is a Kernel object)
           y : 1d array_like object integer (N)
              class labels (only two classes)
        """

        Karr = np.asarray(K, dtype=np.float)
        yarr = np.asarray(y, dtype=np.int)

        if Karr.ndim != 2:
            raise ValueError("K must be a 2d array_like object")

        if yarr.ndim != 1:
            raise ValueError("y must be an 1d array_like object")
        
        if Karr.shape[0] != yarr.shape[0]:
            raise ValueError("K, y shape mismatch")

        if self._kernel is None:
            if Karr.shape[0] != Karr.shape[1]:
                raise ValueError("K must be a square matrix")
        else:
            self._x = Karr.copy()
            Karr = self._kernel.kernel(Karr, Karr)

        self._labels = np.unique(yarr)
        if self._labels.shape[0] != 2:
            raise ValueError("number of classes must be = 2")
        
        n = yarr.shape[0]
        
        idx1 = np.where(yarr==self._labels[0])[0]
        idx2 = np.where(yarr==self._labels[1])[0]
        n1 = idx1.shape[0]
        n2 = idx2.shape[0]
        
        K1, K2 = Karr[:, idx1], Karr[:, idx2]
        
        N1 = np.dot(np.dot(K1, np.eye(n1) - (1 / float(n1))), K1.T)
        N2 = np.dot(np.dot(K2, np.eye(n2) - (1 / float(n2))), K2.T)
        N = N1 + N2 + np.diag(np.repeat(self._lmb, n))
        Ni = np.linalg.inv(N)

        m1 = np.sum(K1, axis=1) / float(n1)
        m2 = np.sum(K2, axis=1) / float(n2)
        d = (m1 - m2)
        M = np.dot(d.reshape(-1, 1), d.reshape(1, -1))

        self._alpha = np.linalg.solve(N, d)
        self._b = - np.dot(self._alpha, (n1 * m1 + n2 * m2) / float(n))

    def labels(self):
        """Outputs the name of labels.
        """
        
        return self._labels
        
    def pred(self, Kt):
        """Compute the predicted response.
      
        :Parameters:
           Kt : 1d or 2d array_like object
              precomputed test kernel matrix. (if kernel=None);
              test data in input space (if kernel is a Kernel object).
            
        :Returns:        
            p : integer or 1d numpy array
                the predicted class(es)
        """

        
        if self._alpha is None:
            raise ValueError("no model computed; run learn()")

        Ktarr = np.asarray(Kt, dtype=np.float)
        if self._kernel is not None:
            Ktarr = self._kernel.kernel(Ktarr, self._x)

        try:
            s = np.sign(np.dot(self._alpha, Ktarr.T) + self._b)
        except ValueError:
            raise ValueError("Kt, alpha: shape mismatch")

        return np.where(s==1, self._labels[0], self._labels[1]) \
            .astype(np.int)

    def alpha(self):
        """Return alpha.
        """
        return self._alpha

    def b(self):
        """Return b.
        """
        return self._b
