## Iterative RELIEF for Feature Weighting.

## This is an implementation of Iterative RELIEF algorithm described in:
## Yijun Sun. 'Iterative RELIEF for Feature Weightinig: Algorithms,
## Theories and Application'. In IEEE Transactions on Pattern Analysis
## and Machine Intelligence, 2006.
    
## This code is written by Davide Albanese, <albanese@fbk.eu>.
## (C) 2007 mlpy Developers.

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

__all__ = ['SigmaError', 'IRelief']

from numpy import *


class SigmaError(Exception):
    pass


def norm_w(x, w):
    """
    Compute sum_i( w[i] * |x[i]| ).

    See p. 7.
    """
    return (w * abs(x)).sum()


def norm(x, n):
    """
    Compute n-norm.
    """
    return (sum(abs(x)**n))**(1.0/float(n))


def kernel(d, sigma):
    """
    Kernel.

    See p. 7.
    """
    return exp(-d/float(sigma))  


def compute_M_H(y):
    """
    Compute sets M[n] = {i:1<=i<=N, y[i]!=y[n]}.
    Compute sets H[n] = {i:1<=i<=N, y[i]==y[n], i!=n}.

    See p. 6.
    """
    M, H = [], []
    for n in range(y.shape[0]):
        Mn = where(y != y[n])[0].tolist()
        M.append(Mn)
        Hn = where(y == y[n])[0]
        Hn = Hn[Hn != n].tolist()
        H.append(Hn)
    return (M, H)
    

def compute_distance_kernel(x, w, sigma):
    """
    Compute matrix dk[i][j] = f(||x[i] - x[j]||_w).

    See p. 7.
    """
    d = zeros((x.shape[0], x.shape[0]), dtype = float)
    for i in range(x.shape[0]):
        for j in range(i + 1, x.shape[0]):
            d[i][j] = norm_w(x[i]-x[j], w)
            d[j][i] = d[i][j]
    dk = kernel(d, sigma)
   
    return dk


def compute_prob(x, dist_k, i, n, indices):
    """
    See Eqs. (8), (9)
    """

    den = dist_k[n][indices].sum()    
    if den == 0.0:
        raise SigmaError("sigma (kernel parameter) too small")
    
    return dist_k[n][i] / float(den)


def compute_gn(x, dist_k, n, Mn):
    """
    See p. 7 and Eq. (10).
    """

    num = dist_k[n][Mn].sum()
    R = range(x.shape[0])
    R.remove(n)
    den = dist_k[n][R].sum()
    if den == 0.0:
        raise SigmaError("sigma (kernel parameter) too small")

    return 1.0 - (num / float(den))
       

def compute_w(x, w, M, H, sigma):
    """
    See Eq. (12).
    """

    N = x.shape[0]
    I = x.shape[1]

    # Compute ni
    ni = zeros(I, dtype = float)
    dist_k = compute_distance_kernel(x, w, sigma)
    for n in range(N):        
        m_n = zeros(I, dtype = float)
        h_n = zeros(I, dtype = float)
        for i in M[n]:
            a_in = compute_prob(x, dist_k, i, n, M[n])
            m_in = abs(x[n] - x[i])
            m_n += a_in * m_in
        for i in H[n]:
            b_in = compute_prob(x, dist_k, i, n, H[n])
            h_in = abs(x[n] - x[i])
            h_n += b_in * h_in        
        g_n = compute_gn(x, dist_k, n, M[n])
        ni += g_n * (m_n - h_n)            

    ni = ni / float(N)
        
    # Compute (ni)+ / ||(ni)+||_2
    ni_p = maximum(ni, 0.0)
    ni_p_norm2 = norm(ni_p, 2)
   
    return ni_p / ni_p_norm2


def compute_irelief(x, y, T, sigma, theta):
    """
    See I-RELIEF Algorithm at p. 8.
    """

    w_old = ones(x.shape[1]) / float(x.shape[1])
    M, H = compute_M_H(y)
    
    for t in range(T):
        w = compute_w(x, w_old, M, H, sigma) 
        stp = norm(w - w_old, 2)
        if stp < theta:
            break
        w_old = w
    return (w, t + 1)


class IRelief:
    """Iterative RELIEF for feature weighting.
    """
   
    def __init__(self, T=1000, sigma=1.0, theta=0.001):
        """
        :Parameters:
          T : integer (> 0)
            max loops
          sigma : float (> 0.0)
            kernel width
          theta : float (> 0.0)
            convergence parameter
        """

        if T <= 0:
            raise ValueError("T (max loops) must be > 0")
        if sigma <= 0.0:
            raise ValueError("sigma (kernel width) must be > 0.0")
        if theta <= 0.0:
            raise ValueError("theta (convergence parameter) must be > 0.0")
         
        self._T = T
        self._sigma = sigma
        self._theta = theta
        self._loops = None
        self._w = None

    def learn(self, x, y):
        """Compute the feature weights.

        :Parameters:
           x : 2d array_like object
              training data (N, P)
           y : 1d array_like object integer (only two classes)
              target values (N)

        :Raises:
           SigmaError
        """
        
        xarr = asarray(x, dtype=float)
        yarr = asarray(y, dtype=int)

        if xarr.ndim != 2:
            raise ValueError("x must be a 2d array_like object")
        
        if yarr.ndim != 1:
            raise ValueError("y must be an 1d array_like object")
        
        if xarr.shape[0] != yarr.shape[0]:
            raise ValueError("x, y: shape mismatch")

        if unique(y).shape[0] != 2:
            raise ValueError("number of classes must be = 2")

        self._w, self._loops = \
            compute_irelief(xarr, yarr, self._T, self._sigma, self._theta)

    def weights(self):
        """Returns the feature weights.
        """
        
        if self._w is None:
            raise ValueError("no model computed.")

        return self._w

    def loops(self):
        """Returns the number of loops.
        """

        if self._w is None:
            raise ValueError("no model computed.")

        return self._loops
        
