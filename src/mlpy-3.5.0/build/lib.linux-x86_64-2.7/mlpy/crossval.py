## Cross Validation Submodule

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

__all__ = ['cv_kfold', 'cv_random', 'cv_all']


import itertools
import numpy as np


def cv_kfold(n, k, strat=None, seed=0):
    """Returns train and test indexes for k-fold 
    cross-validation.
    
    :Parameters:
       n : int (n > 1) 
          number of indexes
       k : int (k > 1) 
          number of iterations (folds). The case `k` = `n`
          is known as leave-one-out cross-validation.
       strat : None or 1d array_like integer (of length `n`)
          labels for stratification. If `strat` is not None
          returns 'stratified' k-fold CV indexes, where
          each subsample has roughly the same label proportions
          of `strat`.
       seed : int
          random seed

    :Returns:
       idx: list of tuples
          list of `k` tuples containing the train and 
          test indexes
    
    Example:

    >>> import mlpy
    >>> idx = mlpy.cv_kfold(n=12, k=3)
    >>> for tr, ts in idx: tr, ts
    ... 
    (array([2, 8, 1, 7, 9, 3, 0, 5]), array([ 6, 11,  4, 10]))
    (array([ 6, 11,  4, 10,  9,  3,  0,  5]), array([2, 8, 1, 7]))
    (array([ 6, 11,  4, 10,  2,  8,  1,  7]), array([9, 3, 0, 5]))
    >>> strat = [0,0,0,0,0,0,0,0,1,1,1,1]
    >>> idx = mlpy.cv_kfold(12, k=4, strat=strat)
    >>> for tr, ts in idx: tr, ts
    ... 
    (array([ 1,  7,  3,  0,  5,  4,  8, 10,  9]), array([ 6,  2, 11]))
    (array([ 6,  2,  3,  0,  5,  4, 11, 10,  9]), array([1, 7, 8]))
    (array([ 6,  2,  1,  7,  5,  4, 11,  8,  9]), array([ 3,  0, 10]))
    (array([ 6,  2,  1,  7,  3,  0, 11,  8, 10]), array([5, 4, 9]))
    """

    
    if n < 2:
        raise ValueError("n must be > 1")

    if k < 2:
        raise ValueError("k must be > 1")
    
    if strat is not None:
        _strat = np.asarray(strat, dtype=np.int)
        if n != _strat.shape[0]:
            raise ValueError("a, strat: shape mismatch")
    else:
        _strat = np.zeros(n, dtype=np.int)
    
    labels = np.unique(_strat)

    # check k
    kmax = np.min([np.sum(l == _strat) for l in labels])
    if k > kmax:
        raise ValueError('k must be <= %d' % kmax)

    np.random.seed(seed)

    splits = []
    for l in labels:
        tmp = np.where(l == _strat)[0]
        np.random.shuffle(tmp)
        splits.append(np.array_split(tmp, k))

    idx = []
    for i in range(k):
        idx1, idx2 = [], []
        for j in range(len(splits)):
            idx1.extend(splits[j][:i] + splits[j][i+1:])
            idx2.extend(splits[j][i])
        idx.append((np.concatenate(idx1), np.asarray(idx2)))
            
    return idx


def cv_random(n, k, p, strat=None, seed=0):
    """Returns train and test indexes for random subsampling 
    cross-validation. The proportion of the train/test indexes
    is not dependent on the number of iterations `k`.
        
    :Parameters:
       n : int (n > 1)
          number of indexes
       k : int (k > 0) 
          number of iterations (folds)
       p : float (0 <= p <= 100) 
          percentage of indexes in test
       strat : None or  1d array_like integer (of length `n`)
          labels for stratification. If `strat` is not None
          returns 'stratified' random subsampling CV indexes,
          where each subsample has roughly the same label 
          proportions of `strat`.
       seed : int
          random seed
          
    :Returns:
        idx: list of tuples
          list of `k` tuples containing the train and 
          test indexes

    Example:
    
    >>> import mlpy
    >>> ap = mlpy.cv_random(n=12, k=4, p=30)
    >>> for tr, ts in ap: tr, ts
    ... 
    (array([ 6, 11,  4, 10,  2,  8,  1,  7,  9]), array([3, 0, 5]))
    (array([ 5,  2,  3,  4,  9,  0, 11,  7,  6]), array([ 1, 10,  8]))
    (array([ 6,  1, 10,  2,  7,  5, 11,  0,  3]), array([4, 9, 8]))
    (array([2, 4, 8, 9, 5, 6, 1, 0, 7]), array([10, 11,  3]))
    """

    if n < 2:
        raise ValueError("n must be > 1")

    if k < 2:
        raise ValueError("k must be > 1")
    
    if (p < 0) or (p > 100):
        raise ValueError("p must be in [0, 100]")

    if strat is not None:
        _strat = np.asarray(strat, dtype=np.int)
        if n != _strat.shape[0]:
            raise ValueError("a, strat: shape mismatch")
    else:
        _strat = np.zeros(n, dtype=np.int)

    labels = np.unique(_strat)

    # check p
    pmin = np.min([np.sum(l == _strat) for l in labels])**-1 * 100
    if p < pmin:
        raise ValueError('p must be >= %.3f' % pmin)
            
    np.random.seed(seed)

    idx = []
    for _ in range(k):        
        idx1, idx2 = [], []
        for l in labels:
            tmp = np.where(l == _strat)[0]
            g = tmp.shape[0] - int(0.01*p*tmp.shape[0])
            np.random.shuffle(tmp)
            idx1.append(tmp[:g])
            idx2.append(tmp[g:])
        
        idx.append((np.concatenate(idx1), np.concatenate(idx2)))

    return idx


def cv_all(n, p):
    """Returns train and test indexes for all-combinations 
    cross-validation.

    :Parameters:
       n : int (n > 1)
          number of indexes
       p : float (0 <= p <= 100) 
          percentage of indexes in test

    :Returns:
       idx : list of tuples
          list of tuples containing the train and 
          test indexes

    Example

    >>> import mlpy
    >>> idx = mlpy.cv_all(n=4, p=50)
    >>> for tr, ts in idx: tr, ts
    ... 
    (array([2, 3]), array([0, 1]))
    (array([1, 3]), array([0, 2]))
    (array([1, 2]), array([0, 3]))
    (array([0, 3]), array([1, 2]))
    (array([0, 2]), array([1, 3]))
    (array([0, 1]), array([2, 3]))
    >>> idx = mlpy.cv_all(a, 10) # ValueError: p must be >= 25.000
    """

    if n < 2:
        raise ValueError("n must be > 1")

    if (p < 0.0) or (p > 100.0):
        raise ValueError("p must be in [0, 100]")
    
    # check p
    pmin = n**-1 * 100
    if p < pmin:
        raise ValueError('p must be >= %.3f' % pmin)
    
    k = int(0.01 * p * n)
    a = np.arange(n)
    tmp = np.asarray(list(itertools.combinations(a, k)))

    idx = []
    for idx2 in tmp:
        idx1 = np.setdiff1d(a, idx2)
        idx.append((idx1, idx2))

    return idx
