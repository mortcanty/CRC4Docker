## This code is written by Davide Albanese, <albanese@fbk.eu>.
## (C) 2008 mlpy Developers.

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

__all__ = ['error', 'error_p', 'error_n', 'accuracy', 
           'sensitivity', 'specificity', 'ppv', 'npv', 
           'mcc', 'auc_wmw', 'mse', 'r2', 'r2_corr']


import numpy as np

"""
Compute metrics for assessing the performance of
classification/regression models.

The Confusion Matrix:

Total Samples       (ts) | Actual Positives (ap) | Actual Negatives (an)
------------------------------------------------------------------------
Predicted Positives (pp) | True Positives   (tp) | False Positives  (fp)
------------------------------------------------------------------------
Predicted Negatives (pn) | False Negatives  (fn) | True Negatives   (tn)
"""


def is_binary(x):
    
    ux = np.unique(x)
    for elem in ux:
        if elem not in [-1, 1]:
            return False
    return True

def true_pos(t, p):
    w = (t == 1)
    return (t[w] == p[w]).sum()

def true_neg(t, p):
    w = (t == -1)
    return (t[w] == p[w]).sum()

def false_pos(t, p):
    w = (t == -1)
    return (t[w] != p[w]).sum()

def false_neg(t, p):
    w = (t == 1)
    return (t[w] != p[w]).sum()


def error(t, p):
    """Error for binary and multiclass classification
    problems.

    :Parameters:
      t : 1d array_like object integer
        target values
      p : 1d array_like object integer
        predicted values

    :Returns:
      error : float, in range [0.0, 1.0]
    """
  
    tarr = np.asarray(t, dtype=np.int)
    parr = np.asarray(p, dtype=np.int)

    if tarr.shape[0] != parr.shape[0]:
        raise ValueError("t, p: shape mismatch")
        
    return (tarr != parr).sum() / float(tarr.shape[0])


def accuracy(t, p):
    """Accuracy for binary and multiclass classification
    problems.

    :Parameters:
      t : 1d array_like object integer
        target values
      p : 1d array_like object integer
        predicted values

    :Returns:
      accuracy : float, in range [0.0, 1.0]
    """
  
    tarr = np.asarray(t, dtype=np.int)
    parr = np.asarray(p, dtype=np.int)

    if tarr.shape[0] != parr.shape[0]:
        raise ValueError("t, p: shape mismatch")
        
    return (tarr == parr).sum() / float(tarr.shape[0])


def error_p(t, p):
    """Compute the positive error as:

    error_p = fn / ap
    
    Only binary classification problems with t[i] = -1/+1
    are allowed.

    :Parameters:
      t : 1d array_like object integer (-1/+1)
        target values
      p : 1d array_like object integer (-1/+1)
        predicted values

    :Returns:
      errorp : float, in range [0.0, 1.0]
    """

    tarr = np.asarray(t, dtype=np.int)
    parr = np.asarray(p, dtype=np.int)

    if tarr.shape[0] != parr.shape[0]:
        raise ValueError("t, p: shape mismatch")
    
    if not is_binary(tarr):
        raise ValueError("only binary classification problems"
            " with t[i] = -1/+1 are allowed.")

    fn = false_neg(tarr, parr)
    ap = float((true_pos(tarr, parr) + fn))
    
    if ap == 0:
        return 0.0

    return fn / ap


def error_n(t, p):
    """Compute the negative error as:

    error_n = fp / an
    
    Only binary classification problems with t[i] = -1/+1
    are allowed.

    :Parameters:
      t : 1d array_like object integer (-1/+1)
        target values
      p : 1d array_like object integer (-1/+1)
        predicted values

    :Returns:
      errorp : float, in range [0.0, 1.0]
    """

    tarr = np.asarray(t, dtype=np.int)
    parr = np.asarray(p, dtype=np.int)

    if tarr.shape[0] != parr.shape[0]:
        raise ValueError("t, p: shape mismatch")
    
    if not is_binary(tarr):
        raise ValueError("only binary classification problems"
            " with t[i] = -1/+1 are allowed.")

    fp = false_pos(tarr, parr)
    an = float((true_neg(tarr, parr) + fp))
    
    if an == 0:
        return 0.0

    return fp / an


def sensitivity(t, p):
    """Sensitivity, computed as:

    sensitivity = tp / ap
    
    Only binary classification problems with t[i] = -1/+1
    are allowed.

    :Parameters:
      t : 1d array_like object integer (-1/+1)
        target values
      p : 1d array_like object integer (-1/+1)
        predicted values

    :Returns:
      sensitivity : float, in range [0.0, 1.0]
    """

    tarr = np.asarray(t, dtype=np.int)
    parr = np.asarray(p, dtype=np.int)

    if tarr.shape[0] != parr.shape[0]:
        raise ValueError("t, p: shape mismatch")
    
    if not is_binary(tarr):
        raise ValueError("only binary classification problems"
            " with t[i] = -1/+1 are allowed.")

    tp = true_pos(tarr, parr)
    ap = float((tp + false_neg(tarr, parr)))
    
    if ap == 0:
        return 0.0

    return tp / ap


def specificity(t, p):
    """Specificity, computed as:

    specificity = tn / an
    
    Only binary classification problems with t[i] = -1/+1
    are allowed.

    :Parameters:
      t : 1d array_like object integer (-1/+1)
        target values
      p : 1d array_like object integer (-1/+1)
        predicted values

    :Returns:
      sensitivity : float, in range [0.0, 1.0]
    """

    tarr = np.asarray(t, dtype=np.int)
    parr = np.asarray(p, dtype=np.int)

    if tarr.shape[0] != parr.shape[0]:
        raise ValueError("t, p: shape mismatch")
    
    if not is_binary(tarr):
        raise ValueError("only binary classification problems"
            " with t[i] = -1/+1 are allowed.")
    
    tn = true_neg(tarr, parr)
    an = float((false_pos(tarr, parr) + tn))

    if an == 0:
        return 0.0

    return tn / an


def ppv(t, p):
    """Positive Predictive Value (PPV) computed as:

    ppv = tp / pp

    Only binary classification problems with t[i] = -1/+1
    are allowed.
    
    :Parameters:
      t : 1d array_like object integer (-1/+1)
        target values
      p : 1d array_like object integer (-1/+1)
        predicted values

    :Returns:
      PPV : float, in range [0.0, 1.0]
    """

    tarr = np.asarray(t, dtype=np.int)
    parr = np.asarray(p, dtype=np.int)

    if tarr.shape[0] != parr.shape[0]:
        raise ValueError("t, p: shape mismatch")
    
    if not is_binary(tarr):
        raise ValueError("only binary classification problems"
            " with t[i] = -1/+1 are allowed.")

    tp = true_pos(tarr, parr)
    pp = float((tp + false_pos(tarr, parr)))
    
    if pp == 0:
        return 0.0

    return tp / pp


def npv(t, p):
    """Negative Predictive Value (NPV), computed as:

    npv = tn / pn

    Only binary classification problems with t[i] = -1/+1
    are allowed.
    
    :Parameters:
      t : 1d array_like object integer (-1/+1)
        target values
      p : 1d array_like object integer (-1/+1)
        predicted values

    :Returns:
      NPV : float, in range [0.0, 1.0]
    """

    tarr = np.asarray(t, dtype=np.int)
    parr = np.asarray(p, dtype=np.int)

    if tarr.shape[0] != parr.shape[0]:
        raise ValueError("t, p: shape mismatch")
    
    if not is_binary(tarr):
        raise ValueError("only binary classification problems"
            " with t[i] = -1/+1 are allowed.")
       
    tn = true_neg(tarr, parr)
    pn = float((tn + false_neg(tarr, parr)))
    
    if pn == 0:
        return 0.0  

    return tn / pn


def mcc(t, p):
    """Matthews Correlation Coefficient (MCC), computed as:

    MCC = ((tp*tn)-(fp*fn)) / sqrt((tp+fn)*(tp+fp)*(tn+fn)*(tn+fp))
    
    Only binary classification problems with t[i] = -1/+1 are allowed.
    
    Returns a value between -1 and +1. A MCC of +1 represents 
    a perfect prediction, 0 an average random prediction and 
    -1 an inverse prediction.
    If any of the four sums in the denominator is zero,
    the denominator is set to one; this results in a Matthews
    Correlation Coefficient of zero, which can be shown to be
    the correct limiting value.

    :Parameters:
      t : 1d array_like object integer (-1/+1)
        target values
      p : 1d array_like object integer (-1/+1)
        predicted values

    :Returns:
      MCC : float, in range [-1.0, 1.0]
    """

    tarr = np.asarray(t, dtype=np.int)
    parr = np.asarray(p, dtype=np.int)

    if tarr.shape[0] != parr.shape[0]:
        raise ValueError("t, p: shape mismatch")
    
    if not is_binary(tarr):
        raise ValueError("only binary classification problems"
            " with t[i] = -1/+1 are allowed.")

    tp = true_pos(tarr, parr)
    tn = true_neg(tarr, parr)
    fp = false_pos(tarr, parr)  
    fn = false_neg(tarr, parr)

    den = np.sqrt((tp+fn)*(tp+fp)*(tn+fn)*(tn+fp))
    if den == 0.0:
        den = 1.0

    num = np.float((tp*tn)-(fp*fn))
    return num / den


def auc_wmw(t, p):
    """Compute the AUC by using the Wilcoxon-Mann-Whitney
    statistic. Only binary classification problems with
    t[i] = -1/+1 are allowed.

    :Parameters:
      t : 1d array_like object integer (-1/+1)
        target values
      p : 1d array_like object (negative/positive values)
        predicted values
      
    :Returns:
      AUC : float, in range [0.0, 1.0]
    """

    tarr = np.asarray(t, dtype=np.int)
    parr = np.asarray(p, dtype=np.float)

    if tarr.shape[0] != parr.shape[0]:
        raise ValueError("t, p: shape mismatch")
    
    if not is_binary(tarr):
        raise ValueError("only binary classification problems"
            " with t[i] = -1/+1 are allowed.")
    
    idxp = np.where(tarr ==  1)[0]
    idxn = np.where(tarr == -1)[0]
    
    auc = 0.0
    for i in idxp:
        for j in idxn:
            if (p[i] - p[j]) > 0.0:
                auc += 1.0
                
    return auc / float(idxp.shape[0] * idxn.shape[0])


def mse(t, p):
    """Mean Squared Error (MSE).

    :Parameters:
      t : 1d array_like object
        target values
      p : 1d array_like object
        predicted values
      
    :Returns:
      MSE : float
    """

    tarr = np.asarray(t, dtype=np.float)
    parr = np.asarray(p, dtype=np.float)

    if tarr.shape[0] != parr.shape[0]:
        raise ValueError("t, p: shape mismatch")

    n = tarr.shape[0]

    return np.sum((tarr - parr)**2) / n


def r2(t, p):
    """Coefficient of determination (R^2)
    computed as 1 - (sserr/sstot), where `sserr` is
    the sum of squares of residuals and `sstot` is
    the total sum of squares.
    
    :Parameters:
      t : 1d array_like object
        target values
      p : 1d array_like object
        predicted values
      
    :Returns:
      R^2 : float
    """

    tarr = np.asarray(t, dtype=np.float)
    parr = np.asarray(p, dtype=np.float)

    if tarr.shape[0] != parr.shape[0]:
        raise ValueError("t, p: shape mismatch")

    sserr = np.sum((tarr - parr)**2)
    sstot = np.sum((tarr - tarr.mean())**2)

    return 1. - (sserr / sstot)


def r2_corr(t, p):
    """Coefficient of determination (R^2)
    computed as square of the correlation
    coefficient.
    
    :Parameters:
      t : 1d array_like object
        target values
      p : 1d array_like object
        predicted values
      
    :Returns:
      R^2 : float
    """

    tarr = np.asarray(t, dtype=np.float)
    parr = np.asarray(p, dtype=np.float)

    if tarr.shape[0] != parr.shape[0]:
        raise ValueError("t, p: shape mismatch")

    return np.corrcoef(parr, tarr)[0,1]**2
