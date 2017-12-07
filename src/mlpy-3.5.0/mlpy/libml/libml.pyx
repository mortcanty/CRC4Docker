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

from clibml cimport *

np.import_array()


cdef class KNN:
    """k-Nearest Neighbor (euclidean distance).
    """
    cdef NearestNeighbor nn
    cdef int k
    cdef int *classes

    def __cinit__(self, k):
        """Initialization.
        
        :Parameters:
           k : int
              number of nearest neighbors
        """

        self.nn.x = NULL
        self.nn.y = NULL
        self.nn.classes = NULL
        self.k = int(k)
        self.classes = NULL
        
    def learn(self, x, y):
        """Learn method.
        
        :Parameters:	
           x : 2d array_like object (N,P)
              training data 
           y : 1d array_like integer 
              class labels
        """

        cdef int ret
        cdef np.ndarray[np.float_t, ndim=2] xarr
        cdef np.ndarray[np.int_t, ndim=1] yarr
        cdef np.ndarray[np.int32_t, ndim=1] ynew
        cdef double *xp
        cdef double **xpp
        cdef int i

        xarr = np.ascontiguousarray(x, dtype=np.float)
        yarr = np.ascontiguousarray(y, dtype=np.int)
        
        if self.k > xarr.shape[0]:
            raise ValueError("k must be smaller than number of samples")

        yu = np.unique(yarr)
        if yu.shape[0] <= 1:
            raise ValueError("y: number of classes must be >=2")

        self._free()
        
        # save original labels
        self.classes = <int *> malloc (yu.shape[0] * sizeof(int))
        for i in range(yu.shape[0]):
            self.classes[i] = yu[i]
        
        # transform labels
        ynew = np.empty(yarr.shape[0], dtype=np.int32)
        if yu.shape[0] == 2:
            cond_neg = (yarr == yu[0])
            cond_pos = (yarr == yu[1])
            ynew[cond_neg], ynew[cond_pos] = -1, 1
        else:
            for i in range(yu.shape[0]):
                cond = (yarr == yu[i])
                ynew[cond] = i + 1
    
        xp = <double *> xarr.data
        xpp = <double **> malloc (xarr.shape[0] * sizeof(double*))
        for i in range(xarr.shape[0]):
            xpp[i] = xp + (i * xarr.shape[1])
        
        ret = compute_nn(&self.nn, <int> xarr.shape[0], <int> xarr.shape[1],
                          xpp, <int *> ynew.data, self.k, DIST_EUCLIDEAN)
        free(xpp)

        if ret == 1:
            raise MemoryError("out of memory")
        
    def pred(self, t):
        """Predict KNN model on a test point(s).
        
        :Parameters:
           t : 1d or 2d array_like object ([M,] P)
              test point(s)
              
        :Returns:
	   p : int or 1d numpy array
              the predicted value(s). Retuns the smallest label
              minus one (KNN.labels()[0]-1) when the classification
              is not unique.
        """
        
        cdef int i
        cdef np.ndarray[np.float_t, ndim=1] tiarr
        cdef double *margin
        cdef double *tdata
        

        if self.nn.x is NULL:
            raise ValueError("no model computed")
        
        tarr = np.ascontiguousarray(t, dtype=np.float)
        if tarr.ndim > 2:
            raise ValueError("t must be an 1d or a 2d array_like object")

        if tarr.shape[-1] != self.nn.d:
            raise ValueError("t, model: shape mismatch")

        if tarr.ndim == 1:
            tiarr = tarr
            p = predict_nn(&self.nn, <double *> tiarr.data, &margin)
            free(margin)
            if p == -2:
                raise MemoryError("out of memory")
            elif p == 0:
                ret = self.classes[0] - 1
            else:
                if self.nn.nclasses == 2:
                    if p == -1: ret = self.classes[0]
                    else: ret = self.classes[1]
                else:
                    ret = self.classes[p-1]

        else:
            ret = np.empty(tarr.shape[0], dtype=np.int)
            for i in range(tarr.shape[0]):
                tiarr = tarr[i]
                tdata = <double *> tiarr.data
                p = predict_nn(&self.nn, tdata, &margin)
                free(margin)
                if p == -2:
                    raise MemoryError("out of memory")
                elif p == 0:
                    ret[i] = self.classes[0] - 1
                else:
                    if self.nn.nclasses == 2:
                        if p == -1: ret[i] = self.classes[0]
                        else: ret[i] = self.classes[1]
                    else:
                        ret[i] = self.classes[p-1]
            
        return ret
            
    def nclasses(self):
        """Returns the number of classes.
        """
        
        if self.nn.x is NULL:
            raise ValueError("no model computed")

        return self.nn.nclasses

    def labels(self):
        """Outputs the name of labels.
        """
        
        if self.nn.x is NULL:
            raise ValueError("no model computed")
        
        ret = np.empty(self.nn.nclasses, dtype=np.int)
        for i in range(self.nn.nclasses):
            ret[i] = self.classes[i]

        return ret

    def _free(self):
        if self.nn.x is not NULL:
            for i in range(self.nn.n):
                free(self.nn.x[i])
            free(self.nn.x)

        if self.nn.y is not NULL:
            free(self.nn.y)
        
        if self.nn.classes is not NULL:
            free(self.nn.classes)

        if self.classes is not NULL:
            free(self.classes)

    def __dealloc__(self):
        self._free()


cdef class ClassTree:
    """Classification Tree (gini index).
    """
    cdef Tree tree
    cdef int stumps
    cdef int minsize
    cdef int *classes
 
    def __cinit__(self, stumps=False, minsize=0):
        """Initialization.
        
        :Parameters:
           stumps : bool
              True: compute single split or False: standard tree
           minsize : int (>=0)
              minimum number of cases required to split a leaf
        """

        self.tree.x = NULL
        self.tree.y = NULL
        self.tree.classes = NULL
        self.tree.node = NULL
        self.classes = NULL

        self.stumps = int(bool(stumps))
        self.minsize = int(minsize)

        if self.minsize < 0:
            raise ValueError("minsize must be >= 0")
        
    def learn(self, x, y):
        """Learn method.
        
        :Parameters:	
           x : 2d array_like object (N x P)
              training data 
           y : 1d array_like integer 
              class labels
        """

        cdef int ret
        cdef np.ndarray[np.float_t, ndim=2] xarr
        cdef np.ndarray[np.int_t, ndim=1] yarr
        cdef np.ndarray[np.int32_t, ndim=1] ynew
        cdef double *xp
        cdef double **xpp
        cdef int i

        xarr = np.ascontiguousarray(x, dtype=np.float)
        yarr = np.ascontiguousarray(y, dtype=np.int)
       
        yu = np.unique(yarr)
        if yu.shape[0] <= 1:
            raise ValueError("y: number of classes must be >=2")
        
        self._free()

        # save original labels
        self.classes = <int *> malloc (yu.shape[0] * sizeof(int))
        for i in range(yu.shape[0]):
            self.classes[i] = yu[i]
            
        # transform labels
        ynew = np.empty(yarr.shape[0], dtype=np.int32)
        if yu.shape[0] == 2:
            cond_neg = (yarr == yu[0])
            cond_pos = (yarr == yu[1])
            ynew[cond_neg], ynew[cond_pos] = -1, 1
        else:
            for i in range(yu.shape[0]):
                cond = (yarr == yu[i])
                ynew[cond] = i + 1

        xp = <double *> xarr.data
        xpp = <double **> malloc (xarr.shape[0] * sizeof(double*))
        for i in range(xarr.shape[0]):
            xpp[i] = xp + (i * xarr.shape[1])
        
        ret = compute_tree(&self.tree, <int> xarr.shape[0], <int> xarr.shape[1],
                          xpp, <int *> ynew.data, self.stumps, self.minsize)
        free(xpp)

        if ret == 1:
            raise MemoryError("out of memory")
        
    def pred(self, t):
        """Predict Tree model on a test point(s).
        
        :Parameters:
           t : 1d or 2d array_like object ([M,] P)
              test point(s)
              
        :Returns:
	   p : int or 1d numpy array
              the predicted value(s). Retuns the smallest label
              minus one (ClassTree.labels()[0]-1) when the classification
              is not unique.          
        """ 
        
        cdef int i, p
        cdef np.ndarray[np.float_t, ndim=1] tiarr
        cdef double *margin
        cdef double *tdata
        

        if self.tree.x is NULL:
            raise ValueError("no model computed")
        
        tarr = np.ascontiguousarray(t, dtype=np.float)
        if tarr.ndim > 2:
            raise ValueError("t must be an 1d or a 2d array_like object")

        if tarr.shape[-1] != self.tree.d:
            raise ValueError("t, model: shape mismatch")

        if tarr.ndim == 1:
            tiarr = tarr
            p = predict_tree(&self.tree, <double *> tiarr.data, &margin)
            free(margin)
            if p == -2:
                raise MemoryError("out of memory")
            elif p == 0:
                ret = self.classes[0] - 1
            else:
                if self.tree.nclasses == 2:
                    if p == -1: ret = self.classes[0]
                    else: ret = self.classes[1]
                else:
                    ret = self.classes[p-1]

        else:
            ret = np.empty(tarr.shape[0], dtype=np.int)
            for i in range(tarr.shape[0]):
                tiarr = tarr[i]
                tdata = <double *> tiarr.data
                p = predict_tree(&self.tree, tdata, &margin)
                free(margin)
                if p == -2:
                    raise MemoryError("out of memory")
                elif p == 0:
                    ret[i] = self.classes[0] - 1
                else:
                    if self.tree.nclasses == 2:
                        if p == -1: ret[i] = self.classes[0]
                        else: ret[i] = self.classes[1]
                    else:
                        ret[i] = self.classes[p-1]
            
        return ret

    def nclasses(self):
        """Returns the number of classes.
        """
        
        if self.tree.x is NULL:
            raise ValueError("no model computed")

        return self.tree.nclasses

    def labels(self):
        """Outputs the name of labels.
        """
        
        if self.tree.x is NULL:
            raise ValueError("no model computed")
        
        ret = np.empty(self.tree.nclasses, dtype=np.int)
        for i in range(self.tree.nclasses):
            ret[i] = self.classes[i]

        return ret

    def _free(self):
        cdef int i
        
        if self.tree.x is not NULL:
            for i in range(self.tree.n):
                free(self.tree.x[i])
            free(self.tree.x)

        if self.tree.y is not NULL:
            free(self.tree.y)
        
        if self.tree.classes is not NULL:
            free(self.tree.classes)
            
        if self.tree.node is not NULL:
            free(self.tree.node[0].npoints_for_class)
            free(self.tree.node[0].priors)

            for i in range(1, self.tree.nnodes):
                free(self.tree.node[i].data)
                free(self.tree.node[i].classes)
                free(self.tree.node[i].npoints_for_class)
                free(self.tree.node[i].priors)
            
            free(self.tree.node)
    
        if self.classes is not NULL:
            free(self.classes)

    def __dealloc__(self):
        self._free()


cdef class MaximumLikelihoodC:
    """Maximum Likelihood Classifier.
    """

    cdef MaximumLikelihood ml
    cdef int *classes

    def __cinit__(self):
        """Initialization.
        """
        
        self.ml.classes = NULL
        self.ml.npoints_for_class = NULL
        self.ml.mean = NULL
        self.ml.covar = NULL
        self.ml.inv_covar = NULL
        self.ml.priors = NULL
        self.ml.det = NULL
        self.classes = NULL
        
    def learn(self, x, y):
        """Learn method.
        
        :Parameters:	
           x : 2d array_like object (N,P)
              training data 
           y : 1d array_like integer 
              class labels
        """

        cdef int ret
        cdef np.ndarray[np.float_t, ndim=2] xarr
        cdef np.ndarray[np.int_t, ndim=1] yarr
        cdef np.ndarray[np.int32_t, ndim=1] ynew
        cdef double *xp
        cdef double **xpp
        cdef int i

        xarr = np.ascontiguousarray(x, dtype=np.float)
        yarr = np.ascontiguousarray(y, dtype=np.int)

        yu = np.unique(yarr)
        if yu.shape[0] <= 1:
            raise ValueError("y: number of classes must be >=2")

        self._free()
        
        # save original labels
        self.classes = <int *> malloc (yu.shape[0] * sizeof(int))
        for i in range(yu.shape[0]):
            self.classes[i] = yu[i]
        
        # transform labels
        ynew = np.empty(yarr.shape[0], dtype=np.int32)
        if yu.shape[0] == 2:
            cond_neg = (yarr == yu[0])
            cond_pos = (yarr == yu[1])
            ynew[cond_neg], ynew[cond_pos] = -1, 1
        else:
            for i in range(yu.shape[0]):
                cond = (yarr == yu[i])
                ynew[cond] = i + 1
    
        xp = <double *> xarr.data
        xpp = <double **> malloc (xarr.shape[0] * sizeof(double*))
        for i in range(xarr.shape[0]):
            xpp[i] = xp + (i * xarr.shape[1])
        
        ret = compute_ml(&self.ml, <int> xarr.shape[0], <int> xarr.shape[1],
                          xpp, <int *> ynew.data)
        free(xpp)

        if ret == 1:
            raise MemoryError("out of memory")
        
    def pred(self, t):
        """Predict Maximum Likelihood model on a test point(s).
        
        :Parameters:
           t : 1d or 2d array_like object ([M,] P)
              test point(s)
              
        :Returns:
	   p : int or 1d numpy array
              the predicted value(s). Retuns the smallest label
              minus one (MaximumLikelihoodC.labels()[0]-1) when 
              the classification is not unique.
        """
        
        cdef int i
        cdef np.ndarray[np.float_t, ndim=1] tiarr
        cdef double *margin
        cdef double *tdata
        

        if self.ml.mean is NULL:
            raise ValueError("no model computed")
        
        tarr = np.ascontiguousarray(t, dtype=np.float)
        if tarr.ndim > 2:
            raise ValueError("t must be an 1d or a 2d array_like object")

        if tarr.shape[-1] != self.ml.d:
            raise ValueError("t, model: shape mismatch")

        if tarr.ndim == 1:
            tiarr = tarr
            p = predict_ml(&self.ml, <double *> tiarr.data, &margin)
            free(margin)
            if p == -2:
                raise MemoryError("out of memory")
            elif p == 0:
                ret = self.classes[0] - 1
            else:
                if self.ml.nclasses == 2:
                    if p == -1: ret = self.classes[0]
                    else: ret = self.classes[1]
                else:
                    ret = self.classes[p-1]

        else:
            ret = np.empty(tarr.shape[0], dtype=np.int)
            for i in range(tarr.shape[0]):
                tiarr = tarr[i]
                tdata = <double *> tiarr.data
                p = predict_ml(&self.ml, tdata, &margin)
                free(margin)
                if p == -2:
                    raise MemoryError("out of memory")
                elif p == 0:
                    ret[i] = self.classes[0] - 1
                else:
                    if self.ml.nclasses == 2:
                        if p == -1: ret[i] = self.classes[0]
                        else: ret[i] = self.classes[1]
                    else:
                        ret[i] = self.classes[p-1]
            
        return ret
            
    def nclasses(self):
        """Returns the number of classes.
        """
        
        if self.ml.mean is NULL:
            raise ValueError("no model computed")

        return self.ml.nclasses

    def labels(self):
        """Outputs the name of labels.
        """
        
        if self.ml.mean is NULL:
            raise ValueError("no model computed")
        
        ret = np.empty(self.ml.nclasses, dtype=np.int)
        for i in range(self.ml.nclasses):
            ret[i] = self.classes[i]

        return ret

    def _free(self):
        cdef int i
        cdef int j
                
        if self.ml.classes is not NULL:
            free(self.ml.classes)
        
        if self.ml.npoints_for_class is not NULL:
            free(self.ml.npoints_for_class)
        
        if self.ml.mean is not NULL:
            for i in range(self.ml.nclasses):
                free(self.ml.mean[i])
            free(self.ml.mean)
            
        if self.ml.covar is not NULL:
            for i in range(self.ml.nclasses):
                for j in range(self.ml.d):
                    free(self.ml.covar[i][j])
                free(self.ml.covar[i])
            free(self.ml.covar)
            
        if self.ml.inv_covar is not NULL:
            for i in range(self.ml.nclasses):
                for j in range(self.ml.d):
                    free(self.ml.inv_covar[i][j])
                free(self.ml.inv_covar[i])
            free(self.ml.inv_covar)

        if self.ml.priors is not NULL:
            free(self.ml.priors)
            
        if self.ml.det is not NULL:
            free(self.ml.det)
            
        if self.classes is not NULL:
            free(self.classes)

    def __dealloc__(self):
        self._free()
