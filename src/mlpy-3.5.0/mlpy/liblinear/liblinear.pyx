## This code is written by Davide Albanese, <albanese@fbk.eu>
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


import numpy as np
cimport numpy as np
from libc.stdlib cimport *

from cliblinear cimport *

    
cdef void print_null(char *s):
    pass

# array 1D to feature node
cdef feature_node *array1d_to_node(np.ndarray[np.float64_t, ndim=1] x):
    cdef int i, k
    cdef np.ndarray[np.int32_t, ndim=1] nz
    cdef feature_node *ret

    nz = np.nonzero(x)[0].astype(np.int32)
    nf = nz.shape[0] + 2 # bias term + last (-1) term
        
    ret = <feature_node*> malloc (nf * sizeof(feature_node))

    k = 0
    for i in nz:
        ret[k].index = i+1
        ret[k].value = x[i]
        k += 1

    ret[nz.shape[0]].index = x.shape[0] + 1 # add bias term
    ret[nz.shape[0]].value = 1 # add bias term
    ret[nz.shape[0]+1].index = -1 # last term
                
    return ret

# array 2D to svm node
cdef feature_node **array2d_to_node(np.ndarray[np.float64_t, ndim=2] x):
    cdef int i
    cdef feature_node **ret

    ret = <feature_node **> malloc \
        (x.shape[0] * sizeof(feature_node *))
    
    for i in range(x.shape[0]):
        ret[i] = array1d_to_node(x[i])
            
    return ret

# array 1D to vector
cdef int *array1d_to_vector(np.ndarray[np.int32_t, ndim=1] y):
     cdef int i
     cdef int *ret

     ret = <int *> malloc (y.shape[0] * sizeof(int))

     for i in range(y.shape[0]):
         ret[i] = y[i]

     return ret


cdef class LibLinear:
    cdef problem problem
    cdef parameter parameter
    cdef model *model
    cdef int learn_disabled
        
    SOLVER_TYPE = ['l2r_lr',
                   'l2r_l2loss_svc_dual',
                   'l2r_l2loss_svc',
                   'l2r_l1loss_svc_dual',
                   'mcsvm_cs',
                   'l1r_l2loss_svc',
                   'l1r_lr',
                   'l2r_lr_dual']

    def __cinit__(self, solver_type='l2r_lr', C=1, eps=0.01, weight={}):
        """LibLinear is a simple class for solving large-scale regularized
        linear classification. It currently supports L2-regularized logistic
        regression/L2-loss support vector classification/L1-loss support vector
        classification, and L1-regularized L2-loss support vector classification/
        logistic regression.
                
        :Parameters:
            solver_type : string
                solver, can be one of 'l2r_lr', 'l2r_l2loss_svc_dual',
                'l2r_l2loss_svc', 'l2r_l1loss_svc_dual', 'mcsvm_cs',
                'l1r_l2loss_svc', 'l1r_lr', 'l2r_lr_dual'
            C : float
                cost of constraints violation
            eps : float
                stopping criterion
            weight : dict 
                changes the penalty for some classes (if the weight for a
                class is not changed, it is set to 1). For example, to
                change penalty for classes 1 and 2 to 0.5 and 0.8
                respectively set weight={1:0.5, 2:0.8}
        """

        set_print_string_function(&print_null)
        self.learn_disabled = 0

        try:
            self.parameter.solver_type = self.SOLVER_TYPE.index(solver_type)
        except ValueError:
            raise ValueError("invalid solver_type")

        self.parameter.C = C
        self.parameter.eps = eps
                
        # weight
        self.parameter.nr_weight = len(weight)
        self.parameter.weight_label = <int *> malloc \
            (len(weight) * sizeof(int))
        self.parameter.weight = <double *> malloc \
            (len(weight) * sizeof(double))
        try:
            for i, key in enumerate(weight):
                self.parameter.weight_label[i] = int(key)
                self.parameter.weight[i] = float(weight[key])
        except ValueError:
            raise ValueError("invalid weight")

        self.model = NULL

    def __dealloc__(self):
        self._free_problem()
        self._free_model()
        self._free_param()
        
    def _load_problem(self, x, y):
        """Convert the data into liblinear problem struct
        """
        
        xarr = np.ascontiguousarray(x, dtype=np.float64)
        yarr = np.ascontiguousarray(y, dtype=np.int32)

        if xarr.ndim != 2:
            raise ValueError("x must be a 2d array_like object")

        if yarr.ndim != 1:
            raise ValueError("y must be an 1d array_like object")

        if xarr.shape[0] != yarr.shape[0]:
            raise ValueError("x, y: shape mismatch")
        
        self.problem.x = array2d_to_node(xarr)
        self.problem.y = array1d_to_vector(yarr)
        self.problem.l = xarr.shape[0]
        self.problem.n = xarr.shape[1] + 1 # + 1 for the bias term
        self.problem.bias = 1
        
    def learn(self, x, y):
        """Learning method.
        
        :Parameters:
      
            x : 2d array_like object
                training data (N, P)
            y : 1d array_like object
                target values (N)
        """

        cdef char *ret

        srand(1)

        if self.learn_disabled:
            raise ValueError("learn method is disabled (model from file)")
        
        self._free_problem()
        self._load_problem(x, y)
        ret = check_parameter(&self.problem, &self.parameter)

        if ret != NULL:
            raise ValueError(ret)

        self._free_model()
        self.model = train(&self.problem, &self.parameter)

    def pred(self, t):
        """Does classification on test vector(s) t.
                
        :Parameters:
            t : 1d (one sample) or 2d array_like object
                test data ([M,] P)
            
        :Returns:
            p : int or 1d numpy array
                the predicted class(es) for t is returned.
        """

        cdef int i
        cdef feature_node *test_node

        tarr = np.ascontiguousarray(t, dtype=np.float64)

        if tarr.ndim > 2:
            raise ValueError("t must be an 1d or a 2d array_like object")
        
        if self.model is NULL:
            raise ValueError("no model computed")

        if tarr.ndim == 1:
            test_node = array1d_to_node(tarr)
            p = predict(self.model, test_node)
            free(test_node)
        else:
            p = np.empty(tarr.shape[0], dtype=np.int)
            for i in range(tarr.shape[0]):
                test_node = array1d_to_node(tarr[i])
                p[i] = predict(self.model, test_node)
                free(test_node)
        
        return p


    def pred_values(self, t):
        """Returns D decision values. D is 1 if there are two 
        classes except multi-class svm by Crammer and Singer 
        ('mcsvm_cs'), and is the number of classes otherwise.
        The pred() method returns the class with the highest 
        decision value.
                
        :Parameters:
            t : 1d (one sample) or 2d array_like object
                test data ([M,] P)
            
        :Returns:
            decision values : 1d (D) or 2d numpy array (M, D)
                decision values for each observation.
        """

        cdef int i, j
        cdef feature_node *test_node
        cdef double *dec_values

        tarr = np.ascontiguousarray(t, dtype=np.float64)

        if tarr.ndim > 2:
            raise ValueError("t must be an 1d or a 2d array_like object")
        
        if self.model is NULL:
            raise ValueError("no model computed")

        if (self.SOLVER_TYPE[self.parameter.solver_type] != \
                'mcsvm_cs') and self.model.nr_class == 2:
            n = 1
        else:
            n = self.model.nr_class

        dec_values = <double *> malloc (n * sizeof(double))

        if tarr.ndim == 1:
            dec_values_arr = np.empty(n, dtype=np.float)
            test_node = array1d_to_node(tarr)
            p = predict_values(self.model, test_node, dec_values)
            free(test_node)
            for j in range(n):
                dec_values_arr[j] = dec_values[j]
        else:
            dec_values_arr = np.empty((tarr.shape[0], n), dtype=np.float)
            for i in range(tarr.shape[0]):
                test_node = array1d_to_node(tarr[i])
                p = predict_values(self.model, test_node, dec_values)
                free(test_node)
                for j in range(n):
                    dec_values_arr[i, j] = dec_values[j]
        
        free(dec_values)
        return dec_values_arr


    def pred_probability(self, t):
        """Returns C (number of classes) probability estimates. 
        The simple probability model of logistic regression
        is used.

        :Parameters:
            t : 1d (one sample) or 2d array_like object
                test data ([M,] P)
            
        :Returns:
            probability estimates : 1d (C) or 2d numpy array (M, C)
                probability estimates for each observation.
        """

        cdef int i, j
        cdef feature_node *test_node
        cdef double *prob_estimates

        tarr = np.ascontiguousarray(t, dtype=np.float64)

        if tarr.ndim > 2:
            raise ValueError("t must be an 1d or a 2d array_like object")
        
        if self.model is NULL:
            raise ValueError("no model computed")

        prob_estimates = <double *> malloc (self.model.nr_class *
                                            sizeof(double))

        if tarr.ndim == 1:
            prob_estimates_arr = np.empty(self.model.nr_class, dtype=np.float)
            test_node = array1d_to_node(tarr)
            p = predict_probability(self.model, test_node, prob_estimates)
            free(test_node)
            for j in range(self.model.nr_class):
                prob_estimates_arr[j] = prob_estimates[j]
        else:
            prob_estimates_arr = np.empty((tarr.shape[0], self.model.nr_class), 
                                      dtype=np.float)
            for i in range(tarr.shape[0]):
                test_node = array1d_to_node(tarr[i])
                p = predict_probability(self.model, test_node, prob_estimates)
                free(test_node)
                for j in range(self.model.nr_class):
                    prob_estimates_arr[i, j] = prob_estimates[j]
        
        free(prob_estimates)
        return prob_estimates_arr


    def nfeature(self):
        """Returns the number of attributes.
        """

        if self.model is NULL:
            raise ValueError("no model")

        return self.model.nr_feature

    def nclasses(self):
        """Returns the number of classes.
        """
        
        if self.model is NULL:
            raise ValueError("no model")

        return self.model.nr_class

    def labels(self):
        """Outputs the name of labels.
        """
        
        if self.model is NULL:
            raise ValueError("no model computed")

        if self.model.label is NULL:
            return None
        else:
            ret = np.empty(self.model.nr_class, dtype=np.int32)
            for i in range(self.model.nr_class):
                ret[i] = self.model.label[i]
            return ret
    
    def _w(self):
               
        cdef int i, n
        cdef np.ndarray[np.float64_t, ndim=1] w

        if self.model is NULL:
            raise ValueError("no model")

        if (self.SOLVER_TYPE[self.parameter.solver_type] == \
                'mcsvm_cs') or self.model.nr_class != 2:
            n = self.problem.n * self.model.nr_class
        else:
            n = self.problem.n
            
        w = np.empty(n, dtype=np.float64)

        for i in range(n):
            w[i] = self.model.w[i]

        if (self.SOLVER_TYPE[self.parameter.solver_type] == \
                'mcsvm_cs') or self.model.nr_class != 2:
            return w.reshape(-1, self.model.nr_class).T
        else:
            return w

    def w(self):
        """Returns the coefficients.
        For 'mcsvm_cs' solver and for multiclass classification 
        returns a 2d numpy array where w[i] contains the
        coefficients of label i. For binary classification 
        an 1d numpy array is returned.
        """

        w = self._w()
        
        if w.ndim == 1:
            return w[:-1]
        else:
            return w[:, :-1]
       
    def bias(self):
        """Returns the bias term(s).
        For 'mcsvm_cs' solver and for multiclass classification
        returns a 1d numpy array where b[i] contains the
        bias of label i (.labels()[i]). For binary classification 
        a float is returned.
        """

        w = self._w()
        
        if w.ndim == 1:
            return w[-1]
        else:
            return w[:, -1]
        
    cdef void _free_problem(self):
        cdef int i
        
        if self.problem.x is not NULL:
            for i in range(self.problem.l):
                free(self.problem.x[i])
            free(self.problem.x)

        if self.problem.y is not NULL:
            free(self.problem.y)
        
    cdef void _free_model(self):
        free_and_destroy_model(&self.model)

    cdef void _free_param(self):
        destroy_param(&self.parameter)
    
    @classmethod
    def load_model(cls, filename):
        """Loads model from file. Returns a LibLinear object
        with the learn() method disabled.
        """
        
        ret = LibLinear()

        try:
            ret.model = load_model(filename)
        except:
            raise ValueError("invalid filename")
        else:
            if ret.model is NULL:
                raise IOError("model file could not be loaded")
            
        ret.parameter.solver_type = ret.model.param.solver_type
        ret.learn_disabled = 1
        
        return ret

    def save_model(self, filename):
        """Saves a model to a file.        
        """
        
        cdef int ret
        
        if self.model is NULL:
            raise ValueError("no model computed")
        
        ret = save_model(filename, self.model)
        
        if ret == -1:
            raise IOError("problem with save_model()")
