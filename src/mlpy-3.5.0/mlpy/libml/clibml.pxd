cdef extern from "src/ml.h":
    int TRUE
    int FALSE
    int SORT_ASCENDING
    int SORT_DESCENDING
    int DIST_SQUARED_EUCLIDEAN
    int DIST_EUCLIDEAN
    int SVM_KERNEL_LINEAR
    int SVM_KERNEL_GAUSSIAN
    int SVM_KERNEL_POLINOMIAL
    int BAGGING
    int AGGREGATE
    int ADABOOST

    ctypedef struct NearestNeighbor:
        int n
        int d
        double **x
        int *y
        int nclasses
        int *classes
        int k
        int dist
        
    int compute_nn(NearestNeighbor *nn, int n, int d, double *x[], int y[],
                   int k, int dist)
    int predict_nn(NearestNeighbor *nn, double x[], double **margin)
    
    ctypedef struct Node:
        double **data
        int *classes
        int npoints
        int nvar
        int nclasses
        int *npoints_for_class
        double *priors
        int node_class
        int terminal
        int left
        int right
        int var
        double value
        
    ctypedef struct Tree:
        int n
        int d
        double **x
        int *y
        int nclasses
        int *classes
        Node *node
        int nnodes
        int stumps
        int minsize

    int compute_tree(Tree *tree, int n, int d, double *x[], int y[],
                   int stumps, int minsize)
    int predict_tree(Tree *tree, double x[], double **margin)

    ctypedef struct MaximumLikelihood:
        int nclasses
        int *classes
        int *npoints_for_class
        int d
        double **mean
        double ***covar        
        double ***inv_covar
        double *priors
        double *det

    int compute_ml(MaximumLikelihood *ml, int n, int d, double *x[], int y[])
    int predict_ml(MaximumLikelihood *ml, double x[], double **margin)
