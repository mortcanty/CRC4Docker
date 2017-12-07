#ifndef ML_H
#define ML_H

#include <stdlib.h>
#include <stdio.h>


#define TRUE 1
#define FALSE 0

#define SORT_ASCENDING 1
#define SORT_DESCENDING 2

#define DIST_SQUARED_EUCLIDEAN 1
#define DIST_EUCLIDEAN 2


#define SVM_KERNEL_LINEAR 1
#define SVM_KERNEL_GAUSSIAN 2
#define SVM_KERNEL_POLINOMIAL 3

#define BAGGING 1
#define AGGREGATE 2
#define ADABOOST 3

#define drand48() (((float) rand())/((float) RAND_MAX))
#define srand48(x) (srand((x)))

/*NN*/

typedef struct NearestNeighbor
{
  int n; /*number of examples*/
  int d; /*number of variables*/
  double **x; /*the data*/
  int *y; /*their classes*/
  int nclasses; /*the number of classes*/
  int *classes; /*the classes*/
  int k;  /*number of nn (for the test phase)*/
  int dist;  /*type of distance (for the test phase)*/
} NearestNeighbor;

typedef struct ENearestNeighbor
{
  NearestNeighbor *nn; /*the nn's*/
  int nmodels; /*number of nn models*/
  double *weights; /*models weights*/
  int nclasses; /*the number of classes*/
  int *classes; /*the classes*/
  int k;/*number of nn (for the test phase)*/
  int dist;  /*type of distance (for the test phase)*/
} ENearestNeighbor;

/*TREE*/

typedef struct Node
{
  double **data; /*data contained into the node*/
  int *classes; /*classes of the data*/
  int npoints; /*number of data into the data*/
  int nvar; /*dimension of the data*/
  int nclasses; /*number of classes*/
  int *npoints_for_class; /*number of elements for each class*/
  double *priors; /*prior probabilities for each class*/
  int node_class; /*class associated to the current node*/
  int terminal; /*is the node terminal? TRUE or FALSE*/
  int left; /*his left child*/
  int right; /*his right child*/
  int var; /*variable used to split this node (if not terminal)*/
  double value; /*value of the variable for splitting the data*/
} Node;


typedef struct Tree
{
  int n; /*number of examples*/
  int d; /*number of variables*/
  double **x; /*the data*/
  int *y; /*their classes*/
  int nclasses; /*the number of classes*/
  int *classes; /*the classes*/
  Node *node; /*the nodes*/
  int nnodes; /*number of nodes*/
  int stumps; /*if it is stamps: TRUE or FALSE*/
  int minsize; /*minsize for splitting a node*/
} Tree;

typedef struct ETree
{
  Tree *tree; /*the trees*/
  int nmodels; /*number of trees*/
  double *weights; /*models weights*/
  int nclasses; /*the number of classes*/
  int *classes; /*the classes*/
} ETree;

/*SVM*/

typedef struct SupportVectorMachine
{
  int n;/*number of examples*/
  int d;/*number of features*/
  double **x;/*training data*/
  int *y;/*class labels*/
  double C;/*bias/variance parameter*/
  double tolerance;/*tolerance for testing KKT conditions*/
  double eps;/*convergence parameters:used in both takeStep and mqc functions*/
  int kernel_type;/*kernel type:1 linear, 2 gaussian, 3 polynomial*/
  double two_sigma_squared;/*kernel parameter*/
  double *alph;/*lagrangian coefficients*/
  double b;/*offset*/
  double *w;/*hyperplane parameters (linearly separable  case)*/
  double *error_cache;/*error for each training point*/ 
  int end_support_i;/*set to N, never changed*/
  double (*learned_func)(int, struct SupportVectorMachine *);/*the SVM*/
  double (*kernel_func)(int, int, struct SupportVectorMachine *);/*the kernel*/
  double delta_b;/*gap between old and updated offset*/
  double *precomputed_self_dot_product;/*squared norm of the training data*/
  double *Cw;/*weighted C parameter (sen/spe)*/
  int non_bound_support;/*number of non bound SV*/
  int bound_support;/*number of bound SV*/
  int maxloops; /*maximum number of optimization loops*/
  int convergence; /*to assess convergence*/
  int verbose; /*verbosity */
  double **K; /*precomputed kernel matrix (for RSFN)*/
} SupportVectorMachine;

typedef struct ESupportVectorMachine
{
  SupportVectorMachine *svm; /*the svm's*/
  int nmodels; /*number of svm's*/
  double *weights; /*modeles weights*/
} ESupportVectorMachine;

/*ML*/

typedef struct MaximumLikelihood
{
  int nclasses; /*number of classes*/
  int *classes; /*array of the class names*/
  int *npoints_for_class; /*number of examples contained in each class*/
  int d; /*number of predictor variables*/
  double **mean; /*for each class, the mean value of the examples stored
		   in an array of length nvars*/
  double ***covar; /*for each class, the covariance matrix of the esamples
		     stored in a matrix of dimension nvars x nvars*/
  double ***inv_covar; /*for each class, the inverse of the covariance matrix
			 stored in a matrix of dimension nvars x nvars*/
  double *priors; /* prior probabilities of each class*/
  double *det; /*for each class, the determinant of the inverse of the
		 covariance matrix*/
} MaximumLikelihood;

typedef struct EMaximumLikelihood
{
  MaximumLikelihood *ml; /*the ml's*/
  int nmodels; /*number of ml's*/
  double *weights; /*models weights*/
  int nclasses; /*the number of classes*/
  int *classes; /*the classes*/
} EMaximumLikelihood;


/*RSFN*/

typedef struct SlopeFunctions
{
  double *w;
  double *b;
  int *i;
  int *j;
  int nsf;
} SlopeFunctions;

typedef struct RegularizedSlopeFunctionNetworks
{
  double **x;
  int d;
  SupportVectorMachine svm;
  SlopeFunctions sf;
  double threshold;
} RegularizedSlopeFunctionNetworks;

typedef struct ERegularizedSlopeFunctionNetworks
{
  RegularizedSlopeFunctionNetworks *rsfn; 
  int nmodels; 
  double *weights;
} ERegularizedSlopeFunctionNetworks;

typedef struct RegularizationNetwork
{
  int n; /*number of examples*/
  int d; /*number of variables*/
  double **x; /*the data*/
  double *y; /*their values*/
  double lambda;/*regular. param.*/
  double sigma;/*rbf kernel param.*/
  double *c; /*linear expansion coeffs*/
} RegularizationNetwork;


typedef struct TerminatedRamps
{
  double **w;
  double *alpha;
  double *b;
  int *i;
  int *j;
  double *y_min;
  double *y_max;
  int ntr;
} TerminatedRamps;

typedef struct TerminatedRampsRegularizationNetwork
{
  int n; /*number of examples*/
  int d; /*number of variables*/
  double **x; /*the data*/
  double *y; /*their values*/
  double lambda;/*regular. param.*/
  double *c; /*linear expansion coeffs*/
  TerminatedRamps tr; /*the ramp functions*/
} TerminatedRampsRegularizationNetwork;

/***************
FUNCTIONS
***************/

/*memory*/
int *ivector(long n);
double *dvector(long n);
double **dmatrix(long n, long m);
int **imatrix(long n, long m);
int free_ivector(int *v);
int free_dvector(double *v);
int free_dmatrix(double **M, long n, long m);
int free_imatrix(int **M, long n, long m);

/*read data*/
int get_line(char **line,FILE *fp);

int read_classification_data(char file[],char sep,double ***x,int **y,int *r,
			     int *c);
int read_regression_data(char file[],char sep,double ***x,double **y,int *r,
			 int *c);

/*parser*/
int parser(int n,char *array[],char ***flag,char ***value,int *nflags);
char *get_value(char *flag[],char *value[],int nflags,char opt[]);

/*sorting*/
void dsort(double a[], int ib[],int n, int action);
void isort(int a[], int ib[],int n, int action);

/*random sampling*/
int sample(int n, double prob[], int nsamples, int **samples, int replace,
	   int seed);

/*unique*/
int iunique(int y[], int n, int **values);
int dunique(double y[], int n, double **values);

/*distance*/
double l1_distance(double x[],double y[],int n);
double euclidean_squared_distance(double x[],double y[],int n);
double euclidean_distance(double x[],double y[],int n);
double scalar_product(double x[],double y[],int n);
double euclidean_norm(double x[],int n);

/*inverse matrix and determinant*/
int inverse(double *A[],double *inv_A[],int n);
double determinant(double *A[],int n);

/*nn*/
int compute_nn(NearestNeighbor *nn,int n,int d,double *x[],int y[],
	       int k, int dist);

int compute_enn(ENearestNeighbor *enn,int n,int d,double *x[],int y[],
		int method,int nmodels,int k, int dist);

int predict_nn(NearestNeighbor *nn, double x[],double **margin);

int predict_enn(ENearestNeighbor *enn, double x[],double **margin);

/*tree*/
int compute_tree(Tree *tree,int n,int d,double *x[],
		 int y[],int stumps,int minsize);

int compute_etree(ETree *etree,int n,int d,double *x[], int y[], int method, 
		  int nmodels,int stumps, int minsize);


int predict_tree(Tree *tree, double x[],double **margin);

int predict_etree(ETree *enn, double x[],double **margin);

/*svm*/
int compute_svm(SupportVectorMachine *svm,int n,int d,double *x[],int y[],
		 int kernel,double kp,double C,double tol,
		 double eps,int maxloops,int verbose, double W[]);

int compute_esvm(ESupportVectorMachine *esvm,int n,int d,double *x[],
		 int y[], int method,int nmodels,int kernel,double kp,
		 double C,double tol,double eps,int maxloops,int verbose);

int predict_svm(SupportVectorMachine *svm,double x[],double **margin);

int predict_esvm(ESupportVectorMachine *esvm,double x[],double **margin);

/*ml*/
int compute_ml(MaximumLikelihood *ml,int n,int d,double *x[],int y[]);

int compute_eml(EMaximumLikelihood *eml,int n,int d,double *x[],
		int y[], int method,int nmodels);

int predict_ml(MaximumLikelihood *ml, double x[],double **margin);

int predict_eml(EMaximumLikelihood *enn, double x[],double **margin);

/*rsfn*/
int compute_rsfn(RegularizedSlopeFunctionNetworks *rsfn,int n,int d,
		 double *x[],int y[],double C,double tol,
		 double eps,int maxloops,int verbose,double W[],
		 double threshold,int knn);

int compute_ersfn(ERegularizedSlopeFunctionNetworks *ersfn,int n,
		  int d,double *x[],
		 int y[],int method,int nmodels, double C,double tol,
		  double eps,int maxloops,int verbose,double threshold,
		  int knn);

int predict_rsfn(RegularizedSlopeFunctionNetworks *rsfn,double x[],
		 double **margin);

int predict_ersfn(ERegularizedSlopeFunctionNetworks *ersfn, double x[],
		  double **margin);

/*rn*/
int compute_rn(RegularizationNetwork *rn,int n,int d,
	       double *x[],double y[],double lambda,double sigma);

double trrbf_kernel(double x1[], double x2[],int d,double sigma);

/*trrn*/
int compute_trrn(TerminatedRampsRegularizationNetwork *trrn,int n,int d,
                 double *x[],double y[],double lambda,int k_nn);

double tr_kernel(double x1[], double x2[],
		 TerminatedRampsRegularizationNetwork *trrn);

/*ttest*/
int ttest(double *data1,int n1,double *data2,int n2,double *t,double *prob);

#endif /* ML_H */
