#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "ml.h"

static void svm_smo(SupportVectorMachine *svm);
static int examineExample(int i1, SupportVectorMachine *svm);
static int takeStep(int i1, int i2, SupportVectorMachine *svm);

static double learned_func_linear(int k, SupportVectorMachine *svm);
static double learned_func_nonlinear(int k, SupportVectorMachine *svm);

static double rbf_kernel(int i1, int i2, SupportVectorMachine *svm);
static double polinomial_kernel(int i1, int i2, SupportVectorMachine *svm);
static double dot_product_func(int i1, int i2, SupportVectorMachine *svm);

static int compute_svm_bagging(ESupportVectorMachine *esvm,int n,int d,
			       double *x[],int y[],int nmodels,int kernel,
			       double kp,double C,double tol,double eps,
			       int maxloops,int verbose);
static int compute_svm_aggregate(ESupportVectorMachine *esvm,int n,int d,
			       double *x[],int y[],int nmodels,int kernel,
			       double kp,double C,double tol,double eps,
			       int maxloops,int verbose);
static int compute_svm_adaboost(ESupportVectorMachine *esvm,int n,int d,
			       double *x[],int y[],int nmodels,int kernel,
			       double kp,double C,double tol,double eps,
			       int maxloops,int verbose);


int compute_svm(SupportVectorMachine *svm,int n,int d,double *x[],int y[],
		int kernel,double kp,double C,double tol,
		double eps,int maxloops,int verbose,double W[])
     /*
       compute svm model.x,y,n,d are the input data.
       kernel is the kernel type (see ml.h), kp is the kernel parameter 
       (for gaussian and polynomial kernel), C is the regularization parameter.
       eps and tol determine convergence, maxloops is thae maximum number
       of optimization loops, W is an array (of length n) of weights for 
       cost-sensitive  classification.

       Return value: 0 on success, 1 otherwise.
     */
{
  int i,j;
  int nclasses;
  int *classes;

  svm->n=n;
  svm->d=d;
  svm->C=C;
  svm->tolerance=tol;
  svm->eps=eps;
  svm->two_sigma_squared=kp;
  svm->kernel_type=kernel;
  svm->maxloops=maxloops;
  svm->verbose=verbose;

  svm->b=0.0;

  if(C<=0){
    fprintf(stderr,"compute_svm: regularization parameter C must be > 0\n");
    return 1;
  }
  if(eps<=0){
    fprintf(stderr,"compute_svm: parameter eps must be > 0\n");
    return 1;
  }
  if(tol<=0){
    fprintf(stderr,"compute_svm: parameter tol must be > 0\n");
    return 1;
  }
  if(maxloops<=0){
    fprintf(stderr,"compute_svm: parameter maxloops must be > 0\n");
    return 1;
  }
  if(W){
    for(i=0;i<n;i++)
      if(W[i]<=0){
	fprintf(stderr,"compute_svm: parameter W[%d] must be > 0\n",i);
	return 1;
      }
  }

  switch(kernel){
  case SVM_KERNEL_LINEAR:
    break;
  case SVM_KERNEL_GAUSSIAN:
    if(kp <=0){
      fprintf(stderr,"compute_svm: parameter kp must be > 0\n");
      return 1;
    }
    break;
  case SVM_KERNEL_POLINOMIAL:
    if(kp <=0){
      fprintf(stderr,"compute_svm: parameter kp must be > 0\n");
      return 1;
    }
    break;
  default:
    fprintf(stderr,"compute_svm: kernel not recognized\n");
    return 1;
  }

  nclasses=iunique(y,n, &classes);

  if(nclasses<=0){
    fprintf(stderr,"compute_svm: iunique error\n");
    return 1;
  }
  if(nclasses==1){
    fprintf(stderr,"compute_svm: only 1 class recognized\n");
    return 1;
  }
  if(nclasses==2)
    if(classes[0] != -1 || classes[1] != 1){
      fprintf(stderr,"compute_svm: for binary classification classes must be -1,1\n");
      return 1;
    }
  if(nclasses>2){
    fprintf(stderr,"compute_svm: multiclass classification not allowed\n");
    return 1;
  }

  if(kernel==SVM_KERNEL_LINEAR)
    if(!(svm->w=dvector(d))){
      fprintf(stderr,"compute_svm: out of memory\n");
      return 1;
    }
  if(!(svm->Cw=dvector(n))){
    fprintf(stderr,"compute_svm: out of memory\n");
    return 1;
  }
  if(!(svm->alph=dvector(n))){
    fprintf(stderr,"compute_svm: out of memory\n");
    return 1;
  }
  if(!(svm->error_cache=dvector(n))){
    fprintf(stderr,"compute_svm: out of memory\n");
    return 1;
  }
  if(!(svm->precomputed_self_dot_product=dvector(n))){
    fprintf(stderr,"compute_svm: out of memory\n");
    return 1;
  }
  
  for(i=0;i<n;i++)
    svm->error_cache[i]=-y[i];

  if(W){
    for(i=0;i<n;i++)
      svm->Cw[i]=svm->C * W[i];
  }else{
    for(i=0;i<n;i++)
      svm->Cw[i]=svm->C;
  }    
  

  if(!(svm->x=dmatrix(n,d))){
    fprintf(stderr,"compute_svm: out of memory\n");
    return 1;
  }
  if(!(svm->y=ivector(n))){
    fprintf(stderr,"compute_svm: out of memory\n");
    return 1;
  }

  for(i=0;i<n;i++){
    for(j=0;j<d;j++)
      svm->x[i][j]=x[i][j];
    svm->y[i]=y[i];
  }

  svm_smo(svm);
  
  svm->non_bound_support=svm->bound_support=0;
  for(i=0;i<n;i++){
    if(svm->alph[i]>0){
      if(svm->alph[i]< svm->Cw[i])
	svm->non_bound_support++;
      else
	svm->bound_support++;
    }
  }
  
  free_ivector(classes);

  return 0;
}

int compute_esvm(ESupportVectorMachine *esvm,int n,int d,double *x[],
		 int y[],int method,int nmodels,int kernel,double kp,
		 double C,double tol,double eps,int maxloops,int verbose)
    /*
       compute ensamble of svm models.x,y,n,d are the input data.
       method is one of BAGGING,AGGREGATE,ADABOOST.
       kernel is the kernel type (see ml.h), kp is the kernel parameter 
       (for gaussian and polynomial kernel), C is the regularization parameter.
       eps and tol determine convergence, maxloops is thae maximum number
       of optimization loops.

       Return value: 0 on success, 1 otherwise.
     */
{
  switch(method){
  case BAGGING:
    if(compute_svm_bagging(esvm,n,d,x,y,nmodels,kernel,kp,C,tol,eps,
			   maxloops,verbose) != 0){
      fprintf(stderr,"compute_esvm: compute_svm_bagging error\n");
      return 1;
    }
    break;
  case AGGREGATE:
    if(compute_svm_aggregate(esvm,n,d,x,y,nmodels,kernel,kp,C,tol,eps,
			maxloops,verbose) != 0){
      fprintf(stderr,"compute_esvm: compute_svm_aggregate error\n");
      return 1;
    }
    break;
  case ADABOOST:
    if( compute_svm_adaboost(esvm,n,d,x,y,nmodels,kernel,kp,C,tol,eps,
			     maxloops,verbose) != 0){
      fprintf(stderr,"compute_esvm: compute_svm_adaboost error\n");
      return 1;
    }
    break;
  default:
    fprintf(stderr,"compute_esvm: ensamble method not recognized\n");
    return 1;
    break;
  }

  return 0;
}
     
int predict_svm(SupportVectorMachine *svm,double x[],double **margin)
     /*
       predicts svm model on a test point x. the array margin (of length
       nclasses) shall contain the margin of the classes.

       Return value: the predicted value on success (-1 or 1), 
       0 on succes with non unique classification.
     */

{
  int i,j;
  double y = 0.0;
  double K;

  if(svm->kernel_type==SVM_KERNEL_GAUSSIAN){
    for(i = 0; i < svm->n; i++){
      if(svm->alph[i] > 0){
	K=0.0;
	for(j=0;j<svm->d;j++)
	  K+=(svm->x[i][j]-x[j])*(svm->x[i][j]-x[j]);
	y += svm->alph[i] * svm->y[i] * exp(-K/svm->two_sigma_squared);
      }
    }
    y -= svm->b;
  }

  if(svm->kernel_type==SVM_KERNEL_LINEAR){
    K=0.0;
    for(j=0;j<svm->d;j++)
      K+=svm->w[j]*x[j];
    y=K-svm->b;
  }

  if(svm->kernel_type==SVM_KERNEL_POLINOMIAL){
    for(i = 0; i < svm->n; i++){
      if(svm->alph[i] > 0){
	K=1.0;
	for(j=0;j<svm->d;j++)
	  K+=svm->x[i][j]*x[j];
	
	y += svm->alph[i] * svm->y[i] * pow(K,svm->two_sigma_squared);
      }
    }
    y -= svm->b;
  }

  (*margin)=dvector(2);
  if(y>0){
    (*margin)[1]=y;
    return 1;
  }else if(y<0){
    (*margin)[0]=-y;
    return -1;
  }else
    return 0;
    
}


int predict_esvm(ESupportVectorMachine *esvm, double x[],double **margin)
     /*
       predicts svm model on a test point x. Proportions of neighbours
       for each class will be stored within the array margin 
       (an array of length svm->nclasses). 

       
       Return value: the predicted value on success (-1 or 1 for
       binary classification; 1,...,nclasses in the multiclass case),
       0 on succes with non unique classification, -2 otherwise.
     */
{
  int b;
  int pred;
  double *tmpmargin;
  
  if(!((*margin)=dvector(2))){
    fprintf(stderr,"predict_esvm: out of memory\n");
    return -2;
  }
  
  for(b=0;b<esvm->nmodels;b++){
    pred=predict_svm(&(esvm->svm[b]), x,&tmpmargin);
    if(pred < -1){
      fprintf(stderr,"predict_esvm: predict_svm error\n");
      return -2;
    }
    if(pred==-1)
      (*margin)[0] += esvm->weights[b];
    else if(pred==1)
      (*margin)[1] += esvm->weights[b];
    
    free_dvector(tmpmargin);
  }
  
  if((*margin)[0] > (*margin)[1])
    return -1;
  else if((*margin)[0] < (*margin)[1])
    return 1;
  else
    return 0;

  return -2;
  
}



    
static int compute_svm_bagging(ESupportVectorMachine *esvm,int n,int d,
			       double *x[],int y[],int nmodels,int kernel,
			       double kp,double C,double tol,double eps,
			       int maxloops,int verbose)
{
  int i,b;
  int *samples;
  double **trx;
  int *try;
  int nclasses;
  int *classes;

  if(nmodels<1){
    fprintf(stderr,"compute_svm_bagging: nmodels must be greater than 0\n");
    return 1;
  }

  if(C<=0){
    fprintf(stderr,"compute_svm_bagging: regularization parameter C must be > 0\n");
    return 1;
  }
  if(eps<=0){
    fprintf(stderr,"compute_svm_bagging: parameter eps must be > 0\n");
    return 1;
  }
  if(tol<=0){
    fprintf(stderr,"compute_svm_bagging: parameter tol must be > 0\n");
    return 1;
  }
  if(maxloops<=0){
    fprintf(stderr,"compute_svm_bagging: parameter maxloops must be > 0\n");
    return 1;
  }

  switch(kernel){
  case SVM_KERNEL_LINEAR:
    break;
  case SVM_KERNEL_GAUSSIAN:
    if(kp <=0){
      fprintf(stderr,"compute_svm_bagging: parameter kp must be > 0\n");
      return 1;
    }
    break;
  case SVM_KERNEL_POLINOMIAL:
    if(kp <=0){
      fprintf(stderr,"compute_svm_bagging: parameter kp must be > 0\n");
      return 1;
    }
    break;
  default:
    fprintf(stderr,"compute_svm_bagging: kernel not recognized\n");
    return 1;
  }

  nclasses=iunique(y,n, &classes);

  if(nclasses<=0){
    fprintf(stderr,"compute_svm_bagging: iunique error\n");
    return 1;
  }
  if(nclasses==1){
    fprintf(stderr,"compute_svm_bagging: only 1 class recognized\n");
    return 1;
  }
  if(nclasses==2)
    if(classes[0] != -1 || classes[1] != 1){
      fprintf(stderr,"compute_svm_bagging: for binary classification classes must be -1,1\n");
      return 1;
    }
  if(nclasses>2){
    fprintf(stderr,"compute_svm_bagging: multiclass classification not allowed\n");
    return 1;
  }


  if(!(esvm->svm=(SupportVectorMachine *)
       calloc(nmodels,sizeof(SupportVectorMachine)))){
    fprintf(stderr,"compute_svm_bagging: out of memory\n");
    return 1;
  }
  esvm->nmodels=nmodels;
  if(!(esvm->weights=dvector(nmodels))){
    fprintf(stderr,"compute_svm_bagging: out of memory\n");
    return 1;
  }

  for(b=0;b<nmodels;b++)
    esvm->weights[b]=1.0 / (double) nmodels;
  
  if(!(trx=(double **)calloc(n,sizeof(double*)))){
    fprintf(stderr,"compute_svm_bagging: out of memory\n");
    return 1;
  }
  if(!(try=ivector(n))){
    fprintf(stderr,"compute_svm_bagging: out of memory\n");
    return 1;
  }
  
  for(b=0;b<nmodels;b++){
    if(sample(n, NULL, n, &samples, TRUE,b)!=0){
       fprintf(stderr,"compute_svm_bagging: sample error\n");
       return 1;
    }

    for(i =0;i<n;i++){
      trx[i] = x[samples[i]];
      try[i] = y[samples[i]];
    }

    if(compute_svm(&(esvm->svm[b]),n,d,trx,try,kernel,kp,C,
		   tol,eps,maxloops,verbose,NULL)!=0){
      fprintf(stderr,"compute_svm_bagging: compute_svm error\n");
      return 1;
    }
    free_ivector(samples);

  }

  free(trx);
  free_ivector(classes);
  free_ivector(try);
    
  return 0;

}



static int compute_svm_aggregate(ESupportVectorMachine *esvm,int n,int d,
				 double *x[],int y[],int nmodels,int kernel,
				 double kp,double C,double tol,double eps,
				 int maxloops,int verbose)
{
  int i,b;
  int *samples;
  double **trx;
  int *try;
  int indx;
  int nclasses;
  int *classes;

  if(nmodels<1){
    fprintf(stderr,"compute_svm_aggregate: nmodels must be greater than 0\n");
    return 1;
  }

  if(nmodels > n){
    fprintf(stderr,"compute_svm_aggregate: nmodels must be less than n\n");
    return 1;
  }

  if(C<=0){
    fprintf(stderr,"compute_svm_aggregate: regularization parameter C must be > 0\n");
    return 1;
  }
  if(eps<=0){
    fprintf(stderr,"compute_svm_aggregate: parameter eps must be > 0\n");
    return 1;
  }
  if(tol<=0){
    fprintf(stderr,"compute_svm_aggregate: parameter tol must be > 0\n");
    return 1;
  }
  if(maxloops<=0){
    fprintf(stderr,"compute_svm_aggregate: parameter maxloops must be > 0\n");
    return 1;
  }

  switch(kernel){
  case SVM_KERNEL_LINEAR:
    break;
  case SVM_KERNEL_GAUSSIAN:
    if(kp <=0){
      fprintf(stderr,"compute_svm_aggregate: parameter kp must be > 0\n");
      return 1;
    }
    break;
  case SVM_KERNEL_POLINOMIAL:
    if(kp <=0){
      fprintf(stderr,"compute_svm_aggregate: parameter kp must be > 0\n");
      return 1;
    }
    break;
  default:
    fprintf(stderr,"compute_svm_aggregate: kernel not recognized\n");
    return 1;
  }

  nclasses=iunique(y,n, &classes);

  if(nclasses<=0){
    fprintf(stderr,"compute_svm_aggregate: iunique error\n");
    return 1;
  }
  if(nclasses==1){
    fprintf(stderr,"compute_svm_aggregate: only 1 class recognized\n");
    return 1;
  }
  if(nclasses==2)
    if(classes[0] != -1 || classes[1] != 1){
      fprintf(stderr,"compute_svm_aggregate: for binary classification classes must be -1,1\n");
      return 1;
    }
  if(nclasses>2){
    fprintf(stderr,"compute_svm_aggregate: multiclass classification not allowed\n");
    return 1;
  }

  if(!(esvm->svm=(SupportVectorMachine *)
       calloc(nmodels,sizeof(SupportVectorMachine)))){
    fprintf(stderr,"compute_svm_aggregate: out of memory\n");
    return 1;
  }
  esvm->nmodels=nmodels;
  if(!(esvm->weights=dvector(nmodels))){
    fprintf(stderr,"compute_svm_aggregate: out of memory\n");
    return 1;
  }

  for(b=0;b<nmodels;b++)
    esvm->weights[b]=1.0 / (double) nmodels;
  
  if(!(trx=(double **)calloc(n,sizeof(double*)))){
    fprintf(stderr,"compute_svm_aggregate: out of memory\n");
    return 1;
  }
  if(!(try=ivector(n))){
    fprintf(stderr,"compute_svm_aggregate: out of memory\n");
    return 1;
  }
  
  if(sample(nmodels, NULL, n, &samples, TRUE,0)!=0){
    fprintf(stderr,"compute_svm_aggregate: sample error\n");
    return 1;
  }

  for(b=0;b<nmodels;b++){
  
    indx=0;
    for(i=0;i<n;i++)
      if(samples[i] == b){
	trx[indx] = x[i];
	try[indx++] = y[i];
      }

    if(compute_svm(&(esvm->svm[b]),indx,d,trx,try,kernel,kp,C,
		   tol,eps,maxloops,verbose,NULL)!=0){
      fprintf(stderr,"compute_svm_aggregate: compute_svm error\n");
      return 1;
    }

  }

  free_ivector(samples);
  free(trx);
  free_ivector(classes);
  free_ivector(try);
    
  return 0;

}

static int compute_svm_adaboost(ESupportVectorMachine *esvm,int n,int d,
				double *x[],int y[],int nmodels,int kernel,
				double kp,double C,double tol,double eps,
				int maxloops,int verbose)
{
  int i,b;
  int *samples;
  double **trx;
  int *try;
  double *prob;
  double *prob_copy;
  double sumalpha;
  double epsilon;
  int *pred;
  double *margin;
  double sumprob;
  int nclasses;
  int *classes; 

  if(nmodels<1){
    fprintf(stderr,"compute_svm_adaboost: nmodels must be greater than 0\n");
    return 1;
  }

 if(C<=0){
    fprintf(stderr,"compute_svm_adaboost: regularization parameter C must be > 0\n");
    return 1;
  }
  if(eps<=0){
    fprintf(stderr,"compute_svm_adaboost: parameter eps must be > 0\n");
    return 1;
  }
  if(tol<=0){
    fprintf(stderr,"compute_svm_adaboost: parameter tol must be > 0\n");
    return 1;
  }
  if(maxloops<=0){
    fprintf(stderr,"compute_svm_adaboost: parameter maxloops must be > 0\n");
    return 1;
  }

  switch(kernel){
  case SVM_KERNEL_LINEAR:
    break;
  case SVM_KERNEL_GAUSSIAN:
    if(kp <=0){
      fprintf(stderr,"compute_svm_adaboost: parameter kp must be > 0\n");
      return 1;
    }
    break;
  case SVM_KERNEL_POLINOMIAL:
    if(kp <=0){
      fprintf(stderr,"compute_svm_adaboost: parameter kp must be > 0\n");
      return 1;
    }
    break;
  default:
    fprintf(stderr,"compute_svm_adaboost: kernel not recognized\n");
    return 1;
  }

  nclasses=iunique(y,n, &classes);

  if(nclasses<=0){
    fprintf(stderr,"compute_svm_adaboost: iunique error\n");
    return 1;
  }
  if(nclasses==1){
    fprintf(stderr,"compute_svm_adaboost: only 1 class recognized\n");
    return 1;
  }
  if(nclasses==2)
    if(classes[0] != -1 || classes[1] != 1){
      fprintf(stderr,"compute_svm_adaboost: for binary classification classes must be -1,1\n");
      return 1;
    }
  if(nclasses>2){
    fprintf(stderr,"compute_svm_adaboost: multiclass classification not allowed\n");
    return 1;
  }

  if(!(esvm->svm=(SupportVectorMachine *)
       calloc(nmodels,sizeof(SupportVectorMachine)))){
    fprintf(stderr,"compute_svm_adaboost: out of memory\n");
    return 1;
  }

  if(!(esvm->weights=dvector(nmodels))){
    fprintf(stderr,"compute_svm_adaboost: out of memory\n");
    return 1;
  }

  if(!(trx=(double **)calloc(n,sizeof(double*)))){
    fprintf(stderr,"compute_svm_adaboost: out of memory\n");
    return 1;
  }
  if(!(try=ivector(n))){
    fprintf(stderr,"compute_svm_adaboost: out of memory\n");
    return 1;
  }
  
  if(!(prob_copy=dvector(n))){
    fprintf(stderr,"compute_svm_adaboost: out of memory\n");
    return 1;
  }
  if(!(prob=dvector(n))){
    fprintf(stderr,"compute_svm_adaboost: out of memory\n");
    return 1;
  }

  if(!(pred=ivector(n))){
    fprintf(stderr,"compute_svm_adaboost: out of memory\n");
    return 1;
  }

  for(i =0;i<n;i++)
    prob[i]=1.0/(double)n;

  esvm->nmodels=nmodels;
  sumalpha=0.0;
  for(b=0;b<nmodels;b++){

    for(i =0;i<n;i++)
      prob_copy[i]=prob[i];
    if(sample(n, prob_copy, n, &samples, TRUE,b)!=0){
      fprintf(stderr,"compute_svm_adaboost: sample error\n");
      return 1;
    }

    for(i =0;i<n;i++){
      trx[i] = x[samples[i]];
      try[i] = y[samples[i]];
    }
    
    if(compute_svm(&(esvm->svm[b]),n,d,trx,try,kernel,kp,C,
		   tol,eps,maxloops,verbose,NULL)!=0){
      fprintf(stderr,"compute_svm_adaboost: compute_svm error\n");
      return 1;
    }
    free_ivector(samples);

    epsilon=0.0;
    for(i=0;i<n;i++){
      pred[i]=predict_svm(&(esvm->svm[b]),x[i],&margin);
      if(pred[i] < -1 ){
	fprintf(stderr,"compute_svm_adaboost: predict_svm error\n");
	return 1;
      }
      if(pred[i]==0 || pred[i] != y[i])
	epsilon += prob[i];
      free_dvector(margin);
    }
    
    if(epsilon > 0 && epsilon < 0.5){
      esvm->weights[b]=0.5 *log((1.0-epsilon)/epsilon);
      sumalpha+=esvm->weights[b];
    }else{
      esvm->nmodels=b;
      break;
    }
      
    sumprob=0.0;
    for(i=0;i<n;i++){
      prob[i]=prob[i]*exp(-esvm->weights[b]*y[i]*pred[i]);
      sumprob+=prob[i];
    }

    if(sumprob <=0){
      fprintf(stderr,"compute_svm_adaboost: sumprob = 0\n");
      return 1;
    }
    for(i=0;i<n;i++)
      prob[i] /= sumprob;
    
  }
  
  if(esvm->nmodels<=0){
    fprintf(stderr,"compute_svm_adaboost: no models produced\n");
    return 1;
  }

  if(sumalpha <=0){
      fprintf(stderr,"compute_svm_adaboost: sumalpha = 0\n");
      return 1;
  }
  for(b=0;b<esvm->nmodels;b++)
    esvm->weights[b] /= sumalpha;
  
  free(trx);
  free_ivector(classes);
  free_ivector(try);
  free_ivector(pred);
  free_dvector(prob);
  free_dvector(prob_copy);
  return 0;

}





static void svm_smo(SupportVectorMachine *svm)
{
  int i,k;
  int numChanged;
  int examineAll;
  int nloops=0;


  svm->end_support_i=svm->n;

  if(svm->kernel_type==SVM_KERNEL_LINEAR){
    svm->kernel_func=dot_product_func;
    svm->learned_func=learned_func_linear;
  }

  if(svm->kernel_type==SVM_KERNEL_POLINOMIAL){
    svm->kernel_func=polinomial_kernel;
    svm->learned_func=learned_func_nonlinear;
  }

  if(svm->kernel_type==SVM_KERNEL_GAUSSIAN){
    /*
    svm->precomputed_self_dot_product=(double *)calloc(svm->n,sizeof(double));
    */
    for(i=0;i<svm->n;i++)
      svm->precomputed_self_dot_product[i] = dot_product_func(i,i,svm);
    svm->kernel_func=rbf_kernel;
    svm->learned_func=learned_func_nonlinear;
  }

  numChanged=0;
  examineAll=1;

  svm->convergence=1;
  while(svm->convergence==1 &&(numChanged>0 || examineAll)){
    numChanged=0;
    if(examineAll){
      for(k=0;k<svm->n;k++)
	numChanged += examineExample(k,svm);
    }else{
      for(k=0;k<svm->n;k++)
	if(svm->alph[k] > 0 && svm->alph[k] < svm->Cw[k])
	  numChanged += examineExample(k,svm);
    }
    if(examineAll==1)
      examineAll=0;
    else if(numChanged==0)
      examineAll=1;

    nloops+=1;
    if(nloops==svm->maxloops)
      svm->convergence=0;
    if(svm->verbose==1)
      fprintf(stdout,"%6d\b\b\b\b\b\b\b",nloops);
  }

}


static double learned_func_linear(k,svm)
     int k;
     SupportVectorMachine *svm;

{
  double s=0.0;
  int i;
  
  for(i=0;i<svm->d;i++)
    s += svm->w[i] * svm->x[k][i];

  s -= svm->b;

  return s;
}

static double learned_func_nonlinear(k,svm)
     int k;
     SupportVectorMachine *svm;
{
  double s=0.0;
  int i;

  for(i=0;i<svm->end_support_i;i++)
    if(svm->alph[i]>0)
      s += svm->alph[i]*svm->y[i]*svm->kernel_func(i,k,svm);

  s -= svm->b;

  return s;
}

static double polinomial_kernel(i1,i2,svm)
     int i1,i2;
     SupportVectorMachine *svm;
{     
  double s;

  s = pow(1+dot_product_func(i1,i2,svm),svm->two_sigma_squared);
  return s;
}


static double rbf_kernel(i1,i2,svm)
     int i1,i2;
     SupportVectorMachine *svm;
{     
  double s;

  s = dot_product_func(i1,i2,svm);

  s *= -2;

  s += svm->precomputed_self_dot_product[i1] + svm->precomputed_self_dot_product[i2];
  
  return exp(-s/svm->two_sigma_squared);
}


static double dot_product_func(i1,i2,svm)
     int i1,i2;
     SupportVectorMachine *svm;
{ 
  double dot = 0.0;
  int i;

  for(i=0;i<svm->d;i++)
    dot += svm->x[i1][i] * svm->x[i2][i];

  return dot;
}

static int examineExample(i1,svm)
     int i1;
     SupportVectorMachine *svm;
{
  double y1, alph1, E1, r1;
  
  y1=svm->y[i1];
  alph1=svm->alph[i1];
  
  if(alph1>0 && alph1<svm->Cw[i1])
    E1 = svm->error_cache[i1];
  else
    E1 = svm->learned_func(i1,svm)-y1;

  r1 = y1 *E1;

  if((r1<-svm->tolerance && alph1<svm->Cw[i1]) ||(r1>svm->tolerance && alph1>0)){
    {
      int k, i2;
      double tmax;

      for(i2=(-1),tmax=0,k=0;k<svm->end_support_i;k++)
	if(svm->alph[k]>0 && svm->alph[k]<svm->Cw[k]){
	  double E2,temp;

	  E2=svm->error_cache[k];

	  temp=fabs(E1-E2);
      
	  if(temp>tmax){
	    tmax=temp;
	    i2=k;
	  }
	}
  
      if(i2>=0){
	if(takeStep(i1,i2,svm))
	  return 1;
      }
    }
    {
      int k0,k,i2;
      for(k0=(int)(drand48()*svm->end_support_i),k=k0;k<svm->end_support_i+k0;k++){
	i2 = k % svm->end_support_i;
	if(svm->alph[i2]>0 && svm->alph[i2]<svm->Cw[i2]){
	  if(takeStep(i1,i2,svm))
	    return 1;
	}
      }
    }
    {
      int k0,k,i2;

      for(k0=(int)(drand48()*svm->end_support_i),k=k0;k<svm->end_support_i+k0;k++){
	i2 = k % svm->end_support_i;
	if(takeStep(i1,i2,svm))
	  return 1;
      }
    }
  }
  return 0;
}


static int takeStep(int i1, int i2, SupportVectorMachine *svm)
{
  int y1,y2,s;
  double alph1,alph2;
  double a1,a2;
  double E1,E2,L,H,k11,k12,k22,eta,Lobj,Hobj;

  if(i1==i2)
    return 0;

  alph1=svm->alph[i1];
  y1=svm->y[i1];
  if(alph1>0 && alph1<svm->Cw[i1])
    E1=svm->error_cache[i1];
  else
    E1=svm->learned_func(i1,svm)-y1;


  alph2=svm->alph[i2];
  y2=svm->y[i2];
  if(alph2>0 && alph2<svm->Cw[i2])
    E2=svm->error_cache[i2];
  else
    E2=svm->learned_func(i2,svm)-y2;

  s=y1*y2;

  if(y1==y2){
    double gamma;
    
    gamma = alph1+alph2;
    if(gamma-svm->Cw[i1]>0)
      L=gamma-svm->Cw[i1];
    else
      L=0.0;

    if(gamma<svm->Cw[i2])
      H=gamma;
    else
      H=svm->Cw[i2];


  }else{
    double gamma;
    
    gamma = alph2-alph1;

    if(gamma>0)
      L=gamma;
    else
      L=0.0;

    if(svm->Cw[i1]+gamma<svm->Cw[i2])
      H=svm->Cw[i1]+gamma;
    else
      H=svm->Cw[i2];
  }
  
  if(L==H)
    return 0;

  k11=svm->kernel_func(i1,i1,svm);
  k12=svm->kernel_func(i1,i2,svm);
  k22=svm->kernel_func(i2,i2,svm);


  eta=2*k12-k11-k22;

  if(eta<0){
    a2=alph2+y2*(E2-E1)/eta;
    if(a2<L)
      a2=L;
    else if(a2>H)
      a2=H;
  }else{
    {
      double c1,c2;

      c1=eta/2;
      c2=y2*(E1-E2)-eta*alph2;
      Lobj=c1*L*L+c2*L;
      Hobj=c1*H*H+c2*H;
    }
    if(Lobj>Hobj+svm->eps)
      a2=L;
    else if(Lobj<Hobj-svm->eps)
      a2=H;
    else
      a2=alph2;
  }
  
  if(fabs(a2-alph2)<svm->eps*(a2+alph2+svm->eps))
    return 0;

  a1=alph1-s*(a2-alph2);

  if(a1<0){
    a2 += s*a1;
    a1=0;
  }else if(a1>svm->Cw[i1]){
    double t;

    t=a1-svm->Cw[i1];
    a2 += s*t;
    a1=svm->Cw[i1];
  }

  {
    double b1,b2,bnew;

    if(a1>0 && a1 <svm->Cw[i1])
      bnew=svm->b+E1+y1*(a1-alph1)*k11+y2*(a2-alph2)*k12;
    else{
      if(a2>0 && a2 <svm->Cw[i2])
	bnew=svm->b+E2+y1*(a1-alph1)*k12+y2*(a2-alph2)*k22;
      else{
	b1=svm->b+E1+y1*(a1-alph1)*k11+y2*(a2-alph2)*k12;
	b2=svm->b+E2+y1*(a1-alph1)*k12+y2*(a2-alph2)*k22;
	bnew=(b1+b2)/2;
      }
    }

    svm->delta_b=bnew-svm->b;
    svm->b=bnew;
  }

  if(svm->kernel_type==SVM_KERNEL_LINEAR){
    double t1,t2;
    int i;

    t1=y1*(a1-alph1);
    t2=y2*(a2-alph2);

    for(i=0;i<svm->d;i++)
      svm->w[i] += svm->x[i1][i]*t1+svm->x[i2][i]*t2;
  }

  {
    double t1,t2;
    int i;

    t1=y1*(a1-alph1);
    t2=y2*(a2-alph2);
    
    for(i=0;i<svm->end_support_i;i++)
      svm->error_cache[i] += t1*svm->kernel_func(i1,i,svm)+
	t2*svm->kernel_func(i2,i,svm)-svm->delta_b;
    
  }
  
  svm->alph[i1]=a1;
  svm->alph[i2]=a2;

  return 1;


}
