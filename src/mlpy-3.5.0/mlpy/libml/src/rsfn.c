#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "ml.h"

static void proj(SlopeFunctions *sf, double *x_tr[],
		 int d, int y_tr[], double x[], double **x_proj);
static void s_f(SlopeFunctions *sf,double *x[],int y[],int n,int d,
		double threshold,int verbose,int knn);
static void svm_smo(SupportVectorMachine *svm);
static int examineExample(int i1, SupportVectorMachine *svm);
static int takeStep(int i1, int i2, SupportVectorMachine *svm);

static double learned_func_linear(int k, SupportVectorMachine *svm);
static double dot_product_func(int i1, int i2, SupportVectorMachine *svm);



static int compute_rsfn_bagging(ERegularizedSlopeFunctionNetworks *ersfn,
				int n,int d,
				double *x[],int y[],int nmodels,
				double C,double tol,double eps,
				int maxloops,int verbose, double threshold,
				int knn);
static int compute_rsfn_aggregate(ERegularizedSlopeFunctionNetworks *ersfn,
				  int n,int d,
				  double *x[],int y[],int nmodels,
				  double C,double tol,double eps,
				  int maxloops,int verbose, double threshold,
				  int knn);
static int compute_rsfn_adaboost(ERegularizedSlopeFunctionNetworks *ersfn,
				 int n,int d,
				 double *x[],int y[],int nmodels,
				 double C,double tol,double eps,
				 int maxloops,int verbose, double threshold,
				 int knn);


int compute_rsfn(RegularizedSlopeFunctionNetworks *rsfn,int n,int d,
		 double *x[],int y[],double C,double tol,
		 double eps,int maxloops,int verbose,double W[],
		 double threshold,int knn)
{
  int i,j;
  int nclasses;
  int *classes;

  rsfn->svm.n=n;
  rsfn->svm.C=C;
  rsfn->svm.tolerance=tol;
  rsfn->svm.eps=eps;
  rsfn->svm.two_sigma_squared=0.0;
  rsfn->svm.kernel_type=SVM_KERNEL_LINEAR;
  rsfn->svm.maxloops=maxloops;
  rsfn->svm.verbose=verbose;

  rsfn->threshold=threshold;

  rsfn->svm.b=0.0;

  if(C<=0){
    fprintf(stderr,"compute_rsfn: regularization parameter C must be > 0\n");
    return 1;
  }
  if(eps<=0){
    fprintf(stderr,"compute_rsfn: parameter eps must be > 0\n");
    return 1;
  }
  if(tol<=0){
    fprintf(stderr,"compute_rsfn: parameter tol must be > 0\n");
    return 1;
  }
  if(maxloops<=0){
    fprintf(stderr,"compute_rsfn: parameter maxloops must be > 0\n");
    return 1;
  }
  if(threshold<0. || threshold>1.){
    fprintf(stderr,"compute_rsfn: threshold must be in [0,1]\n");
    return 1;
  }
  if(W){
    for(i=0;i<n;i++)
      if(W[i]<=0){
	fprintf(stderr,"compute_rsfn: parameter W[%d] must be > 0\n",i);
	return 1;
      }
  }

  nclasses=iunique(y,n, &classes);

  if(nclasses<=0){
    fprintf(stderr,"compute_rsfn: iunique error\n");
    return 1;
  }
  if(nclasses==1){
    fprintf(stderr,"compute_rsfn: only 1 class recognized\n");
    return 1;
  }
  if(nclasses==2)
    if(classes[0] != -1 || classes[1] != 1){
      fprintf(stderr,"compute_rsfn: for binary classification classes must be -1,1\n");
      return 1;
    }
  if(nclasses>2){
    fprintf(stderr,"compute_rsfn: multiclass classification not allowed\n");
    return 1;
  }

  if(!(rsfn->svm.Cw=dvector(n))){
    fprintf(stderr,"compute_rsfn: out of memory\n");
    return 1;
  }
  if(!(rsfn->svm.alph=dvector(n))){
    fprintf(stderr,"compute_rsfn: out of memory\n");
    return 1;
  }
  if(!(rsfn->svm.error_cache=dvector(n))){
    fprintf(stderr,"compute_rsfn: out of memory\n");
    return 1;
  }

  /*
  if(!(rsfn->svm.precomputed_self_dot_product=dvector(n))){
    fprintf(stderr,"compute_rsfn: out of memory\n");
    return 1;
  }
  */
  if(!(rsfn->svm.K=dmatrix(n,n))){
    fprintf(stderr,"compute_rsfn: out of memory\n");
    return 1;
  }
  
  for(i=0;i<n;i++)
    rsfn->svm.error_cache[i]=-y[i];
  
  if(W){
    for(i=0;i<n;i++)
      rsfn->svm.Cw[i]=rsfn->svm.C * W[i];
  }else{
    for(i=0;i<n;i++)
      rsfn->svm.Cw[i]=rsfn->svm.C;
  }    

  if(verbose > 0)
    fprintf(stdout,"computing slope functions...\n");
  s_f(&(rsfn->sf),x,y,n,d,threshold,verbose,knn);
  if(verbose > 0){
    fprintf(stdout,"nsf=%d\n",rsfn->sf.nsf);
    fprintf(stdout,"...done!\n");
  }

#ifdef DEBUG
  for(i=0;i<rsfn->sf.nsf;i++)
    fprintf(stdout,"w[%f] b[%f] i[%d] j[%d]\n",rsfn->sf.w[i], rsfn->sf.b[i], 
	    rsfn->sf.i[i], rsfn->sf.j[i]);
#endif
  
  if(rsfn->sf.nsf<1){
    fprintf(stderr,"compute_rsfn: no slope functions (try to set threshold to a value lower than its current value %f)\n",threshold);
    return 1;
  }

  if(verbose > 0)
    fprintf(stdout,"projecting training data...\n");

  if(!(rsfn->svm.x=(double **)calloc(n,sizeof(double*)))){
    fprintf(stderr,"compute_rsfn: out of memory\n");
    return 1;
  }
  if(!(rsfn->svm.y=ivector(n))){
    fprintf(stderr,"compute_rsfn: out of memory\n");
    return 1;
  }
  for(i=0;i<n;i++)
    rsfn->svm.y[i]=y[i];
  
  for(i=0;i<n;i++){
    if(verbose > 1){
      fprintf(stdout,"%10d\b\b\b\b\b\b\b\b\b\b\b\b",i);
      fflush(stdout);
    }
    proj(&(rsfn->sf),x,d,y,x[i],&(rsfn->svm.x[i]));
  }
  if(verbose > 0)
    fprintf(stdout,"\n...done!\n");

#ifdef EXPORT_PROJECTED_DATA
   for(i=0;i<n;i++){
     fprintf(stdout,"%d",y[i]);
     for(j=0;j<rsfn->sf.nsf;j++)
       fprintf(stdout," %d:%f",j+1,rsfn->svm.x[i][j]);
     fprintf(stdout,"\n");
   }
   exit(0);
#endif

  rsfn->svm.d=rsfn->sf.nsf;

  if(!(rsfn->svm.w=dvector(rsfn->svm.d))){
    fprintf(stderr,"compute_rsfn: out of memory\n");
    return 1;
  }

  if(verbose > 0)
    fprintf(stdout,"computing linear svm...\n");
  svm_smo(&(rsfn->svm));

  free_dmatrix(rsfn->svm.x,n,rsfn->svm.d);

  if(verbose > 0)
    fprintf(stdout,"...done!\n");

  rsfn->x=dmatrix(n,d);
  for(i=0;i<n;i++)
    for(j=0;j<d;j++)
      rsfn->x[i][j]=x[i][j];
  rsfn->d=d;
      
  rsfn->svm.non_bound_support=rsfn->svm.bound_support=0;
  for(i=0;i<n;i++){
    if(rsfn->svm.alph[i]>0){
      if(rsfn->svm.alph[i]< rsfn->svm.Cw[i])
	rsfn->svm.non_bound_support++;
      else
	rsfn->svm.bound_support++;
    }
  }
  
  free_ivector(classes);

  return 0;
}

int compute_ersfn(ERegularizedSlopeFunctionNetworks *ersfn,int n,
		  int d,double *x[],
		 int y[],int method,int nmodels, double C,double tol,
		  double eps,int maxloops,int verbose,double threshold,
		  int knn)
    /*
       compute ensamble of rsfn models.x,y,n,d are the input data.
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
    if(compute_rsfn_bagging(ersfn,n,d,x,y,nmodels,C,tol,eps,
			   maxloops,verbose,threshold,knn) != 0){
      fprintf(stderr,"compute_ersfn: compute_rsfn_bagging error\n");
      return 1;
    }
    break;
  case AGGREGATE:
    if(compute_rsfn_aggregate(ersfn,n,d,x,y,nmodels,C,tol,eps,
			maxloops,verbose,threshold,knn) != 0){
      fprintf(stderr,"compute_ersfn: compute_rsfn_aggregate error\n");
      return 1;
    }
    break;
  case ADABOOST:
    if( compute_rsfn_adaboost(ersfn,n,d,x,y,nmodels,C,tol,eps,
			     maxloops,verbose,threshold,knn) != 0){
      fprintf(stderr,"compute_ersfn: compute_rsfn_adaboost error\n");
      return 1;
    }
    break;
  default:
    fprintf(stderr,"compute_ersfn: ensamble method not recognized\n");
    return 1;
    break;
  }

  return 0;
}
     
int predict_rsfn(RegularizedSlopeFunctionNetworks *rsfn,double x[],
		double **margin)
{
  double *tmp_x;
  int pred;

  proj(&(rsfn->sf),rsfn->x,rsfn->d,rsfn->svm.y,x,&(tmp_x));

  pred=predict_svm(&(rsfn->svm),tmp_x,margin);

  free_dvector(tmp_x);
  
  return pred;
}

int predict_ersfn(ERegularizedSlopeFunctionNetworks *ersfn, double x[],
		  double **margin)
     /*
       predicts rsfn model on a test point x. Proportions of neighbours
       for each class will be stored within the array margin 
       (an array of length rsfn->nclasses). 

       
       Return value: the predicted value on success (-1 or 1 for
       binary classification; 1,...,nclasses in the multiclass case),
       0 on succes with non unique classification, -2 otherwise.
     */
{
  int b;
  int pred;
  double *tmpmargin;
  
  if(!((*margin)=dvector(2))){
    fprintf(stderr,"predict_ersfn: out of memory\n");
    return -2;
  }
  
  for(b=0;b<ersfn->nmodels;b++){
    pred=predict_rsfn(&(ersfn->rsfn[b]), x,&tmpmargin);
    if(pred < -1){
      fprintf(stderr,"predict_ersfn: predict_rsfn error\n");
      return -2;
    }
    if(pred==-1)
      (*margin)[0] += ersfn->weights[b];
    else if(pred==1)
      (*margin)[1] += ersfn->weights[b];
    
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



static void proj(SlopeFunctions *sf, double *x_tr[],
		 int d, int y_tr[], double x[], double **x_proj)
{
  int t;
  double ps1,ps2;

  (*x_proj)=dvector(sf->nsf);
  
  for(t=0;t<sf->nsf;t++){
    ps1=scalar_product(x,x_tr[sf->i[t]],d);
    ps2=scalar_product(x,x_tr[sf->j[t]],d);
    (*x_proj)[t]=sf->w[t]*(y_tr[sf->i[t]]*ps1+y_tr[sf->j[t]]*ps2)+sf->b[t];
    if((*x_proj)[t]>1)
      (*x_proj)[t]=1.;
    if((*x_proj)[t]<-1)
      (*x_proj)[t]=-1.;
  }
}


static void s_f(SlopeFunctions *sf,double *x[],int y[],int n,int d,
		double threshold,int verbose,int knn)
{

  if(knn>0){
    int i,j;
    double ps_ij;
    double *ps;
    int *who;
    int *sort_index;
    double *dist_who;
    int nwho;
    int indx;
    int do_it;
    int h;
    
    ps=dvector(n);
    for(i=0;i<n;i++)
      ps[i]=scalar_product(x[i],x[i],d);


    sf->w=dvector(1);
    sf->b=dvector(1);
    sf->i=ivector(1);
    sf->j=ivector(1);
    
    sf->nsf=0;
    
    who=ivector(n);
    dist_who=dvector(n);
    sort_index=ivector(n);
    for(i=0;i<n;i++){
      if(verbose > 1){
	fprintf(stdout,"%10d\b\b\b\b\b\b\b\b\b\b\b\b",i);
	fflush(stdout);
      }
      nwho=0;
      for(j=0;j<n;j++){
	if(y[j] != y[i]){
	  who[nwho]=j;
	  dist_who[nwho++]=euclidean_squared_distance(x[i],x[j],d);
	}
      }
      for(j=0;j<nwho;j++)
	sort_index[j]=j;
      dsort(dist_who,sort_index,nwho,SORT_ASCENDING);
      for(j=0;j<knn;j++){
	indx=who[sort_index[j]];
	do_it=TRUE;
	for(h=0;h<sf->nsf;h++)
	  if((sf->i[h]==indx) && (sf->j[h]==i)){
	    do_it=FALSE;
	    break;
	  }
	if(do_it){
	  ps_ij=scalar_product(x[i],x[indx],d);
	  
	  sf->w[sf->nsf]=(y[indx]-y[i])/
	    (y[indx]*ps[indx]-y[i]*ps[i]-(y[indx]-y[i])*ps_ij);
	  sf->b[sf->nsf]=y[i]-sf->w[sf->nsf]*(y[i]*ps[i]+y[indx]*ps_ij);
	  sf->i[sf->nsf]=i;
	  sf->j[sf->nsf]=indx;
	  
	  sf->nsf++;
	  sf->w=(double*)realloc(sf->w,(sf->nsf+1)*sizeof(double));
	  sf->b=(double*)realloc(sf->b,(sf->nsf+1)*sizeof(double));
	  sf->i=(int*)realloc(sf->i,(sf->nsf+1)*sizeof(int));
	  sf->j=(int*)realloc(sf->j,(sf->nsf+1)*sizeof(int));
	}
      }
    }
    if(verbose > 0)
      fprintf(stdout,"\n");
  }else{
    int i,j,k;
    double ps_ij;
    double *ps;
    int n_below_one;
    double ps1,ps2;
    double out;
    double w_tmp;
    double b_tmp;
    int save;

  
    ps=dvector(n);
    for(i=0;i<n;i++)
      ps[i]=scalar_product(x[i],x[i],d);
    
    threshold=n*(1.-threshold);
    
    sf->w=dvector(1);
    sf->b=dvector(1);
    sf->i=ivector(1);
    sf->j=ivector(1);
    
    sf->nsf=0;
    
    for(i=0;i<n;i++){
      if(verbose > 1)
	if(i%100==0){
	  fprintf(stdout,"%10d\b\b\b\b\b\b\b\b\b\b\b\b",i);
	  fflush(stdout);
	}
      if(y[i]==-1){
	for(j=0;j<n;j++){
	  if(y[j]==1){
	  
	    ps_ij=scalar_product(x[i],x[j],d);
	  
	    w_tmp=(y[j]-y[i])/
	      (y[j]*ps[j]-y[i]*ps[i]-(y[j]-y[i])*ps_ij);
	  
	    b_tmp=y[i]-w_tmp*(y[i]*ps[i]+y[j]*ps_ij);

	    n_below_one=0;
	    save=1;
	    for(k=0;k<n;k++){
	      ps1=scalar_product(x[k],x[i],d);
	      ps2=scalar_product(x[k],x[j],d);
	    
	      out=w_tmp*(y[i]*ps1 + y[j]*ps2)+b_tmp;

	      if(out>=1)
		out=1.;
	      else if(out<=-1)
		out=-1.;
	      else
		n_below_one++;

	      if(n_below_one>threshold){
		save=0;
		break;
	      }
	    }

	    if(save){
	      sf->w[sf->nsf]=w_tmp;
	      sf->b[sf->nsf]=b_tmp;
	      sf->i[sf->nsf]=i;
	      sf->j[sf->nsf]=j;
	      sf->nsf++;
	      sf->w=(double*)realloc(sf->w,(sf->nsf+1)*sizeof(double));
	      sf->b=(double*)realloc(sf->b,(sf->nsf+1)*sizeof(double));
	      sf->i=(int*)realloc(sf->i,(sf->nsf+1)*sizeof(int));
	      sf->j=(int*)realloc(sf->j,(sf->nsf+1)*sizeof(int));
	    
	    }
	  }
	}
      }
    }
  
  
    if(verbose > 0)
      fprintf(stdout,"\n");
    free_dvector(ps);
  }
}



static int compute_rsfn_bagging(ERegularizedSlopeFunctionNetworks *ersfn,
				int n,int d,
				double *x[],int y[],int nmodels,
				double C,double tol,double eps,
				int maxloops,int verbose,double threshold,
				int knn)
{
  int i,b;
  int *samples;
  double **trx;
  int *try;
  int nclasses;
  int *classes;

  if(nmodels<1){
    fprintf(stderr,"compute_rsfn_bagging: nmodels must be greater than 0\n");
    return 1;
  }

  if(C<=0){
    fprintf(stderr,"compute_rsfn_bagging: regularization parameter C must be > 0\n");
    return 1;
  }
  if(eps<=0){
    fprintf(stderr,"compute_rsfn_bagging: parameter eps must be > 0\n");
    return 1;
  }
  if(tol<=0){
    fprintf(stderr,"compute_rsfn_bagging: parameter tol must be > 0\n");
    return 1;
  }
  if(maxloops<=0){
    fprintf(stderr,"compute_rsfn_bagging: parameter maxloops must be > 0\n");
    return 1;
  }

  if(threshold<0. || threshold>1.){
    fprintf(stderr,"compute_rsfn_bagging: threshold must be in [0,1]\n");
    return 1;
  }

  nclasses=iunique(y,n, &classes);

  if(nclasses<=0){
    fprintf(stderr,"compute_rsfn_bagging: iunique error\n");
    return 1;
  }
  if(nclasses==1){
    fprintf(stderr,"compute_rsfn_bagging: only 1 class recognized\n");
    return 1;
  }
  if(nclasses==2)
    if(classes[0] != -1 || classes[1] != 1){
      fprintf(stderr,"compute_rsfn_bagging: for binary classification classes must be -1,1\n");
      return 1;
    }
  if(nclasses>2){
    fprintf(stderr,"compute_rsfn_bagging: multiclass classification not allowed\n");
    return 1;
  }


  if(!(ersfn->rsfn=(RegularizedSlopeFunctionNetworks *)
       calloc(nmodels,sizeof(RegularizedSlopeFunctionNetworks)))){
    fprintf(stderr,"compute_rsfn_bagging: out of memory\n");
    return 1;
  }
  ersfn->nmodels=nmodels;
  if(!(ersfn->weights=dvector(nmodels))){
    fprintf(stderr,"compute_rsfn_bagging: out of memory\n");
    return 1;
  }

  for(b=0;b<nmodels;b++)
    ersfn->weights[b]=1.0 / (double) nmodels;
  
  if(!(trx=(double **)calloc(n,sizeof(double*)))){
    fprintf(stderr,"compute_rsfn_bagging: out of memory\n");
    return 1;
  }
  if(!(try=ivector(n))){
    fprintf(stderr,"compute_rsfn_bagging: out of memory\n");
    return 1;
  }
  
  for(b=0;b<nmodels;b++){
    if(sample(n, NULL, n, &samples, TRUE,b)!=0){
       fprintf(stderr,"compute_rsfn_bagging: sample error\n");
       return 1;
    }

    for(i =0;i<n;i++){
      trx[i] = x[samples[i]];
      try[i] = y[samples[i]];
    }

    if(compute_rsfn(&(ersfn->rsfn[b]),n,d,trx,try,C,
		   tol,eps,maxloops,verbose,NULL,threshold,knn)!=0){
      fprintf(stderr,"compute_rsfn_bagging: compute_rsfn error\n");
      return 1;
    }
    free_ivector(samples);

  }

  free(trx);
  free_ivector(classes);
  free_ivector(try);
    
  return 0;

}



static int compute_rsfn_aggregate(ERegularizedSlopeFunctionNetworks *ersfn,
				  int n,int d,
				  double *x[],int y[],int nmodels,
				  double C,double tol,double eps,
				  int maxloops,int verbose,double threshold,
				  int knn)
{
  int i,b;
  int *samples;
  double **trx;
  int *try;
  int indx;
  int nclasses;
  int *classes;

  if(nmodels<1){
    fprintf(stderr,"compute_rsfn_aggregate: nmodels must be greater than 0\n");
    return 1;
  }

  if(nmodels > n){
    fprintf(stderr,"compute_rsfn_aggregate: nmodels must be less than n\n");
    return 1;
  }

  if(C<=0){
    fprintf(stderr,"compute_rsfn_aggregate: regularization parameter C must be > 0\n");
    return 1;
  }
  if(eps<=0){
    fprintf(stderr,"compute_rsfn_aggregate: parameter eps must be > 0\n");
    return 1;
  }
  if(tol<=0){
    fprintf(stderr,"compute_rsfn_aggregate: parameter tol must be > 0\n");
    return 1;
  }
  if(maxloops<=0){
    fprintf(stderr,"compute_rsfn_aggregate: parameter maxloops must be > 0\n");
    return 1;
  }

  if(threshold<0. || threshold>1.){
    fprintf(stderr,"compute_rsfn_aggregate: threshold must be in [0,1]\n");
    return 1;
  }

  nclasses=iunique(y,n, &classes);

  if(nclasses<=0){
    fprintf(stderr,"compute_rsfn_aggregate: iunique error\n");
    return 1;
  }
  if(nclasses==1){
    fprintf(stderr,"compute_rsfn_aggregate: only 1 class recognized\n");
    return 1;
  }
  if(nclasses==2)
    if(classes[0] != -1 || classes[1] != 1){
      fprintf(stderr,"compute_rsfn_aggregate: for binary classification classes must be -1,1\n");
      return 1;
    }
  if(nclasses>2){
    fprintf(stderr,"compute_rsfn_aggregate: multiclass classification not allowed\n");
    return 1;
  }

  if(!(ersfn->rsfn=(RegularizedSlopeFunctionNetworks *)
       calloc(nmodels,sizeof(RegularizedSlopeFunctionNetworks)))){
    fprintf(stderr,"compute_rsfn_aggregate: out of memory\n");
    return 1;
  }
  ersfn->nmodels=nmodels;
  if(!(ersfn->weights=dvector(nmodels))){
    fprintf(stderr,"compute_rsfn_aggregate: out of memory\n");
    return 1;
  }

  for(b=0;b<nmodels;b++)
    ersfn->weights[b]=1.0 / (double) nmodels;
  
  if(!(trx=(double **)calloc(n,sizeof(double*)))){
    fprintf(stderr,"compute_rsfn_aggregate: out of memory\n");
    return 1;
  }
  if(!(try=ivector(n))){
    fprintf(stderr,"compute_rsfn_aggregate: out of memory\n");
    return 1;
  }
  
  if(sample(nmodels, NULL, n, &samples, TRUE,0)!=0){
    fprintf(stderr,"compute_rsfn_aggregate: sample error\n");
    return 1;
  }

  for(b=0;b<nmodels;b++){
  
    indx=0;
    for(i=0;i<n;i++)
      if(samples[i] == b){
	trx[indx] = x[i];
	try[indx++] = y[i];
      }

    if(compute_rsfn(&(ersfn->rsfn[b]),indx,d,trx,try,C,
		   tol,eps,maxloops,verbose,NULL,threshold,knn)!=0){
      fprintf(stderr,"compute_rsfn_aggregate: compute_rsfn error\n");
      return 1;
    }

  }

  free_ivector(samples);
  free(trx);
  free_ivector(classes);
  free_ivector(try);
    
  return 0;

}

static int compute_rsfn_adaboost(ERegularizedSlopeFunctionNetworks *ersfn,
				 int n,int d,
				 double *x[],int y[],int nmodels,
				 double C,double tol,double eps,
				 int maxloops,int verbose,double threshold,
				 int knn)
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
    fprintf(stderr,"compute_rsfn_adaboost: nmodels must be greater than 0\n");
    return 1;
  }

 if(C<=0){
    fprintf(stderr,"compute_rsfn_adaboost: regularization parameter C must be > 0\n");
    return 1;
  }
  if(eps<=0){
    fprintf(stderr,"compute_rsfn_adaboost: parameter eps must be > 0\n");
    return 1;
  }
  if(tol<=0){
    fprintf(stderr,"compute_rsfn_adaboost: parameter tol must be > 0\n");
    return 1;
  }
  if(maxloops<=0){
    fprintf(stderr,"compute_rsfn_adaboost: parameter maxloops must be > 0\n");
    return 1;
  }


  if(threshold<0. || threshold>1.){
    fprintf(stderr,"compute_rsfn_adaboost: threshold must be in [0,1]\n");
    return 1;
  }


  nclasses=iunique(y,n, &classes);

  if(nclasses<=0){
    fprintf(stderr,"compute_rsfn_adaboost: iunique error\n");
    return 1;
  }
  if(nclasses==1){
    fprintf(stderr,"compute_rsfn_adaboost: only 1 class recognized\n");
    return 1;
  }
  if(nclasses==2)
    if(classes[0] != -1 || classes[1] != 1){
      fprintf(stderr,"compute_rsfn_adaboost: for binary classification classes must be -1,1\n");
      return 1;
    }
  if(nclasses>2){
    fprintf(stderr,"compute_rsfn_adaboost: multiclass classification not allowed\n");
    return 1;
  }

  if(!(ersfn->rsfn=(RegularizedSlopeFunctionNetworks *)
       calloc(nmodels,sizeof(RegularizedSlopeFunctionNetworks)))){
    fprintf(stderr,"compute_rsfn_adaboost: out of memory\n");
    return 1;
  }

  if(!(ersfn->weights=dvector(nmodels))){
    fprintf(stderr,"compute_rsfn_adaboost: out of memory\n");
    return 1;
  }

  if(!(trx=(double **)calloc(n,sizeof(double*)))){
    fprintf(stderr,"compute_rsfn_adaboost: out of memory\n");
    return 1;
  }
  if(!(try=ivector(n))){
    fprintf(stderr,"compute_rsfn_adaboost: out of memory\n");
    return 1;
  }
  
  if(!(prob_copy=dvector(n))){
    fprintf(stderr,"compute_rsfn_adaboost: out of memory\n");
    return 1;
  }
  if(!(prob=dvector(n))){
    fprintf(stderr,"compute_rsfn_adaboost: out of memory\n");
    return 1;
  }

  if(!(pred=ivector(n))){
    fprintf(stderr,"compute_rsfn_adaboost: out of memory\n");
    return 1;
  }

  for(i =0;i<n;i++)
    prob[i]=1.0/(double)n;

  ersfn->nmodels=nmodels;
  sumalpha=0.0;
  for(b=0;b<nmodels;b++){

    for(i =0;i<n;i++)
      prob_copy[i]=prob[i];
    if(sample(n, prob_copy, n, &samples, TRUE,b)!=0){
      fprintf(stderr,"compute_rsfn_adaboost: sample error\n");
      return 1;
    }

    for(i =0;i<n;i++){
      trx[i] = x[samples[i]];
      try[i] = y[samples[i]];
    }
    
    if(compute_rsfn(&(ersfn->rsfn[b]),n,d,trx,try,C,
		   tol,eps,maxloops,verbose,NULL,threshold,knn)!=0){
      fprintf(stderr,"compute_rsfn_adaboost: compute_rsfn error\n");
      return 1;
    }
    free_ivector(samples);

    epsilon=0.0;
    for(i=0;i<n;i++){
      pred[i]=predict_rsfn(&(ersfn->rsfn[b]),x[i],&margin);
      if(pred[i] < -1 ){
	fprintf(stderr,"compute_rsfn_adaboost: predict_rsfn error\n");
	return 1;
      }
      if(pred[i]==0 || pred[i] != y[i])
	epsilon += prob[i];
      free_dvector(margin);
    }
    
    if(epsilon > 0 && epsilon < 0.5){
      ersfn->weights[b]=0.5 *log((1.0-epsilon)/epsilon);
      sumalpha+=ersfn->weights[b];
    }else{
      ersfn->nmodels=b;
      break;
    }
      
    sumprob=0.0;
    for(i=0;i<n;i++){
      prob[i]=prob[i]*exp(-ersfn->weights[b]*y[i]*pred[i]);
      sumprob+=prob[i];
    }

    if(sumprob <=0){
      fprintf(stderr,"compute_rsfn_adaboost: sumprob = 0\n");
      return 1;
    }
    for(i=0;i<n;i++)
      prob[i] /= sumprob;
    
  }
  
  if(ersfn->nmodels<=0){
    fprintf(stderr,"compute_rsfn_adaboost: no models produced\n");
    return 1;
  }

  if(sumalpha <=0){
      fprintf(stderr,"compute_rsfn_adaboost: sumalpha = 0\n");
      return 1;
  }
  for(b=0;b<ersfn->nmodels;b++)
    ersfn->weights[b] /= sumalpha;
  
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

  svm->kernel_func=dot_product_func;
  svm->learned_func=learned_func_linear;

  if(svm->verbose > 0)
    fprintf(stdout,"precomputing scalar products...\n");

  for(i=0;i<svm->n;i++){
    if(svm->verbose > 1){
      fprintf(stdout,"%10d\b\b\b\b\b\b\b\b\b\b\b\b",i);
      fflush(stdout);
    }
    for(k=i;k<svm->n;k++){
      svm->K[i][k]=svm->kernel_func(i,k,svm);
      if(k!=i)
        svm->K[k][i]=svm->K[i][k];
    }
  }
 if(svm->verbose > 0 )
   fprintf(stdout,"\n");

  numChanged=0;
  examineAll=1;

  if(svm->verbose > 0)
    fprintf(stdout,"optimization loops...\n");
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
    if(svm->verbose > 1){
      fprintf(stdout,"%6d\b\b\b\b\b\b\b",nloops);
      fflush(stdout);
    }
  }
  if(svm->verbose > 0){
    if(svm->convergence==1)
      fprintf(stdout,"\n...done!\n");
    else
      fprintf(stdout,"\n...done! but did not converged\n");
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


static int takeStep(i1,i2,svm)
     int i1,i2;
     SupportVectorMachine *svm;
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

  k11=svm->K[i1][i1];
  k12=svm->K[i1][i2];
  k22=svm->K[i2][i2];

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

  {
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
      svm->error_cache[i] += t1*svm->K[i1][i]+
	t2*svm->K[i2][i]-svm->delta_b;
    
  }
  
  svm->alph[i1]=a1;
  svm->alph[i2]=a2;

  return 1;


}
