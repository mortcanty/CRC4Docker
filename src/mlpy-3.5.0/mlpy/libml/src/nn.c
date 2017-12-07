#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "ml.h"

static int compute_nn_bagging(ENearestNeighbor *enn,int n,int d,double *x[],
			      int y[], int nmodels,int k, int dist);
static int compute_nn_aggregate(ENearestNeighbor *enn,int n,int d,double *x[],
				int y[], int nmodels,int k, int dist);
static int compute_nn_adaboost(ENearestNeighbor *enn,int n,int d,double *x[],
			       int y[], int nmodels,int k, int dist);

int compute_nn(NearestNeighbor *nn,int n,int d,double *x[],int y[],
	       int k, int dist)
     /*
       Compute nn model. x,y,n,d are the input data.
       k is the number of NN.
       dist is the adopted distance.

       Return value: 0 on success, 1 otherwise.
     */
{
  int i,j;

  // mlpy comments
  /*
  if(k>n){
    fprintf(stderr,"compute_nn: k must be smaller than n\n");
    return 1;
  }
  
  switch(dist){
  case DIST_SQUARED_EUCLIDEAN:
    break;
  case DIST_EUCLIDEAN:
    break;
  default:
    fprintf(stderr,"compute_nn: distance not recognized\n");
    return 1;
  }
  */

  nn->n=n;
  nn->d=d;
  nn->k=k;
  nn->dist=dist;

  
  nn->nclasses=iunique(y,n, &(nn->classes));
  
  // mlpy comments
  /*
  if(nn->nclasses<=0){
    fprintf(stderr,"compute_nn: iunique error\n");
    return 1;
  }
  if(nn->nclasses==1){
    fprintf(stderr,"compute_nn: only 1 class recognized\n");
    return 1;
  }

  if(nn->nclasses==2)
    if(nn->classes[0] != -1 || nn->classes[1] != 1){
      fprintf(stderr,"compute_nn: for binary classification classes must be -1,1\n");
      return 1;
    }
  
  if(nn->nclasses>2)
    for(i=0;i<nn->nclasses;i++)
      if(nn->classes[i] != i+1){
	fprintf(stderr,"compute_nn: for %d-class classification classes must be 1,...,%d\n",nn->nclasses,nn->nclasses);
	return 1;
      }
  */

  if(!(nn->x=dmatrix(n,d))){
    //fprintf(stderr,"compute_nn: out of memory\n");
    return 1;
  }
  if(!(nn->y=ivector(n))){
    //fprintf(stderr,"compute_nn: out of memory\n");
    return 1;
  }

  for(i=0;i<n;i++){
    for(j=0;j<d;j++)
      nn->x[i][j]=x[i][j];
    nn->y[i]=y[i];
  }

  return 0;

}

int compute_enn(ENearestNeighbor *enn,int n,int d,double *x[],
		 int y[],int method,int nmodels,int k,int dist)
    /*
       compute ensamble of nn models.x,y,n,d are the input data.
       method is one of BAGGING,AGGREGATE,ADABOOST.
        k is the number of NN.
       dist is the adopted distance.

       Return value: 0 on success, 1 otherwise.
     */
{
  switch(method){
  case BAGGING:
    if(compute_nn_bagging(enn,n,d,x,y,nmodels,k,dist) != 0){
      fprintf(stderr,"compute_enn: compute_nn_bagging error\n");
      return 1;
    }
    break;
  case AGGREGATE:
    if(compute_nn_aggregate(enn,n,d,x,y,nmodels,k,dist) != 0){
      fprintf(stderr,"compute_enn: compute_nn_aggregate error\n");
      return 1;
    }
    break;
  case ADABOOST:
    if( compute_nn_adaboost(enn,n,d,x,y,nmodels,k,dist) != 0){
      fprintf(stderr,"compute_enn: compute_nn_adaboost error\n");
      return 1;
    }
    break;
  default:
    fprintf(stderr,"compute_enn: ensamble method not recognized\n");
    return 1;
    break;
  }

  return 0;
}
     
int predict_nn(NearestNeighbor *nn, double x[],double **margin)
     /*
       predicts nn model on a test point x. Proportions of neighbours
       for each class will be stored within the array margin 
       (an array of length nn->nclasses). 

       
       Return value: the predicted value on success (-1 or 1 for
       binary classification; 1,...,nclasses in the multiclass case),
       0 on succes with non unique classification, -2 otherwise.
     */
{
  int i,j;
  double *dist;
  int *indx;
  int *knn_pred;
  double one_k;
  int pred_class=-2;
  double pred_n;

  // mlpy comments

  if(!((*margin)=dvector(nn->nclasses))){
    //fprintf(stderr,"predict_nn: out of memory\n");
    return -2;
  }
  if(!(dist=dvector(nn->n))){
    //fprintf(stderr,"predict_nn: out of memory\n");
    return -2;
  }
  if(!(indx=ivector(nn->n))){
    //fprintf(stderr,"predict_nn: out of memory\n");
    return -2;
  }
  if(!(knn_pred=ivector(nn->k))){
    //fprintf(stderr,"predict_nn: out of memory\n");
    return -2;
  }

  switch(nn->dist){
  case DIST_SQUARED_EUCLIDEAN:
    for(i=0;i<nn->n;i++)
      dist[i]=euclidean_squared_distance(x,nn->x[i],nn->d);
    break;
  case DIST_EUCLIDEAN:
    for(i=0;i<nn->n;i++)
      dist[i]=euclidean_squared_distance(x,nn->x[i],nn->d);
    break;
  default:
    //fprintf(stderr,"predict_nn: distance not recognized\n");
    return -2;
  }

  
  for(i=0;i<nn->n;i++)
    indx[i]=i;
  dsort(dist,indx,nn->n,SORT_ASCENDING);

  for(i=0;i<nn->k;i++)
    knn_pred[i]=nn->y[indx[i]];

  one_k=1.0/nn->k;
  for(i=0;i<nn->k;i++)
    for(j=0;j<nn->nclasses;j++)
      if(knn_pred[i] == nn->classes[j]){
	(*margin)[j] += one_k;
	break;
      }

  pred_class=nn->classes[0];
  pred_n=(*margin)[0];
  for(j=1;j<nn->nclasses;j++)
    if((*margin)[j]> pred_n){
      pred_class=nn->classes[j];
      pred_n=(*margin)[j];
    }
  
  for(j=0;j<nn->nclasses;j++)
    if(nn->classes[j] != pred_class)
      if(fabs((*margin)[j]-pred_n) < one_k/10.0){
	pred_class = 0;
	break;
      }
  
  free_dvector(dist);
  free_ivector(indx);
  free_ivector(knn_pred);
  
  return pred_class;
  
}



int predict_enn(ENearestNeighbor *enn, double x[],double **margin)
     /*
       predicts nn model on a test point x. Proportions of neighbours
       for each class will be stored within the array margin 
       (an array of length nn->nclasses). 

       
       Return value: the predicted value on success (-1 or 1 for
       binary classification; 1,...,nclasses in the multiclass case),
       0 on succes with non unique classification, -2 otherwise.
     */
{
  int i,b;
  int pred;
  double *tmpmargin;
  double maxmargin;
  
  if(!((*margin)=dvector(enn->nclasses))){
    fprintf(stderr,"predict_enn: out of memory\n");
    return -2;
  }
  
  if(enn->nclasses==2){
    for(b=0;b<enn->nmodels;b++){
      pred=predict_nn(&(enn->nn[b]), x,&tmpmargin);
      if(pred < -1){
	fprintf(stderr,"predict_enn: predict_nn error\n");
	return -2;
      }
      if(pred==-1)
	(*margin)[0] += enn->weights[b];
      else if(pred==1)
	(*margin)[1] += enn->weights[b];
      
      free_dvector(tmpmargin);
    }

    if((*margin)[0] > (*margin)[1])
      return -1;
    else if((*margin)[0] < (*margin)[1])
      return 1;
    else
      return 0;
  }else{
    for(b=0;b<enn->nmodels;b++){
      pred=predict_nn(&(enn->nn[b]), x,&tmpmargin);
      if(pred < -1){
	fprintf(stderr,"predict_enn: predict_nn error\n");
	return -2;
      }
      
      if(pred>0)
	(*margin)[pred-1] += enn->weights[b];
      
      free_dvector(tmpmargin);
    }

    maxmargin=0.0;
    pred=0;
    for(i=0;i<enn->nclasses;i++)
      if((*margin)[i]>maxmargin){
	maxmargin=(*margin)[i];
	pred=i;
      }

    for(i=0;i<enn->nclasses;i++)
      if(i != pred)
	if((*margin)[i] == maxmargin)
	  return 0;

    return pred+1;
  }

  return -2;
  
}


static int compute_nn_bagging(ENearestNeighbor *enn,int n,int d,double *x[],
			      int y[], int nmodels,int k, int dist)
{
  int i,b;
  int *samples;
  double **trx;
  int *try;

  if(nmodels<1){
    fprintf(stderr,"compute_nn_bagging: nmodels must be greater than 0\n");
    return 1;
  }

  if(k>n){
    fprintf(stderr,"compute_nn_bagging: k must be smaller than n\n");
    return 1;
  }
  
  switch(dist){
  case DIST_SQUARED_EUCLIDEAN:
    break;
  case DIST_EUCLIDEAN:
    break;
  default:
    fprintf(stderr,"compute_nn_bagging: distance not recognized\n");
    return 1;
  }

  enn->nclasses=iunique(y,n, &(enn->classes));

  if(enn->nclasses<=0){
    fprintf(stderr,"compute_nn_bagging: iunique error\n");
    return 1;
  }
  if(enn->nclasses==1){
    fprintf(stderr,"compute_nn_bagging: only 1 class recognized\n");
    return 1;
  }

  if(enn->nclasses==2)
    if(enn->classes[0] != -1 || enn->classes[1] != 1){
      fprintf(stderr,"compute_nn_bagging: for binary classification classes must be -1,1\n");
      return 1;
    }
  
  if(enn->nclasses>2)
    for(i=0;i<enn->nclasses;i++)
      if(enn->classes[i] != i+1){
	fprintf(stderr,"compute_nn_bagging: for %d-class classification classes must be 1,...,%d\n",enn->nclasses,enn->nclasses);
	return 1;
      }

  if(!(enn->nn=(NearestNeighbor *)calloc(nmodels,sizeof(NearestNeighbor)))){
    fprintf(stderr,"compute_nn_bagging: out of memory\n");
    return 1;
  }
  enn->nmodels=nmodels;
  if(!(enn->weights=dvector(nmodels))){
    fprintf(stderr,"compute_nn_bagging: out of memory\n");
    return 1;
  }
  enn->k=k;
  enn->dist=dist;

  for(b=0;b<nmodels;b++)
    enn->weights[b]=1.0 / (double) nmodels;
  
  if(!(trx=(double **)calloc(n,sizeof(double*)))){
    fprintf(stderr,"compute_nn_bagging: out of memory\n");
    return 1;
  }
  if(!(try=ivector(n))){
    fprintf(stderr,"compute_nn_bagging: out of memory\n");
    return 1;
  }
  
  for(b=0;b<nmodels;b++){
    if(sample(n, NULL, n, &samples, TRUE,b)!=0){
       fprintf(stderr,"compute_nn_bagging: sample error\n");
       return 1;
    }

    for(i =0;i<n;i++){
      trx[i] = x[samples[i]];
      try[i] = y[samples[i]];
    }

    if(compute_nn(&(enn->nn[b]),n,d,trx,try,k,dist)!=0){
      fprintf(stderr,"compute_nn_bagging: compute_nn error\n");
      return 1;
    }
    free_ivector(samples);

  }

  free(trx);
  free_ivector(try);
    
  return 0;

}



static int compute_nn_aggregate(ENearestNeighbor *enn,int n,int d,double *x[],
				int y[],int nmodels,int k, int dist)
{
  int i,b;
  int *samples;
  double **trx;
  int *try;
  int indx;

  if(nmodels<1){
    fprintf(stderr,"compute_nn_aggregate: nmodels must be greater than 0\n");
    return 1;
  }

  if(nmodels > n){
    fprintf(stderr,"compute_nn_aggregate: nmodels must be less than n\n");
    return 1;
  }

  if(k>n){
    fprintf(stderr,"compute_nn_aggregate: k must be smaller than n\n");
    return 1;
  }
  
  switch(dist){
  case DIST_SQUARED_EUCLIDEAN:
    break;
  case DIST_EUCLIDEAN:
    break;
  default:
    fprintf(stderr,"compute_nn_aggregate: distance not recognized\n");
    return 1;
  }

  enn->nclasses=iunique(y,n, &(enn->classes));

  if(enn->nclasses<=0){
    fprintf(stderr,"compute_nn_aggregate: iunique error\n");
    return 1;
  }
  if(enn->nclasses==1){
    fprintf(stderr,"compute_nn_aggregate: only 1 class recognized\n");
    return 1;
  }

  if(enn->nclasses==2)
    if(enn->classes[0] != -1 || enn->classes[1] != 1){
      fprintf(stderr,"compute_nn_aggregate: for binary classification classes must be -1,1\n");
      return 1;
    }
  
  if(enn->nclasses>2)
    for(i=0;i<enn->nclasses;i++)
      if(enn->classes[i] != i+1){
	fprintf(stderr,"compute_nn_aggregate: for %d-class classification classes must be 1,...,%d\n",enn->nclasses,enn->nclasses);
	return 1;
      }

  if(!(enn->nn=(NearestNeighbor *)calloc(nmodels,sizeof(NearestNeighbor)))){
    fprintf(stderr,"compute_nn_aggregate: out of memory\n");
    return 1;
  }
  enn->nmodels=nmodels;
  if(!(enn->weights=dvector(nmodels))){
    fprintf(stderr,"compute_nn_aggregate: out of memory\n");
    return 1;
  }
  enn->k=k;
  enn->dist=dist;

  for(b=0;b<nmodels;b++)
    enn->weights[b]=1.0 / (double) nmodels;
  
  if(!(trx=(double **)calloc(n,sizeof(double*)))){
    fprintf(stderr,"compute_nn_aggregate: out of memory\n");
    return 1;
  }
  if(!(try=ivector(n))){
    fprintf(stderr,"compute_nn_aggregate: out of memory\n");
    return 1;
  }
  
  if(sample(nmodels, NULL, n, &samples, TRUE,0)!=0){
    fprintf(stderr,"compute_nn_aggregate: sample error\n");
    return 1;
  }

  for(b=0;b<nmodels;b++){
  
    indx=0;
    for(i=0;i<n;i++)
      if(samples[i] == b){
	trx[indx] = x[i];
	try[indx++] = y[i];
      }

    if(compute_nn(&(enn->nn[b]),indx,d,trx,try,k,dist)!=0){
      fprintf(stderr,"compute_nn_aggregate: compute_nn error\n");
      return 1;
    }

  }

  free_ivector(samples);
  free(trx);
  free_ivector(try);
    
  return 0;

}

static int compute_nn_adaboost(ENearestNeighbor *enn,int n,int d,double *x[],
			       int y[],int nmodels,int k, int dist)
{
  int i,b;
  int *samples;
  double **trx;
  int *try;
  double *prob;
  double *prob_copy;
  double sumalpha;
  double eps;
  int *pred;
  double *margin;
  double sumprob;
  

  if(nmodels<1){
    fprintf(stderr,"compute_nn_adaboost: nmodels must be greater than 0\n");
    return 1;
  }

  if(k>n){
    fprintf(stderr,"compute_nn_adaboost: k must be smaller than n\n");
    return 1;
  }
  
  switch(dist){
  case DIST_SQUARED_EUCLIDEAN:
    break;
  case DIST_EUCLIDEAN:
    break;
  default:
    fprintf(stderr,"compute_nn_adaboost: distance not recognized\n");
    return 1;
  }

  enn->nclasses=iunique(y,n, &(enn->classes));

  if(enn->nclasses<=0){
    fprintf(stderr,"compute_nn_adaboost: iunique error\n");
    return 1;
  }
  if(enn->nclasses==1){
    fprintf(stderr,"compute_nn_adaboost: only 1 class recognized\n");
    return 1;
  }

  if(enn->nclasses==2)
    if(enn->classes[0] != -1 || enn->classes[1] != 1){
      fprintf(stderr,"compute_nn_adaboost: for binary classification classes must be -1,1\n");
      return 1;
    }
  
  if(enn->nclasses>2){
    fprintf(stderr,"compute_nn_adaboost: multiclass classification not allowed\n");
    return 1;
  }

  if(!(enn->nn=(NearestNeighbor *)calloc(nmodels,sizeof(NearestNeighbor)))){
    fprintf(stderr,"compute_nn_adaboost: out of memory\n");
    return 1;
  }

  if(!(enn->weights=dvector(nmodels))){
    fprintf(stderr,"compute_nn_adaboost: out of memory\n");
    return 1;
  }
  enn->k=k;
  enn->dist=dist;

  if(!(trx=(double **)calloc(n,sizeof(double*)))){
    fprintf(stderr,"compute_nn_adaboost: out of memory\n");
    return 1;
  }
  if(!(try=ivector(n))){
    fprintf(stderr,"compute_nn_adaboost: out of memory\n");
    return 1;
  }
  
  if(!(prob_copy=dvector(n))){
    fprintf(stderr,"compute_nn_adaboost: out of memory\n");
    return 1;
  }
  if(!(prob=dvector(n))){
    fprintf(stderr,"compute_nn_adaboost: out of memory\n");
    return 1;
  }

  if(!(pred=ivector(n))){
    fprintf(stderr,"compute_nn_adaboost: out of memory\n");
    return 1;
  }

  for(i =0;i<n;i++)
    prob[i]=1.0/(double)n;

  enn->nmodels=nmodels;
  sumalpha=0.0;
  for(b=0;b<nmodels;b++){

    for(i =0;i<n;i++)
      prob_copy[i]=prob[i];
    if(sample(n, prob_copy, n, &samples, TRUE,b)!=0){
      fprintf(stderr,"compute_nn_adaboost: sample error\n");
      return 1;
    }

    for(i =0;i<n;i++){
      trx[i] = x[samples[i]];
      try[i] = y[samples[i]];
    }
    
    if(compute_nn(&(enn->nn[b]),n,d,trx,try,k,dist)!=0){
      fprintf(stderr,"compute_nn_adaboost: compute_nn error\n");
      return 1;
    }
    free_ivector(samples);

    eps=0.0;
    for(i=0;i<n;i++){
      pred[i]=predict_nn(&(enn->nn[b]),x[i],&margin);
      if(pred[i] < -1 ){
	fprintf(stderr,"compute_nn_adaboost: predict_nn error\n");
	return 1;
      }
      if(pred[i]==0 || pred[i] != y[i])
	eps += prob[i];
      free_dvector(margin);
    }
    
    if(eps > 0 && eps < 0.5){
      enn->weights[b]=0.5 *log((1.0-eps)/eps);
      sumalpha+=enn->weights[b];
    }else{
      enn->nmodels=b;
      break;
    }
      
    sumprob=0.0;
    for(i=0;i<n;i++){
      prob[i]=prob[i]*exp(-enn->weights[b]*y[i]*pred[i]);
      sumprob+=prob[i];
    }

    if(sumprob <=0){
      fprintf(stderr,"compute_nn_adaboost: sumprob = 0\n");
      return 1;
    }
    for(i=0;i<n;i++)
      prob[i] /= sumprob;
    
  }
  
  if(enn->nmodels<=0){
    fprintf(stderr,"compute_nn_adaboost: no models produced\n");
    return 1;
  }

  if(sumalpha <=0){
      fprintf(stderr,"compute_nn_adaboost: sumalpha = 0\n");
      return 1;
  }
  for(b=0;b<enn->nmodels;b++)
    enn->weights[b] /= sumalpha;
  
  free(trx);
  free_ivector(try);
  free_ivector(pred);
  free_dvector(prob);
  free_dvector(prob_copy);
  return 0;

}

