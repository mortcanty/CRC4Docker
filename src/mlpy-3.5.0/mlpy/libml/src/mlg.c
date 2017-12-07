#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "ml.h"

static void compute_covar(double ***mat,MaximumLikelihood *ml,int class);
static void compute_mean(double ***mat,MaximumLikelihood *ml,int class);

static int compute_ml_bagging(EMaximumLikelihood *eml,int n,int d,double *x[],
			      int y[],int nmodels);
static int compute_ml_aggregate(EMaximumLikelihood *eml,int n,int d,
				double *x[],int y[],int nmodels);
static int compute_ml_adaboost(EMaximumLikelihood *eml,int n,int d,double *x[],
			      int y[],int nmodels);

int compute_ml(MaximumLikelihood *ml,int n,int d,double *x[],int y[])
     /*
       Compute ml model, given a matrix of examples x of dimension
       n x d. Classes of each example are contained in y.
       
       Return value: 0 on success, 1 otherwise.
     */
{
  double ***tmpMat;
  int *index;
  int i,j,k;

  
  ml->nclasses=iunique(y,n, &(ml->classes));

  if(ml->nclasses<=0){
    fprintf(stderr,"compute_ml: iunique error\n");
    return 1;
  }

  if(ml->nclasses==1){
    fprintf(stderr,"compute_ml: only 1 class recognized\n");
    return 1;
  }

  if(ml->nclasses==2)
    if(ml->classes[0] != -1 || ml->classes[1] != 1){
      fprintf(stderr,"compute_ml: for binary classification classes must be -1,1\n");
      return 1;
    }

  if(ml->nclasses>2)
    for(i=0;i<ml->nclasses;i++)
      if(ml->classes[i] != i+1){
        fprintf(stderr,"compute_ml: for %d-class classification classes must be 1,...,%d\n",ml->nclasses,ml->nclasses);
        return 1;
      }


  if(!(ml->npoints_for_class=ivector(ml->nclasses))){
    fprintf(stderr,"compute_ml: out of memory\n");
    return 1;
  }

  for(i=0;i<n;i++){
    for(j=0;j<ml->nclasses;j++){
      if(y[i] == ml->classes[j]){
	ml->npoints_for_class[j]+=1;
      }
    }
  }  

  ml->d = d;
  if(!(ml->priors = dvector(ml->nclasses))){
    fprintf(stderr,"compute_ml: out of memory\n");
    return 1;
  }
  if(!(ml->mean=dmatrix(ml->nclasses,ml->d))){
    fprintf(stderr,"compute_ml: out of memory\n");
    return 1;
  }
  if(!(ml->det = dvector(ml->nclasses))){
    fprintf(stderr,"compute_ml: out of memory\n");
    return 1;
  }
  if(!(ml->covar = (double ***) calloc(ml->nclasses, sizeof(double **)))){
    fprintf(stderr,"compute_ml: out of memory\n");
    return 1;
  }
  for(i=0;i<ml->nclasses;i++)
    if(!(ml->covar[i] = dmatrix(ml->d,ml->d))){
      fprintf(stderr,"compute_ml: out of memory\n");
      return 1;
    }
  
  if(!(tmpMat = (double ***) calloc(ml->nclasses, sizeof(double **)))){
    fprintf(stderr,"compute_ml: out of memory\n");
    return 1;
  }
  for(i=0;i<ml->nclasses;i++)
    if(!(tmpMat[i] = dmatrix(ml->npoints_for_class[i],ml->d))){
      fprintf(stderr,"compute_ml: out of memory\n");
      return 1;
    }
  

  if(!(index = ivector(ml->nclasses))){
    fprintf(stderr,"compute_ml: out of memory\n");
    return 1;
  }
  for(i=0;i<n;i++)
    for(j=0;j<ml->nclasses;j++)
      if(y[i]==ml->classes[j]){
	for(k=0;k<ml->d;k++)
	  tmpMat[j][index[j]][k] = x[i][k];
	index[j] += 1;
      }
  
  for(i=0;i<ml->nclasses;i++)
    compute_mean(tmpMat,ml,i);

  for(i=0;i<ml->nclasses;i++)
    compute_covar(tmpMat,ml,i);

  for(i=0;i<ml->nclasses;i++)
    ml->priors[i] = (double)ml->npoints_for_class[i] / (double)n;

  for(i=0;i<ml->nclasses;i++)
    free_dmatrix(tmpMat[i],ml->npoints_for_class[i],ml->d);
  /*
    for(j=0;j<ml->npoints_for_class[i];j++)
      free(tmpMat[i][j]);
  */

  if(!(ml->det = dvector(ml->nclasses))){
    fprintf(stderr,"compute_ml: out of memory\n");
    return 1;
  }
  if(!(ml->inv_covar = (double ***) calloc(ml->nclasses, sizeof(double **)))){
    fprintf(stderr,"compute_ml: out of memory\n");
    return 1;
  }
  for(i=0;i<ml->nclasses;i++)
    if(!(ml->inv_covar[i] = dmatrix(ml->d,ml->d))){
      fprintf(stderr,"compute_ml: out of memory\n");
      return 1;   
    }
  
  for(j=0;j<ml->nclasses;j++){
    ml->det[j] = determinant(ml->covar[j],ml->d);
    if(inverse(ml->covar[j],ml->inv_covar[j],ml->d)!=0){
      fprintf(stderr,"compute_ml: error computing inverse covariance matrix of class %d\n",ml->classes[j]);
    }
  }
  
  free(tmpMat);
  free_ivector(index);
  
  return 0;
}

int compute_eml(EMaximumLikelihood *eml,int n,int d,double *x[],
		 int y[],int method,int nmodels)
    /*
       compute ensamble of ml models.x,y,n,d are the input data.
       method is one of BAGGING,AGGREGATE,ADABOOST.

       Return value: 0 on success, 1 otherwise.
     */
{
  switch(method){
  case BAGGING:
    if(compute_ml_bagging(eml,n,d,x,y,nmodels) != 0){
      fprintf(stderr,"compute_eml: compute_ml_bagging error\n");
      return 1;
    }
    break;
  case AGGREGATE:
    if(compute_ml_aggregate(eml,n,d,x,y,nmodels) != 0){
      fprintf(stderr,"compute_eml: compute_ml_aggregate error\n");
      return 1;
    }
    break;
  case ADABOOST:
    if( compute_ml_adaboost(eml,n,d,x,y,nmodels) != 0){
      fprintf(stderr,"compute_eml: compute_ml_adaboost error\n");
      return 1;
    }
    break;
  default:
    fprintf(stderr,"compute_eml: ensamble method not recognized\n");
    return 1;
    break;
  }

  return 0;
}

int predict_ml(MaximumLikelihood *ml, double x[],double **margin)
     /*
       predicts ml model on a test point x. Posteriors probability
       for each class will be stored within the array margin
       (an array of length ml->nclasses).

       Return value: the predicted value on success (-1 or 1 for
       binary classification; 1,...,nclasses in the multiclass case),
       0 on succes with non unique classification, -2 otherwise.

     */
{

  int i,j,c;
  double *tmpVect;
  double *distmean;
  double delta;
  double max_posterior;
  int max_posterior_index;
  double sum_posteriors;

  if(!(tmpVect = dvector(ml->d))){
    fprintf(stderr,"predict_ml: out of memory\n");
    return -2;
  }
  if(!(distmean= dvector(ml->d))){
    fprintf(stderr,"predict_ml: out of memory\n");
    return -2;
  }
  if(!((*margin)= dvector(ml->nclasses))){
    fprintf(stderr,"predict_ml: out of memory\n");
    return -2;  
  }

  for(c=0;c<ml->nclasses;c++){
    for(i=0;i<ml->d;i++)
      distmean[i] = x[i] - ml->mean[c][i];

    for(i=0;i<ml->d;i++)
      tmpVect[i] = 0.0;

    for(i=0;i<ml->d;i++)
      for(j=0;j<ml->d;j++)
	tmpVect[i] += distmean[j] * ml->inv_covar[c][j][i];
    
    delta=0.0;
    for(i=0;i<ml->d;i++)
       delta += tmpVect[i] * distmean[i];

    if(ml->det[c] > 0.0){
      (*margin)[c] = exp(-0.5 * delta)/ sqrt(ml->det[c]);
    }else{
      fprintf(stderr, "predict_ml:  det. of cov. matrix of class %d = 0\n",c);
      return -2;
    }
    (*margin)[c] = (*margin)[c] * ml->priors[c];
  }

  max_posterior = 0.0;
  max_posterior_index =0;
  sum_posteriors = 0.0;
  for(c=0;c<ml->nclasses;c++){
    sum_posteriors += (*margin)[c];
    if((*margin)[c] > max_posterior){
      max_posterior = (*margin)[c];
      max_posterior_index = c;
    }
  }
  for(c=0;c<ml->nclasses;c++)
    (*margin)[c] /= sum_posteriors;
  
  free_dvector(tmpVect);
  free_dvector(distmean);

  return ml->classes[max_posterior_index];

}




int predict_eml(EMaximumLikelihood *eml, double x[],double **margin)
     /*
       predicts ml model on a test point x. Proportions of neighbours
       for each class will be stored within the array margin 
       (an array of length ml->nclasses). 

       
       Return value: the predicted value on success (-1 or 1 for
       binary classification; 1,...,nclasses in the multiclass case),
       0 on succes with non unique classification, -2 otherwise.
     */
{
  int i,b;
  int pred;
  double *tmpmargin;
  double maxmargin;
  
  if(!((*margin)=dvector(eml->nclasses))){
    fprintf(stderr,"predict_eml: out of memory\n");
    return -2;
  }
  
  if(eml->nclasses==2){
    for(b=0;b<eml->nmodels;b++){
      pred=predict_ml(&(eml->ml[b]), x,&tmpmargin);
      if(pred < -1){
	fprintf(stderr,"predict_eml: predict_ml error\n");
	return -2;
      }
      if(pred==-1)
	(*margin)[0] += eml->weights[b];
      else if(pred==1)
	(*margin)[1] += eml->weights[b];
      
      free_dvector(tmpmargin);
    }

    if((*margin)[0] > (*margin)[1])
      return -1;
    else if((*margin)[0] < (*margin)[1])
      return 1;
    else
      return 0;
  }else{
    for(b=0;b<eml->nmodels;b++){
      pred=predict_ml(&(eml->ml[b]), x,&tmpmargin);
      if(pred < -1){
	fprintf(stderr,"predict_eml: predict_ml error\n");
	return -2;
      }
      
      if(pred>0)
	(*margin)[pred-1] += eml->weights[b];
      
      free_dvector(tmpmargin);
    }

    maxmargin=0.0;
    pred=0;
    for(i=0;i<eml->nclasses;i++)
      if((*margin)[i]>maxmargin){
	maxmargin=(*margin)[i];
	pred=i;
      }

    for(i=0;i<eml->nclasses;i++)
      if(i != pred)
	if((*margin)[i] == maxmargin)
	  return 0;

    return pred+1;
  }

  return -2;
  
}


static int compute_ml_bagging(EMaximumLikelihood *eml,int n,int d,double *x[],
			      int y[],int nmodels)
{
  int i,b;
  int *samples;
  double **trx;
  int *try;

  if(nmodels<1){
    fprintf(stderr,"compute_ml_bagging: nmodels must be greater than 0\n");
    return 1;
  }

  eml->nclasses=iunique(y,n, &(eml->classes));

  if(eml->nclasses<=0){
    fprintf(stderr,"compute_ml_bagging: iunique error\n");
    return 1;
  }
  if(eml->nclasses==1){
    fprintf(stderr,"compute_ml_bagging: only 1 class recognized\n");
    return 1;
  }

  if(eml->nclasses==2)
    if(eml->classes[0] != -1 || eml->classes[1] != 1){
      fprintf(stderr,"compute_ml_bagging: for binary classification classes must be -1,1\n");
      return 1;
    }
  
  if(eml->nclasses>2)
    for(i=0;i<eml->nclasses;i++)
      if(eml->classes[i] != i+1){
	fprintf(stderr,"compute_ml_bagging: for %d-class classification classes must be 1,...,%d\n",eml->nclasses,eml->nclasses);
	return 1;
      }

  if(!(eml->ml=(MaximumLikelihood *)calloc(nmodels,sizeof(MaximumLikelihood)))){
    fprintf(stderr,"compute_ml_bagging: out of memory\n");
    return 1;
  }
  eml->nmodels=nmodels;
  if(!(eml->weights=dvector(nmodels))){
    fprintf(stderr,"compute_ml_bagging: out of memory\n");
    return 1;
  }

  for(b=0;b<nmodels;b++)
    eml->weights[b]=1.0 / (double) nmodels;
  
  if(!(trx=(double **)calloc(n,sizeof(double*)))){
    fprintf(stderr,"compute_ml_bagging: out of memory\n");
    return 1;
  }
  if(!(try=ivector(n))){
    fprintf(stderr,"compute_ml_bagging: out of memory\n");
    return 1;
  }
  
  for(b=0;b<nmodels;b++){
    if(sample(n, NULL, n, &samples, TRUE,b)!=0){
       fprintf(stderr,"compute_ml_bagging: sample error\n");
       return 1;
    }

    for(i =0;i<n;i++){
      trx[i] = x[samples[i]];
      try[i] = y[samples[i]];
    }

    if(compute_ml(&(eml->ml[b]),n,d,trx,try)!=0){
      fprintf(stderr,"compute_ml_bagging: compute_ml error\n");
      return 1;
    }
    free_ivector(samples);

  }

  free(trx);
  free_ivector(try);
    
  return 0;

}



static int compute_ml_aggregate(EMaximumLikelihood *eml,int n,int d,
				double *x[], int y[],int nmodels)
{
  int i,b;
  int *samples;
  double **trx;
  int *try;
  int indx;

  if(nmodels<1){
    fprintf(stderr,"compute_ml_aggregate: nmodels must be greater than 0\n");
    return 1;
  }

  if(nmodels > n){
    fprintf(stderr,"compute_ml_aggregate: nmodels must be less than n\n");
    return 1;
  }

  eml->nclasses=iunique(y,n, &(eml->classes));

  if(eml->nclasses<=0){
    fprintf(stderr,"compute_ml_aggregate: iunique error\n");
    return 1;
  }
  if(eml->nclasses==1){
    fprintf(stderr,"compute_ml_aggregate: only 1 class recognized\n");
    return 1;
  }

  if(eml->nclasses==2)
    if(eml->classes[0] != -1 || eml->classes[1] != 1){
      fprintf(stderr,"compute_ml_aggregate: for binary classification classes must be -1,1\n");
      return 1;
    }
  
  if(eml->nclasses>2)
    for(i=0;i<eml->nclasses;i++)
      if(eml->classes[i] != i+1){
	fprintf(stderr,"compute_ml_aggregate: for %d-class classification classes must be 1,...,%d\n",eml->nclasses,eml->nclasses);
	return 1;
      }

  if(!(eml->ml=(MaximumLikelihood *)calloc(nmodels,sizeof(MaximumLikelihood)))){
    fprintf(stderr,"compute_ml_aggregate: out of memory\n");
    return 1;
  }
  eml->nmodels=nmodels;
  if(!(eml->weights=dvector(nmodels))){
    fprintf(stderr,"compute_ml_aggregate: out of memory\n");
    return 1;
  }

  for(b=0;b<nmodels;b++)
    eml->weights[b]=1.0 / (double) nmodels;
  
  if(!(trx=(double **)calloc(n,sizeof(double*)))){
    fprintf(stderr,"compute_ml_aggregate: out of memory\n");
    return 1;
  }
  if(!(try=ivector(n))){
    fprintf(stderr,"compute_ml_aggregate: out of memory\n");
    return 1;
  }
  
  if(sample(nmodels, NULL, n, &samples, TRUE,0)!=0){
    fprintf(stderr,"compute_ml_aggregate: sample error\n");
    return 1;
  }

  for(b=0;b<nmodels;b++){
  
    indx=0;
    for(i=0;i<n;i++)
      if(samples[i] == b){
	trx[indx] = x[i];
	try[indx++] = y[i];
      }

    if(compute_ml(&(eml->ml[b]),indx,d,trx,try)!=0){
      fprintf(stderr,"compute_ml_aggregate: compute_ml error\n");
      return 1;
    }

  }

  free_ivector(samples);
  free(trx);
  free_ivector(try);
    
  return 0;

}

static int compute_ml_adaboost(EMaximumLikelihood *eml,int n,int d,double *x[],
			       int y[],int nmodels)
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
    fprintf(stderr,"compute_ml_adaboost: nmodels must be greater than 0\n");
    return 1;
  }

  eml->nclasses=iunique(y,n, &(eml->classes));

  if(eml->nclasses<=0){
    fprintf(stderr,"compute_ml_adaboost: iunique error\n");
    return 1;
  }
  if(eml->nclasses==1){
    fprintf(stderr,"compute_ml_adaboost: only 1 class recognized\n");
    return 1;
  }

  if(eml->nclasses==2)
    if(eml->classes[0] != -1 || eml->classes[1] != 1){
      fprintf(stderr,"compute_ml_adaboost: for binary classification classes must be -1,1\n");
      return 1;
    }
  
  if(eml->nclasses>2){
    fprintf(stderr,"compute_ml_adaboost: multiclass classification not allowed\n");
    return 1;
  }

  if(!(eml->ml=(MaximumLikelihood *)calloc(nmodels,sizeof(MaximumLikelihood)))){
    fprintf(stderr,"compute_ml_adaboost: out of memory\n");
    return 1;
  }

  if(!(eml->weights=dvector(nmodels))){
    fprintf(stderr,"compute_ml_adaboost: out of memory\n");
    return 1;
  }

  if(!(trx=(double **)calloc(n,sizeof(double*)))){
    fprintf(stderr,"compute_ml_adaboost: out of memory\n");
    return 1;
  }
  if(!(try=ivector(n))){
    fprintf(stderr,"compute_ml_adaboost: out of memory\n");
    return 1;
  }
  
  if(!(prob_copy=dvector(n))){
    fprintf(stderr,"compute_ml_adaboost: out of memory\n");
    return 1;
  }
  if(!(prob=dvector(n))){
    fprintf(stderr,"compute_ml_adaboost: out of memory\n");
    return 1;
  }

  if(!(pred=ivector(n))){
    fprintf(stderr,"compute_ml_adaboost: out of memory\n");
    return 1;
  }

  for(i =0;i<n;i++)
    prob[i]=1.0/(double)n;

  eml->nmodels=nmodels;
  sumalpha=0.0;
  for(b=0;b<nmodels;b++){

    for(i =0;i<n;i++)
      prob_copy[i]=prob[i];
    if(sample(n, prob_copy, n, &samples, TRUE,b)!=0){
      fprintf(stderr,"compute_ml_adaboost: sample error\n");
      return 1;
    }

    for(i =0;i<n;i++){
      trx[i] = x[samples[i]];
      try[i] = y[samples[i]];
    }
    
    if(compute_ml(&(eml->ml[b]),n,d,trx,try)!=0){
      fprintf(stderr,"compute_ml_adaboost: compute_ml error\n");
      return 1;
    }
    free_ivector(samples);

    eps=0.0;
    for(i=0;i<n;i++){
      pred[i]=predict_ml(&(eml->ml[b]),x[i],&margin);
      if(pred[i] < -1 ){
	fprintf(stderr,"compute_ml_adaboost: predict_ml error\n");
	return 1;
      }
      if(pred[i]==0 || pred[i] != y[i])
	eps += prob[i];
      free_dvector(margin);
    }
    
    if(eps > 0 && eps < 0.5){
      eml->weights[b]=0.5 *log((1.0-eps)/eps);
      sumalpha+=eml->weights[b];
    }else{
      eml->nmodels=b;
      break;
    }
      
    sumprob=0.0;
    for(i=0;i<n;i++){
      prob[i]=prob[i]*exp(-eml->weights[b]*y[i]*pred[i]);
      sumprob+=prob[i];
    }

    if(sumprob <=0){
      fprintf(stderr,"compute_ml_adaboost: sumprob = 0\n");
      return 1;
    }
    for(i=0;i<n;i++)
      prob[i] /= sumprob;
    
  }
  
  if(eml->nmodels<=0){
    fprintf(stderr,"compute_ml_adaboost: no models produced\n");
    return 1;
  }

  if(sumalpha <=0){
      fprintf(stderr,"compute_ml_adaboost: sumalpha = 0\n");
      return 1;
  }
  for(b=0;b<eml->nmodels;b++)
    eml->weights[b] /= sumalpha;
  
  free(trx);
  free_ivector(try);
  free_ivector(pred);
  free_dvector(prob);
  free_dvector(prob_copy);
  return 0;

}


static void compute_covar(double ***mat,MaximumLikelihood *ml,int class)
{
  int i, j, k;
  
  for(i = 0; i < ml->d; i++)
    for(j = i; j < ml->d; j++){
      for(k = 0; k < ml->npoints_for_class[class]; k++){
	ml->covar[class][i][j] += (mat[class][k][i] - ml->mean[class][i]) * 
	  (mat[class][k][j] - ml->mean[class][j]);
      }
      ml->covar[class][j][i] = ml->covar[class][i][j];
    }
  for(i = 0; i < ml->d; i++)
    for(j = 0; j < ml->d; j++)
      ml->covar[class][i][j] /= ((double)ml->npoints_for_class[class] - 1.);
}

static void compute_mean(double ***mat,MaximumLikelihood *ml,int class) 
{
  int i,j;
  
  for(i=0;i < ml->d;i++)
    for(j=0;j< ml->npoints_for_class[class];j++)
      ml->mean[class][i] += mat[class][j][i];

  for(i=0;i < ml->d;i++)
    ml->mean[class][i] /= ml->npoints_for_class[class];
}


