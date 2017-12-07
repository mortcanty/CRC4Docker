#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "ml.h"

int compute_rn(RegularizationNetwork *rn,int n,int d,
	       double *x[],double y[],double lambda,double sigma)
{
  int i,j;
  double **K,**inv_K;

  rn->n=n;
  rn->d=d;
  rn->x=x;
  rn->y=y;
  rn->lambda=lambda;
  rn->sigma=sigma;

  K=dmatrix(n,n);
  inv_K=dmatrix(n,n);

  for(i=0;i<n;i++){
    K[i][i]=n*lambda+trrbf_kernel(x[i],x[i],d,sigma);
    for(j=i+1;j<n;j++)
      K[i][j]=K[j][i]=trrbf_kernel(x[i],x[j],d,sigma);
  }

  if(inverse(K,inv_K,n) != 0){
    fprintf(stderr,"compute_rn:error inverting K\n");
    return 1;
  }

  free_dmatrix(K,n,n);
  
  rn->c=dvector(n);
  
  for(i=0;i<n;i++){
    rn->c[i]=0.0;
    
    for(j=0;j<n;j++)
      rn->c[i] += inv_K[i][j]*y[j]; 
  }
  
  free_dmatrix(inv_K,n,n);
  return 0;
}

double trrbf_kernel(double x1[], double x2[],int d,double sigma)
{
  int i;
  double norm=0.0;
  double tmp;

  for(i=0;i<d;i++){
    tmp=x1[i]-x2[i];
    norm += tmp*tmp;
  }

  return exp(-norm/sigma);
}
