#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "ml.h"

static int compute_tr(TerminatedRampsRegularizationNetwork *trrn, double *K[],
		      int k_nn);


int compute_trrn(TerminatedRampsRegularizationNetwork *trrn,int n,int d,
                 double *x[],double y[],double lambda,int k_nn)
{
  int i,j;
  double **K,**inv_K;

  trrn->n=n;
  trrn->d=d;
  trrn->x=x;
  trrn->y=y;
  trrn->lambda=lambda;

  

  K=dmatrix(n,n);

  compute_tr(trrn,K,k_nn);


  for(i=0;i<n;i++)
    K[i][i] += n * lambda;

  inv_K=dmatrix(n,n);

  if(inverse(K,inv_K,n) != 0){
    fprintf(stderr,"compute_rn:error inverting K\n");
    return 1;
  }

  free_dmatrix(K,n,n);

  trrn->c=dvector(n);
  
  for(i=0;i<n;i++){
    trrn->c[i]=0.0;

    for(j=0;j<n;j++)
      trrn->c[i] += inv_K[i][j]*y[j];

  }

  free_dmatrix(inv_K,n,n);
  return 0;
}

double tr_kernel(double x1[], double x2[],
		 TerminatedRampsRegularizationNetwork *trrn)
{
  int t,j;
  double k_t_x1_x2;
  double f_t_x1,f_t_x2;

  k_t_x1_x2=0.0;
  for(t=0;t<trrn->tr.ntr;t++){
    f_t_x1=0.0;
    for(j=0;j<trrn->d;j++)
      f_t_x1 += trrn->tr.w[t][j]*x1[j];
    f_t_x1 += trrn->tr.b[t];

    if(f_t_x1 > trrn->tr.y_max[t])
      f_t_x1=trrn->tr.y_max[t];
    else if(f_t_x1 < trrn->tr.y_min[t])
      f_t_x1=trrn->tr.y_min[t];

    f_t_x2=0.0;
    for(j=0;j<trrn->d;j++)
      f_t_x2 += trrn->tr.w[t][j]*x2[j];
    f_t_x2 += trrn->tr.b[t];

    if(f_t_x2 > trrn->tr.y_max[t])
      f_t_x2=trrn->tr.y_max[t];
    else if(f_t_x2 < trrn->tr.y_min[t])
      f_t_x2=trrn->tr.y_min[t];

    k_t_x1_x2 += f_t_x1 * f_t_x2;
  }

  return k_t_x1_x2;
}


static int compute_tr(TerminatedRampsRegularizationNetwork *trrn, double *K[],
		      int k_nn)
{
  int i,j;
  int t;
  int k;
  int p,q;
  double xi_xj,xi_xi,xj_xj;
  double *f_t;
  double *dist_who;
  int *sort_index;
  int indx;


  trrn->tr.w=dmatrix(1,trrn->d);
  trrn->tr.alpha=dvector(1);
  trrn->tr.b=dvector(1);
  trrn->tr.i=ivector(1);
  trrn->tr.j=ivector(1);
  trrn->tr.y_min=dvector(1);
  trrn->tr.y_max=dvector(1);


  t=0;

  dist_who=dvector(trrn->n);
  sort_index=ivector(trrn->n);

  if(k_nn==0 || k_nn>trrn->n)
    k_nn=trrn->n;

  for(i=0;i<trrn->n;i++){
    fprintf(stderr,"%d\b\b\b\b\b",i);

    for(j=0;j<trrn->n;j++)
      dist_who[j]=euclidean_squared_distance(trrn->x[i],trrn->x[j],trrn->d);
    
    
    for(j=0;j<trrn->n;j++)
      sort_index[j]=j;
    dsort(dist_who,sort_index,trrn->n,SORT_ASCENDING);

    for(j=0;j<k_nn;j++){

      indx=sort_index[j];


      if(indx != i && trrn->y[indx] != trrn->y[i]){
	
	xi_xj=0.0;
	for(k=0;k<trrn->d;k++)
	  xi_xj += trrn->x[i][k] * trrn->x[indx][k];
	xi_xi=0.0;
	for(k=0;k<trrn->d;k++)
	  xi_xi += trrn->x[i][k] * trrn->x[i][k];
	xj_xj=0.0;
	for(k=0;k<trrn->d;k++)
	  xj_xj += trrn->x[indx][k] * trrn->x[indx][k];
	
	trrn->tr.alpha[t] = (trrn->y[indx] - trrn->y[i])/
	  (trrn->y[indx] * xj_xj -  trrn->y[i] * xi_xi - 
	   (trrn->y[indx] - trrn->y[i]) * xi_xj);
	
	trrn->tr.b[t] = trrn->y[i] - trrn->tr.alpha[t] * 
	  ( trrn->y[i] * xi_xi + trrn->y[indx] * xi_xj);
	
	for(k=0;k<trrn->d;k++)
	  trrn->tr.w[t][k] = trrn->tr.alpha[t] * 
	    ( trrn->y[i] * trrn->x[i][k] + trrn->y[indx] * trrn->x[indx][k]);
	
	trrn->tr.i[t]=i;
	trrn->tr.j[t]=indx;
	trrn->tr.y_min[t]= (trrn->y[i] > trrn->y[indx]) ? 
	  trrn->y[indx] : trrn->y[i];
	trrn->tr.y_max[t]= (trrn->y[i] < trrn->y[indx]) ? 
	  trrn->y[indx] : trrn->y[i];
	
	
	f_t=dvector(trrn->n);
	for(p=0;p<trrn->n;p++){
	  f_t[p] = 0.0;
	  for(k=0;k<trrn->d;k++)
	    f_t[p] += trrn->tr.w[t][k] * trrn->x[p][k];
	  f_t[p] += trrn->tr.b[t];
	  
	  if(f_t[p]> trrn->tr.y_max[t])
	    f_t[p] =  trrn->tr.y_max[t];
	  else if(f_t[p] < trrn->tr.y_min[t])
	    f_t[p] =  trrn->tr.y_min[t];
	  
	}
	
	
	for(p=0;p<trrn->n;p++)
	  for(q=p;q<trrn->n;q++)
	    K[p][q] += f_t[p] * f_t[q];
	
	free_dvector(f_t);
	t++;
	

	trrn->tr.w=(double **) realloc(trrn->tr.w, (t+1) * sizeof(double *));
	trrn->tr.w[t]=dvector(trrn->d);
	trrn->tr.alpha=(double *) realloc(trrn->tr.alpha,(t+1)*sizeof(double));
	trrn->tr.b=(double *) realloc(trrn->tr.b, (t+1) * sizeof(double));
	trrn->tr.i=(int *) realloc(trrn->tr.i, (t+1) * sizeof(int));
	trrn->tr.j=(int *) realloc(trrn->tr.j, (t+1) * sizeof(int));
	trrn->tr.y_min=(double *) realloc(trrn->tr.y_min,(t+1)*sizeof(double));
	trrn->tr.y_max=(double *) realloc(trrn->tr.y_max,(t+1)*sizeof(double));
	
      }
    }
  }

  for(p=0;p<trrn->n;p++)
    for(q=0;q<p;q++)
      K[p][q] = K[q][p];

  trrn->tr.ntr=t;

  return 0;
}
