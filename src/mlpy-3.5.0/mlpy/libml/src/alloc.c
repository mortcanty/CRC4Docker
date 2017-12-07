#include <stdlib.h>
#include <stdio.h>

int *ivector(long n)
     /*
       Allocates memory for an array of n integers.

       Return value: a pointer to the allocated  memory or
       NULL if the request fails
     */
       {
  int *v;
  
  if(n<1){
    fprintf(stderr,"ivector: parameter n must be > 0\n");
    return NULL;
  }
  
  if(!(v=(int *)calloc(n,sizeof(int))))
    fprintf(stderr,"ivector: out of memory\n");

  return v;
}

double *dvector(long n)
     /*
       Allocates memory for an array of n doubles
       
       Return value: a pointer to the allocated  memory or
       NULL if the request fails
     */
{
  double *v;

  if(n<1){
    fprintf(stderr,"dvector: parameter n must be > 0\n");
    return NULL;
  }
  
  if (!(v=(double *)calloc(n,sizeof(double)))) 
    fprintf(stderr,"dvector: out of memory\n");

  return v;
}

double **dmatrix(long n, long m)
     /*
       Allocates memory for a matrix of n x m doubles
       
       Return value: a pointer to the allocated  memory or
       NULL if the request fails
     */
{
  double **M;
  int i;

  if(n<1 || m<1){
    fprintf(stderr,"dmatrix: parameters n and m must be > 0\n");
    return NULL;
  }
  
  if(!(M=(double **)calloc(n,sizeof(double*)))){
    fprintf(stderr,"dmatrix: out of memory");
    return NULL;
  }
  
  for(i=0;i<n;i++)
    if(!(M[i]=(double*)dvector(m))){
      fprintf(stderr,"dmatrix: error allocating memory for M[%d]\n",i);
      return NULL;
    }

  if (!M) 
    fprintf(stderr,"dmatrix: out of memory\n");

  return M;
}

int **imatrix(long n, long m)
     /*
       Allocates memory for a matrix of n x m integers
       
       Return value: a pointer to the allocated  memory or
       NULL if the request fails
     */
{
  int **M;
  int i;

  if(n<1 || m<1){
    fprintf(stderr,"imatrix: parameters n and m must be > 0\n");
    return NULL;
  }

  if(!(M=(int **)calloc(n,sizeof(int*)))){
    fprintf(stderr,"imatrix: out of memory\n");
    return NULL;
  }
  
  for(i=0;i<n;i++)
    if(!(M[i]=(int*)ivector(m))){
      fprintf(stderr,"imatrix: error allocating memory for M[%d]\n",i);
      return NULL;
    }

  if (!M) 
    fprintf(stderr,"imatrix: out of memory\n");

  return M;
}

int free_ivector(int *v)
     /*
       Frees the memory space pointed to by v

       Return value: 1 if v is NULL, 0 otherwise
     */
{
  if(!v){
    fprintf(stderr,"free_ivector: pointer v empty\n");
    return 1;
  }

  free(v);
  return 0;
}

int free_dvector(double *v)
     /*
       Frees the memory space pointed to by v

       Return value: 1 if v is NULL, 0 otherwise
     */
{
  if(!v){
    fprintf(stderr,"free_dvector: pointer v empty\n");
    return 1;
  }

  free(v);
  return 0;
}


int free_dmatrix(double **M, long n, long m)
     /*
       Frees the memory space pointed to by a n xm matrix of doubles

       Return value: a positive integer if an error occurred, 0 otherwise
     */
{
  int i;
  
  if(n<1 || m<1){
    fprintf(stderr,"free_dmatrix: parameters n and m must be > 0\n");
    return 1;
  }

  if(!M){
    fprintf(stderr,"free_dmatrix: pointer M empty\n");
    return 2;
  }

  for(i=0;i<n;i++){
    if(!M[i]){
      fprintf(stderr,"free_dmatrix: pointer M[%d] empty\n",i);
      return 3;
    }
    free(M[i]);
  }

  free(M);
  
  return 0;
}


int free_imatrix(int **M, long n, long m)
     /*
       Frees the memory space pointed to by a n xm matrix of integers

       Return value: a positive integer if an error occurred, 0 otherwise
     */
{
  int i;
  
  if(n<1 || m<1){
    fprintf(stderr,"free_imatrix: parameters n and m must be > 0\n");
    return 1;
  }

  if(!M){
    fprintf(stderr,"free_imatrix: pointer M empty\n");
    return 2;
  }

  for(i=0;i<n;i++){
    if(!M[i]){
      fprintf(stderr,"free_imatrix: pointer M[%d] empty\n",i);
      return 3;
    }
    free(M[i]);
  }
  
  free(M);

  return 0;
}


