#include <stdio.h>
#include <stdlib.h>
#include "ml.h"

int  iunique(int y[], int n, int **values)
     /*
       extract unique values from a vector y of n integers.
       
       Return value: the number of unique values on success, 0 otherwise.
     */
{
  int nvalues=1;
  int i,j;
  int addclass;
  int *indx;

  if(!(*values=ivector(1))){
    fprintf(stderr,"iunique: out of memory\n");
    return 0;
  }
    
  (*values)[0]=y[0];
  for(i=1;i<n;i++){
    addclass=1;
    for(j=0;j<nvalues;j++)
      if((*values)[j]==y[i])
        addclass=0;
    if(addclass){
      if(!(*values=(int*)realloc(*values,(nvalues+1)*sizeof(int)))){
	fprintf(stderr,"iunique: out of memory\n");
	return 0;
      }
      (*values)[nvalues++]=y[i];
    }
  }

  if(!(indx=ivector(nvalues))){
    fprintf(stderr,"iunique: out of memory\n");
    return 0;
  }

  isort(*values,indx,nvalues,SORT_ASCENDING);

  if(free_ivector(indx)!=0){
    fprintf(stderr,"iunique: free_ivector error\n");
    return 0;
  }

  return nvalues;
}


int  dunique(double y[], int n, double **values)
     /*
       extract unique values from a vector y of n doubles.
       
       Return value: the number of unique values on success, 0 otherwise.
     */
{
  int nvalues=1;
  int i,j;
  int addclass;
  int *indx;

  if(!(*values=dvector(1))){
    fprintf(stderr,"dunique: out of memory\n");
    return 0;
  }
    
  (*values)[0]=y[0];
  for(i=1;i<n;i++){
    addclass=1;
    for(j=0;j<nvalues;j++)
      if((*values)[j]==y[i])
        addclass=0;
    if(addclass){
      if(!(*values=(double*)realloc(*values,(nvalues+1)*sizeof(double)))){
	fprintf(stderr,"dunique: out of memory\n");
	return 0;
      }
      (*values)[nvalues++]=y[i];
    }
  }

  if(!(indx=ivector(nvalues))){
    fprintf(stderr,"iunique: out of memory\n");
    return 0;
  }

  dsort(*values,indx,nvalues,SORT_ASCENDING);

  if(free_ivector(indx)!=0){
    fprintf(stderr,"iunique: free_ivector error\n");
    return 0;
  }

  return nvalues;
}
