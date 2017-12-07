#include  <stdio.h>
#include  <stdlib.h>
#include "ml.h"

/* Equal probability sampling; with-replacement case */
static void SampleReplace(int k, int n, int *y,int seed)
{
    int i;

    srand48(seed);

    for (i = 0; i < k; i++)
	y[i] = n * drand48();
}

/* Equal probability sampling; without-replacement case */
static void SampleNoReplace(int k, int n, int *y, int *x,int seed)
{
    int i, j;

    srand48(seed);

    for (i = 0; i < n; i++)
	x[i] = i;
    for (i = 0; i < k; i++) {
	j = n * drand48();
	y[i] = x[j];
	x[j] = x[--n];
    }
}

/* Unequal probability sampling; with-replacement case */
static void ProbSampleReplace(int n, double *p, int *perm, int nans, int *ans,
			      int seed)
{
    double rU;
    int i, j;
    int nm1 = n - 1;

    srand48(seed);

    /* record element identities */
    for (i = 0; i < n; i++)
	perm[i] = i;

    /* sort the probabilities into descending order */
    dsort(p, perm, n,SORT_DESCENDING);

    /* compute cumulative probabilities */
    for (i = 1 ; i < n; i++)
	p[i] += p[i - 1];

    /* compute the sample */
    for (i = 0; i < nans; i++) {
	rU = drand48();
	for (j = 0; j < nm1; j++) {
	    if (rU <= p[j])
		break;
	}
	ans[i] = perm[j];
    }
}

/* Unequal probability sampling; without-replacement case */
static void ProbSampleNoReplace(int n, double *p, int *perm,
				int nans, int *ans,int seed)
{
    double rT, mass, totalmass;
    int i, j, k, n1;

    srand48(seed);

    /* Record element identities */
    for (i = 0; i < n; i++)
	perm[i] = i;

    /* Sort probabilities into descending order */
    /* Order element identities in parallel */
    dsort(p, perm, n,SORT_DESCENDING);

    /* Compute the sample */
    totalmass = 1;
    for (i = 0, n1 = n-1; i < nans; i++, n1--) {
	rT = totalmass * drand48();
	mass = 0;
	for (j = 0; j < n1; j++) {
	    mass += p[j];
	    if (rT <= mass)
		break;
	}
	ans[i] = perm[j];
	totalmass -= p[j];
	for(k = j; k < n1; k++) {
	    p[k] = p[k + 1];
	    perm[k] = perm[k + 1];
	}
    }
}

int sample(int n, double prob[], int nsamples, int **samples, int replace,
	    int seed)
     /*
       Extract nsamples sampling from 0,...,n-1.
       If prob is NULL equal probability sampling is implemented, otherwise
       prob will be used for sampling with unequal probability.
       If replace = TRUE (=1), with-replacement case will be considered,
       if replace = FALSE (=0), without-replacement case will be considered,
       Samples are stored into the array *samples.

        Return value: 0 on success, 1 otherwise.
     */
{
  int *x;

  if(!((*samples)=ivector(nsamples))){
    fprintf(stderr,"sample: out of memory\n");
    return 1;
  }


  if(!prob){
    if(replace)
      SampleReplace(nsamples, n, *samples,seed);
    else{
      if(nsamples>n){
	fprintf(stderr,"sample: nsamples must be <= n\n");
	return 1;
      }

      if(!(x=ivector(n))){
	fprintf(stderr,"sample: out of memory\n");
	return 1;
      }

      SampleNoReplace(nsamples,n, *samples, x,seed);

      if(free_ivector(x)!=0){
	fprintf(stderr,"sample: free_ivector error\n");
	return 1;
      }
    }
  }else{
    if(!(x=ivector(n))){
	fprintf(stderr,"sample: out of memory\n");
	return 1;
      }

    if(replace)
      ProbSampleReplace(n, prob, x, nsamples, *samples,seed);
    else{
      if(nsamples>n){
	fprintf(stderr,"sample: nsamples must be <= n\n");
	return 1;
      }

      ProbSampleNoReplace(n, prob, x,nsamples, *samples,seed);
    }

    if(free_ivector(x)!=0){
      fprintf(stderr,"sample: free_ivector error\n");
      return 1;
    }
  }
  
  return 0;

}
