/*  
    This code is written by <albanese@fbk.it>.
    (C) 2010 mlpy Developers.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#define MIN( A , B ) ((A) < (B) ? (A) : (B))

#define INIT_STD 0
#define INIT_PLUSPLUS 1

void init_std(double *data,      /* data points (nn points x pp dimensions) */
	      double *means,     /* means (kk clusters x pp dimensions) */
	      int nn,            /* number od data points */
	      int pp,            /* number of dimensions */
	      int kk,            /* number of clusters */
	      unsigned long seed /* random seed for init */
	      )
{
  int n, p, k;
  int *ridx;
  const gsl_rng_type * T;
  gsl_rng * r;
  
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  gsl_rng_set (r, seed);

  ridx = (int *) malloc (nn * sizeof(int));
      
  for (n=0; n<nn; n++)
    ridx[n] = n;
          
  gsl_ran_shuffle (r, ridx, nn, sizeof (int));
  
  for (k=0; k<kk; k++)
    for (p=0; p<pp; p++)
      means[p + (k * pp)] = data[p + (ridx[k] * pp)];    
  
  free(ridx);
}


/* for init_plus */
void
dist_min(double *a, double *b, int nn)
{
  int n;
  
  for (n=0; n<nn; n++)
    a[n] = MIN (a[n], b[n]);
}


/* for init_plus */
int 
idx_max(double *a, int nn)
{
  int n, idx = 0;
  double max = -DBL_MAX;
  
  for (n=0; n<nn; n++)
    if (a[n] > max)
      {
	max = a[n];
	idx = n;
      }
  
  return idx;
}


void
init_plus(double *data,      /* data points (nn points x pp dimensions) */
	  double *means,     /* means (kk clusters x pp dimensions) */
	  int nn,            /* number od data points */
	  int pp,            /* number of dimensions */
	  int kk,            /* number of clusters */
	  unsigned long seed /* random seed for init */
	  )
{
  int n, p, k;
  double *dist, *distk;
  int sidx;

  const gsl_rng_type *T;
  gsl_rng *r;
 
  
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  gsl_rng_set (r, seed);
 
  dist = (double *) malloc (nn * sizeof(double));
  distk = (double *) malloc (nn * sizeof(double));
    
  /* first mean (randomly selected) */
  sidx = (int) gsl_rng_uniform_int (r, nn);
  gsl_rng_free(r);
  for (p=0; p<pp; p++)
    means[p] = data[p + (sidx * pp)];
  
  /* initialize dist */
  for (n=0; n<nn; n++) /* for each data point */
    dist[n] = DBL_MAX;

  for (k=0; k<kk-1; k++)
    {
      /* for each data point x compute distance from mean k */
      for (n=0; n<nn; n++)
	{
	  distk[n] = 0.0;
	  for (p=0; p<pp; p++)
	    distk[n] += pow(data[p + (n * pp)] - means[p + (k * pp)], 2);
	}
      
      /* for each data point x, compute the distance between x and */
      /* the nearest center that has already been chosen */
      dist_min (dist, distk, nn);
      
      /* add one new data point as a new center, using */           
      sidx = idx_max (dist, nn);

      for (p=0; p<pp; p++)
	means[p + ((k+1) * pp)] = data[p + (sidx * pp)];
    }
    
  free(dist);
  free(distk);
}


/* assignment step */
int
a_step(double *data,  /* data points (nn x pp) */
       double *means, /* means (kk x pp) */
       int *cls,      /* cluster assignement for each data point (nn) */
       int *nelems,   /* number of elements of each cluster (kk) */
       int nn,        /* number od data points */
       int pp,        /* number of dimensions */
       int kk         /* number of clusters */
       )
{
  int n, p, k, kn = 0;
  double dist, dmin;
  int changed = 0;
  
  for (k=0; k<kk; k++)
    nelems[k] = 0; 
   
  for (n=0; n<nn; n++) /* for each data point */
    {
      dmin = DBL_MAX;
      for (k=0; k<kk; k++) /* for each cluster */
	{
	  /* compute distance */
	  dist = 0.0;
	  for (p=0; p<pp; p++)
	    dist += pow(data[p + (n * pp)] - means[p + (k * pp)], 2);
	  
	  /* remember the cluster k if dist < dmin */
	  if (dist < dmin)
	    {
	      dmin = dist;
	      kn = k;
	    }    
	}
      
      
      /* if the cluster assignement change */
      if (kn != cls[n])
	changed++;
      
      /* update clusters and number of elements of each cluster */
      cls[n] = kn;
      nelems[kn]++;
    }
  
  return changed;
}


/* update step */
int
u_step(double *data,  /* data points (nn x pp) */
       double *means, /* means (kk x pp) */
       int *cls,      /* cluster assignement for each data point (nn) */
       int *nelems,   /* number of elements of each cluster (kk) */
       int nn,        /* number od data points */
       int pp,        /* number of dimensions */
       int kk         /* number of clusters */
       )
{
  int n, p, k;
  
  /* reset means */
  for (k=0; k<kk; k++)
    for (p=0; p<pp; p++)
      means[p + (k * pp)] = 0.0;

  for (n=0; n<nn; n++) /* for each data point */
    for (p=0; p<pp; p++) /* for each dimension */
      means[p + (cls[n] * pp)] += data[p + (n * pp)];
  
  for (k=0; k<kk; k++) 
    if (nelems[k] > 0)
      for (p=0; p<pp; p++)
	means[p + (k * pp)] /= nelems[k];
  
  return 1;
}


int
km(double *data,  /* data points (nn x pp) */
   double *means, /* initialized means (kk x pp) */
   int *cls,      /* cluster assignement for each data point (nn) */
   int nn,        /* number od data points */
   int pp,        /* number of dimensions */
   int kk         /* number of clusters */
   )
{
  int n;
  int ret, steps = 0;
  int changed = -1;
  int *nelems; /* number of elements of each cluster (kk) */

  nelems = (int *) malloc (kk * sizeof(int));

  /* init cls */
  for (n=0; n<nn; n++)
    cls[n] = -1.0;

  /* k-means algorithm  */
  while (changed != 0)
    {
      changed = a_step (data, means, cls, nelems, nn, pp, kk);     
      ret = u_step (data, means, cls, nelems, nn, pp, kk);
      steps++;
    }
  
  free(nelems);
 
  return steps;
}
