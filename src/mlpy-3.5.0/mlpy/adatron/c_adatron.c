/*  
    This code is written by Davide Albanese <albanese@fbk.it>.
    (C) 2011 mlpy Developers.
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
#include <float.h>
#include <math.h>

#define MIN( A , B ) ((A) < (B) ? (A) : (B))
#define MAX( A , B ) ((A) > (B) ? (A) : (B))


int adatron(long *y, double *K, int n, double C, int maxsteps, double eps,
	    double *alpha, double *margin)
{
  double *z = NULL;
  double delta;
  int i, j, k;
  double zplus, zminus;
  int nplus, nminus;
  

  z = (double *) malloc (n * sizeof(double));
  
  for (i=0; i<maxsteps; i++)
    {
      for (j=0; j<n; j++)
	{
	  z[j] = 0.0;
	  for (k=0; k<n; k++)
	    /* z_j = sum(alpha_k y_k K_kj) */
	    z[j] += alpha[k] * y[k] * K[j + (k * n)];
	  delta = (1 - (y[j] * z[j])) / K[j + (j * n)];
	  alpha[j] = MIN(MAX(0, alpha[j]+delta), C);  
	}

      /* margin */
      zplus = DBL_MAX; zminus = -DBL_MAX;
      nplus = 0; nminus = 0;
      for (k=0; k<n; k++)
	{
	  if ((y[k]==+1) && (alpha[k]<C))
	    {
	      zplus = MIN(zplus, z[k]);
	      nplus++;
	    }
	  if ((y[k]==-1) && (alpha[k]<C))
	    {
	      zminus = MAX(zminus, z[k]);
	      nminus++;
	    }
	}

      if ((nplus == 0) || (nminus == 0))
	*margin = 0.0;
      else
	*margin = 0.5 * (zplus - zminus);

      if (fabs(1.0 - *margin) < eps)
	break;
    }
  
  free(z);
  return i;
}
