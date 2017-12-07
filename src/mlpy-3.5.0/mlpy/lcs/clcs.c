/*  
    This code is written by Davide Albanese <davide.albanese@gmail.com>.
    (C) mlpy Developers.

    This program is free software: you can redistribute it and/or modify
    it underthe terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// Discovering Similar Multidimensional Trajectories


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "clcs.h"

#define MIN(a, b) ((a) <= (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))


int std(long *x, long *y, char **b, int n, int m)
{
  int i, j;
  int **c;
  int ret;
 
  c = (int **) malloc ((n+1) * sizeof(int *));
  for (i=0; i<=n; i++)
    {
      c[i] = (int *) malloc ((m+1) * sizeof(int));
      for (j=0; j<=m; j++)
	c[i][j] = 0;
    }

  for (i=1; i<=n; i++)
    for (j=1; j<=m; j++) 
      {
	if (x[i-1] == y[j-1])
	  {
	    c[i][j] = c[i-1][j-1] + 1;
	    b[i][j] = 0; // NW
	  }
	else if (c[i-1][j] >= c[i][j-1])
	  {
	    c[i][j] = c[i-1][j];
	    b[i][j] = 1; // N
	  }
	else 
	  {
	    c[i][j] = c[i][j-1];
	    b[i][j] = 2; // W
	  }
      }
 
  ret = c[n][m];

  for (i=0; i<=n; i++)
    free (c[i]);
  free(c);
  
  return ret;
}


int real(double *x, double *y, char **b, int n, int m, double eps, int delta)
{
  int i, j;
  int **c;
  int ret;
 
  c = (int **) malloc ((n+1) * sizeof(int *));
  for (i=0; i<=n; i++)
    {
      c[i] = (int *) malloc ((m+1) * sizeof(int));
      for (j=0; j<=m; j++)
	c[i][j] = 0;
    }

  for (i=1; i<=n; i++)
    for (j=1; j<=m; j++) 
      {
	if ((fabs(x[i-1] - y[j-1]) < eps) &&
	    (fabs((i-1) - (j-1)) <= delta))
	  {
	    c[i][j] = c[i-1][j-1] + 1;
	    b[i][j] = 0; // NW
	  }
	else if (c[i-1][j] >= c[i][j-1])
	  {
	    c[i][j] = c[i-1][j];
	    b[i][j] = 1; // N
	  }
	else 
	  {
	    c[i][j] = c[i][j-1];
	    b[i][j] = 2; // W
	  }
      }
 
  ret = c[n][m];

  for (i=0; i<=n; i++)
    free (c[i]);
  free(c);
  
  return ret;
}


void 
trace(char **b, int n, int m, Path *p)
{
  int i, j, k, z1, z2, d;
  int *px;
  int *py;
      
    
  // allocate path for the worst case
  d = MIN(n, m);
  px = (int *) malloc (d * sizeof(int));
  py = (int *) malloc (d * sizeof(int));

  i = n;
  j = m;
  k = 0;
  
  while ((i > 0) && (j > 0))
    {
      if (b[i][j] == 0)
	{
	  px[k] = i-1;
	  py[k] = j-1;
	  i--;
	  j--;
	  k++;
	}
      else if (b[i][j] == 1)
	i--;
      else
	j--;
    }
       
  p->px = (int *) malloc (k * sizeof(int));
  p->py = (int *) malloc (k * sizeof(int));
  for (z1=0, z2=k-1; z1<k; z1++, z2--)
    {
      p->px[z1] = px[z2];
      p->py[z1] = py[z2];
    }
  p->k = k;
  
  free(px);
  free(py);
}

