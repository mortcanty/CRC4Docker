/*  
    This code is written by Davide Albanese <davide.albanese@gmail.com> and
    Giuseppe Jurman <jurman@fbk.eu>.
    (C) 2011 mlpy Developers

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

/* canberra distance */
double c_canberra(double *x, double *y, long n)
{
  double den;
  double d = 0.0;
  long i;

  for (i=0; i<n; i++)
    {
      den = fabs(x[i]) + fabs(y[i]);
      if (den != 0)
	d += fabs(x[i] - y[i]) / den;
    }
  return d;
}

/* canberra distance with location parameter*/
double c_canberra_location(long *x, long *y, long n, long k)
{
  double d = 0.0;
  double xx, yy;
  long i;
  

  for (i=0; i<n; i++)
    {
      xx = ((x[i] + 1) < k+1) ? (x[i] + 1) : k+1;
      yy = ((y[i] + 1) < k+1) ? (y[i] + 1) : k+1;
      d += fabs(xx - yy) / (xx + yy);
    }

  return d;
}


/* The expected value of the canberra location */
double harm(long n)
{
  double h = 0.0;
  long i;
  
  for(i=1; i<=n; i++)
    h += 1.0 / (double) i;
  
  return h;
}

double e_harm(long n)
{
  return 0.5 * harm(floor((double) n / 2.0));
}

double o_harm(long n)
{
  return harm(n) - 0.5 * harm(floor((double) n / 2.0));
}

double a_harm(long n)
{
  return n % 2 ? o_harm(n) : e_harm(n); 
}

double c_canberra_expected(long n, long k)
{
  double sum = 0.0;
  long t;
  
  for (t=1; t<=k; t++)
    sum += t * (a_harm(2 * k - t) - a_harm(t));
  
  return 2.0 / n * sum + (2.0 * (n - k) / n) * 
    (2 * (k + 1) * (harm(2 * k + 1) - harm(k + 1)) - k);
}

// n: number of lists, p: number of elements of each list
double c_canberra_stability(long *x, long n, long p, long k)
{
  long i, j;
  double expected;
  double d = 0.0;
  
  for(i=0; i<n; i++)
    for(j=i+1; j<n; j++)
      d += c_canberra_location(x+(i*p), x+(j*p), p, k);
  expected = c_canberra_expected(p, k);
  return (d / ((n * (n-1)) / 2.0)) / expected;
}
