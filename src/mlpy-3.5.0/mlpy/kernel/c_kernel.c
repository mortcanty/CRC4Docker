/*  
    This code is written by <albanese@fbk.it>.
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
#include <math.h>


/* x'*y */
double linear(double *x, double *y, int n)
{
  int i;
  double k = 0.0;
  
  for (i=0; i<n; i++)
    k += x[i] * y[i];
  
  return k;
}

/* (gamma*u'*v + b)^d */
double polynomial(double *x, double *y, int n, double gamma,
		  double b, double d)
{
  int i;
  double p = 0.0;
  
  for (i=0; i<n; i++)
    p += x[i] * y[i];
  
  return pow(gamma * p + b, d);
}

/* exp(-||x-y||^2 / 2 * sigma^2) */
double gaussian(double *x, double *y, int n, double sigma)
{
  int i;
  double d = 0.0;
  
  for (i=0; i<n; i++)
    d += pow(x[i]-y[i], 2);
  
  return exp(-d / (2*pow(sigma, 2)));
}

/* exp(-||x-y|| / 2 * sigma^2) */
double exponential(double *x, double *y, int n, double sigma)
{
  int i;
  double d = 0.0;
  
  for (i=0; i<n; i++)
    d += pow(x[i]-y[i], 2);
  
  return exp(-sqrt(d) / (2*pow(sigma, 2)));
}

/* tanh(gamma*x'*y + b) */
double sigmoid(double *x, double *y, int n, double gamma, double b)
{
  int i;
  double p = 0.0;
  
  for (i=0; i<n; i++)
    p += x[i] * y[i];
  
  return tanh(gamma * p + b);
}
