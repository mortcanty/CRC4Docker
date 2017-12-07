#include <math.h>

double l1_distance(double x[],double y[],int n)
{
  int i;
  double out = 0.0;

  for(i=0;i<n;i++)
    out += fabs(x[i] - y[i]);

  return out;
}

double euclidean_squared_distance(double x[],double y[],int n)
{
  int i;
  double out = 0.0;
  double tmp;
  

  for(i=0;i<n;i++){
    tmp = x[i] - y[i];
    out += tmp * tmp;
  }
  

  return out;
}

double euclidean_distance(double x[],double y[],int n)
{
  int i;
  double out = 0.0;
  double tmp;
  

  for(i=0;i<n;i++){
    tmp = x[i] - y[i];
    out += tmp * tmp;
  }
  

  return sqrt(out);
}

  
double scalar_product(double x[],double y[],int n)
{
  double out;
  int i;

  out=0.0;
  for(i=0;i<n;i++)
    out += x[i] * y[i];

  return out;
}

double euclidean_norm(double x[],int n)
{
  return sqrt(scalar_product(x,x,n));
}
    
