#include <stdlib.h>

double linear(double *x, double *y, int n);
double polynomial(double *x, double *y, int n, double gamma, double b, double d);
double gaussian(double *x, double *y, int n, double sigma);
double exponential(double *x, double *y, int n, double sigma);
double sigmoid(double *x, double *y, int n, double gamma, double b);
