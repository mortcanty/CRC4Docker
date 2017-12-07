#include <stdlib.h>


typedef struct Path
{
  int k;
  int *px;
  int *py;
} Path;


int std(long *x, long *y, char **b, int n, int m);
int real(double *x, double *y, char **b, int n, int m, double eps, int delta);
void trace(char **b, int n, int m, Path *p);

