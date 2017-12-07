#include <stdlib.h>

#define INIT_STD 0
#define INIT_PLUSPLUS 1


void
init_std(double *data, double *means, int nn, int pp,
	 int kk, unsigned long seed);

void
init_plus(double *data, double *means, int nn, int pp,
	  int kk, unsigned long seed);

int
km(double *data, double *means, int *cls, int nn, int pp, int kk);
