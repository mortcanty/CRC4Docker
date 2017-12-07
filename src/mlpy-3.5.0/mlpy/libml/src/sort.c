#include "ml.h"

void dsort(double a[], int ib[],int n,int action)
     /*
       Sort a[] (an array of n doubles) by "heapsort" according to action
       action=SORT_ASCENDING (=1) --> sorting in ascending order
       action=SORT_DESCENDING (=2) --> sorting in descending order
       sort ib[] alongside;
       if initially, ib[] = 0...n-1, it will contain the permutation finally
     */
{
  int l, j, ir, i;
  double ra;
  int ii;
  

  if (n <= 1) return;

  a--; ib--;

  l = (n >> 1) + 1;
  ir = n;

  for (;;) {
    if (l > 1) {
      l = l - 1;
      ra = a[l];
      ii = ib[l];
    }
    else {
      ra = a[ir];
      ii = ib[ir];
      a[ir] = a[1];
      ib[ir] = ib[1];
      if (--ir == 1) {
	a[1] = ra;
	ib[1] = ii;
	return;
      }
    }
    i = l;
    j = l << 1;
    switch(action){
    case SORT_DESCENDING:
      while (j <= ir) {
	if (j < ir && a[j] > a[j + 1]) 
	  ++j;
	if (ra > a[j]) {
	  a[i] = a[j];
	  ib[i] = ib[j];
	  j += (i = j);
	}
	else
	  j = ir + 1;
      }
      break;
    case SORT_ASCENDING:
      while (j <= ir) {
	if (j < ir && a[j] < a[j + 1]) 
	  ++j;
	if (ra < a[j]) {
	  a[i] = a[j];
	  ib[i] = ib[j];
	  j += (i = j);
	}
	else
	  j = ir + 1;
      }
      break;
    }
    a[i] = ra;
    ib[i] = ii;
  }
}

void isort(int a[], int ib[],int n,int action)
     /*
       Sort a[] (an array of n integers) by "heapsort" according to action
       action=SORT_ASCENDING (=1) --> sorting in ascending order
       action=SORT_DESCENDING (=2) --> sorting in descending order
       sort ib[] alongside;
       if initially, ib[] = 0...n-1, it will contain the permutation finally
     */
{
  int l, j, ir, i;
  int ra;
  int ii;
  

  if (n <= 1) return;

  a--; ib--;

  l = (n >> 1) + 1;
  ir = n;

  for (;;) {
    if (l > 1) {
      l = l - 1;
      ra = a[l];
      ii = ib[l];
    }
    else {
      ra = a[ir];
      ii = ib[ir];
      a[ir] = a[1];
      ib[ir] = ib[1];
      if (--ir == 1) {
	a[1] = ra;
	ib[1] = ii;
	return;
      }
    }
    i = l;
    j = l << 1;
    switch(action){
    case SORT_DESCENDING:
      while (j <= ir) {
	if (j < ir && a[j] > a[j + 1]) 
	  ++j;
	if (ra > a[j]) {
	  a[i] = a[j];
	  ib[i] = ib[j];
	  j += (i = j);
	}
	else
	  j = ir + 1;
      }
      break;
    case SORT_ASCENDING:
      while (j <= ir) {
	if (j < ir && a[j] < a[j + 1]) 
	  ++j;
	if (ra < a[j]) {
	  a[i] = a[j];
	  ib[i] = ib[j];
	  j += (i = j);
	}
	else
	  j = ir + 1;
      }
      break;
    }
    a[i] = ra;
    ib[i] = ii;
  }
}
