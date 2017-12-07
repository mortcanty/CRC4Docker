/*  
    This code is written by Davide Albanese <davide.albanese@gmail.com>.
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


#define MAX(a, b) ((a) > (b) ? (a):(b))
#define MIN(a, b) ((a) < (b) ? (a):(b))


struct node
{
  long data;
  struct node *next;
} Node;

typedef struct node NODE;


int
enqueue(NODE **head, NODE **tail, int data)
{
  NODE *tmp;
  tmp = malloc(sizeof(NODE));
  
  if (tmp == NULL)
    return 1;
  
  tmp->data = data;
  tmp->next = NULL;
  
  if (*head == NULL)
    *head = tmp;
  else
    (*tail)->next = tmp;
  *tail = tmp;
  
  return 0;
}



int *fp_win(double *x, int n, int span, int *m)
{
  // m: size of output vector

  NODE *head = NULL, *tail = NULL;
  NODE *tmp = NULL;
  int l_min, l_max, r_min, r_max;
  int i, j, k;
  short int is_peak;
  int dist;
  int *ret;


  dist = (span + 1) / 2;

  *m = 0;
  for (i=0; i<n; i++)
    {
      l_min = MAX(i-dist+1, 0);
      l_max = i-1;
      r_min = i+1;
      r_max = MIN(i+dist-1, n-1);
      
      is_peak = 1;
      
      /* left side */
      for (j=l_min; j<=l_max; j++)
	if (x[j] >= x[i])
	  {
	    is_peak = 0;
	    break;
	  }
    
      /* right side */
      if (is_peak == 1)
	for (j=r_min; j<=r_max; j++)
	  if (x[j] >= x[i])
	    {
	      is_peak = 0;
	      break;
	    }
          
      if (is_peak == 1)
	{
	  if (enqueue(&head, &tail, i))
	    return NULL; 
	  *m = *m + 1;
	}
    }

  // build the output vector

  ret = (int *) malloc (*m * sizeof(int));
    
  /* copy and free the list */
  for (k=0; k<*m; k++)
    {
      tmp = head->next;
      ret[k] = head->data;
      free(head);
      head = tmp;
    }
  tail = NULL;

  return ret;
}
