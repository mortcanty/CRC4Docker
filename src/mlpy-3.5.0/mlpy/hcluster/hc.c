/*  
    This code derives from the R amap package and it is modified by Davide
    Albanese <albanese@fbk.it>. 
    
    The Python interface is written by  Davide Albanese <albanese@fbk.it>.
    (C) 2008 mlpy Developers.

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

#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>

#define MAX( A , B ) ((A) > (B) ? (A) : (B))
#define MIN( A , B ) ((A) < (B) ? (A) : (B))

#define WARD 1
#define SINGLE 2
#define COMPLETE 3
#define AVERAGE 4
#define MCQUITTY 5
#define MEDIAN 6
#define CENTROID 7


long ioffst(long n, long i, long j)
{
  return j + i * n - (i + 1) * (i + 2) / 2;
}


/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/*                                                             */
/* Given a HIERARCHIC CLUSTERING, described as a sequence of   */
/* agglomerations, prepare the seq. of aggloms. and "horiz."   */
/* order of objects for plotting the dendrogram using S routine*/
/* 'plclust'.                                                  */
/*                                                             */
/* Parameters:                                                 */
/*                                                             */
/* IA, IB:       vectors of dimension N defining the agglomer- */
/*                ations.                                      */
/* IIA, IIB:     used to store IA and IB values differently    */
/*               (in form needed for S command `plclust`       */
/* IORDER:       "horiz." order of objects for dendrogram      */
/*                                                             */
/* F. Murtagh, ESA/ESO/STECF, Garching, June 1991              */
/* C adaptation:              A. Lucas, Nov 2002               */
/*                                                             */
/* HISTORY                                                     */
/*                                                             */
/* Adapted from routine HCASS, which additionally determines   */
/*  cluster assignments at all levels, at extra comput. expense*/
/*                                                             */
/*-------------------------------------------------------------*/
void 
hcass2(long n, long *ia, long *ib, long *iia, long *iib,
       long *iorder)
{
  long i, j, k, k1, k2, loc;
  
  /*  Following bit is to get seq. of merges into format acceptable to plclust
   *  I coded clusters as lowest seq. no. of constituents; S's `hclust' codes
   *  singletons as -ve numbers, and non-singletons with their seq. nos.
   */

  for (i=0; i<n; i++ ) 
    {
      iia[i] = - ia[i];
      iib[i] = - ib[i];
    }
  
  for (i=0; i<(n-2); i++)
    /* In the following, smallest (+ve or -ve) seq. no. wanted */
    {
      k = MIN(ia[i], ib[i]);
      for (j=i+1; j<(n-1); j++) 
	{
	  if( ia[j] == k )
	    iia[j] = i + 1;
	  if( ib[j] == k )
	    iib[j] = i + 1;
	}
    }
  
  for (i=0; i< (n-1); i++ ) 
    {
      if ((iia[i] > 0) && (iib[i] < 0))
	{
	  k = iia[i];
	  iia[i] = iib[i];
	  iib[i] = k;
	}
      
      if ((iia[i] > 0) && (iib[i] > 0))
	{
	  k1 = MIN (iia[i], iib[i]);
	  k2 = MAX (iia[i], iib[i]);
	  iia[i] = k1;
	  iib[i] = k2;
	}
    }
  
  /* New part for 'order' */

  iorder[0] = - iia[n-2];
  iorder[1] = - iib[n-2];
  loc = 2;
  
  for (i=(n-3); i>=0; i--) 
    for (j=0; j<loc; j++ ) 
      if (-iorder[j] == i+1)
	{
	  /* REPLACE IORDER(J) WITH IIA(I) AND IIB(I) */ 
	  iorder[j] = -iia[i];
	  if (j == (loc-1)) 
	    {
	      loc++;
	      iorder[loc-1]= -iib[i];
	      break; /* for j */
	    }
	  
	  loc++;
	  for (k=loc-1; k>=(j+1); k--)
	    iorder[k] = iorder[k-1];
	  
	  iorder[j+1] = -iib[i];
	  break; /* for j */
	}
}


void 
hclust(long n, int iopt, double *diss, long *ia, long *ib,
       long *iorder, double *height)
       
{
  long im = 0, jm = 0, jj = 0;
  long i, j, ncl, ind, i2, j2, k, ind1, ind2, ind3;
  double inf, dmin, x, xx;
  long *nn;
  double *disnn;
  short int *flag;
  double *membr;
  long *iia;
  long *iib;
  
 
  nn    = (long*) malloc (n * sizeof(long));
  disnn = (double*) malloc (n * sizeof(double));
  flag  = (short int*) malloc (n * sizeof(short int));
  membr = (double*) malloc (n * sizeof(double));

  for (i=0; i<n; i++)
    membr[i] = 1.0;
  
  /* Initialisation */
  for (i=0; i<n ; i++)
    flag[i] = 1;
  
  ncl = n;
  inf = DBL_MAX;

  /*
   * Carry out an agglomeration - first create list of NNs
   */
  for ( i=0; i<(n-1) ; i++)
    {
      dmin = inf;
      for (j=i+1; j<n; j++)
	{
	  ind = ioffst(n, i, j);
	  if (diss[ind] < dmin)
	    {
	      dmin = diss[ind];
	      jm = j;
	    }
	}
      nn[i] = jm;
      disnn[i] = dmin;
    }
 
  /*
   *  Repeat previous steps until N-1 agglomerations carried out.
   */
  while (ncl > 1)
    {
      /*
       * Next, determine least diss. using list of NNs
       */
      dmin = inf;
      for (i=0; i<(n-1) ; i++)
	if (flag[i])
	  if (disnn[i] < dmin )
	    {
	      dmin = disnn[i];
	      im = i;
	      jm = nn[i];
	    }
      ncl = ncl - 1;
      
      /*
       * This allows an agglomeration to be carried out.
       * At step n-ncl, we found dmin = dist[i2, j2]
       */
      
      i2 = MIN (im,jm);
      j2 = MAX (im,jm);
      ia[n-ncl-1] = i2 + 1;
      ib[n-ncl-1] = j2 + 1;
      height[n-ncl-1] = dmin;
    	  
      /*
       * Update dissimilarities from new cluster.
       */
      flag[j2] = 0;
      dmin = inf;
      for (k=0; k<n; k++)
	{
	  if(flag[k] && (k != i2) )
	    {      
	      x =  membr[i2] + membr[j2] + membr[k];

	      if (i2 < k)
		ind1 = ioffst(n, i2, k);
	      else
		ind1 = ioffst(n, k, i2);
	      if (j2 < k)
		ind2 = ioffst(n, j2, k);
	      else
		ind2 = ioffst(n, k, j2);
	      
	      ind3 = ioffst(n, i2, j2);
	      xx = diss[ind3];
	      
	      /*
	       * Gi & Gj are agglomerated => Gii
	       * We are calculating D(Gii,Gk) (for all k)
	       *
	       * diss[ind1] = D(Gi,Gk) (will be replaced by  D(Gii,Gk))
	       * diss[ind2] = D(Gj,Gk) 
	       * xx = diss[ind3] = D(Gi,Gj)
	       *
	       * membr[i2] = #Gi
	       * membr[j2] = #Gj
	       * membr[k]  = #Gk
	       * 
	       * x = #Gi + #Gj + #Gk
	       */
	      switch(iopt)
		{
		   /*
		     * WARD'S MINIMUM VARIANCE METHOD - IOPT=1.
		     */	      
		  case 1: 
		    diss[ind1] = (membr[i2]+membr[k])* diss[ind1] + 
		      (membr[j2]+membr[k])* diss[ind2] - 
		      membr[k] * xx;
		    diss[ind1] = diss[ind1] / x;
		    break; 
	       
		    /*
		     * SINGLE LINK METHOD - IOPT=2.
		     */
		  case 2:
		    diss[ind1] = MIN (diss[ind1],diss[ind2]);
		    break; 
		    /*
		     * COMPLETE LINK METHOD - IOPT=3.
		     */
		  case 3: 
		    diss[ind1] = MAX (diss[ind1],diss[ind2]);
		    break; 
		    /*
		     * AVERAGE LINK (OR GROUP AVERAGE) METHOD - IOPT=4.
		     */
		  case 4:  
		    diss[ind1] = ( membr[i2] * diss[ind1] +
				   membr[j2] * diss[ind2] ) /
		      (membr[i2] + membr[j2]); 
		    break; 
		    /*
		     *  MCQUITTY'S METHOD - IOPT=5.
		     */
		  case 5:
		    diss[ind1] = 0.5 * diss[ind1]+0.5*diss[ind2]; 
		    break;
		    /*
		     * MEDIAN (GOWER'S) METHOD - IOPT=6.
		     */
		  case 6:
		    diss[ind1] = 0.5* diss[ind1]+0.5*diss[ind2] -0.25*xx;
		    break;
		    /*
		     * CENTROID METHOD - IOPT=7.
		     */
		  case 7:
		    diss[ind1] = (membr[i2]*diss[ind1] + membr[j2]*diss[ind2] - 
				  membr[i2] * membr[j2]*xx /
				  (membr[i2] + membr[j2]) ) /
		      (membr[i2] + membr[j2]);
		    break;	  
		} 

	      if ((i2 <= k) && ( diss[ind1] < dmin ))
		{
		  dmin = (double) diss[ind1];
		  jj = k;
		}
	    } 
	}
      
      membr[i2] = membr[i2] + membr[j2];
      disnn[i2] = dmin;
      nn[i2] = jj;
            
      /*
       *  Update list of NNs insofar as this is required.
       */
      for (i=0; i<(n-1); i++)
	if(flag[i] && ((nn[i] == i2) || (nn[i] == j2)))
	  {
	    /* (Redetermine NN of I:)   */
	    dmin = inf;
	    for (j=i+1; j<n; j++)
	      {
		ind = ioffst(n,i,j);
		if (flag[j] && (i != j) && (diss[ind] < dmin))
		  {
		    dmin = diss[ind];
		    jj = j;
		  }

		nn[i] = jj;
		disnn[i] = dmin;
	      }
	  }
    }  
  
  free(nn);
  free(disnn);
  free(flag);
  free(membr);
    
  iia = (long*) malloc (n * sizeof(long));
  iib = (long*) malloc (n * sizeof(long));
  
  hcass2(n, ia, ib, iia, iib, iorder);
 
  for (i=0; i<n; i++ ) 
    {
      ia[i] = iia[i];
      ib[i] = iib[i];
    }
  
  free(iia);
  free(iib);
}


void 
cutree(long n, long *ia, long *ib, double *height,
       double ht, long *ans)
{
  long i;
  long k, l, nclust, m1, m2, j;
  short int *sing, flag;
  long *m_nr, *z;
  long which;
  
  /* compute which (number of clusters at height ht) */
  
  height[n-1] = DBL_MAX;
  flag = 0;
  i = 0;
  while(!flag)
    {
      if(height[i] > ht)
	  flag = 1;
      i++;     
    }
  
  which = n + 1 - i;

  /* using 1-based indices ==> "--" */
  sing = (short int *) malloc(n * sizeof(short int)); sing--;
  m_nr = (long *) malloc(n * sizeof(long)); m_nr--;
  z    = (long *) malloc(n * sizeof(long)); z--;
  
  for(k = 1; k <= n; k++)
    {
      sing[k] = 1;  /* is k-th obs. still alone in cluster ? */
      m_nr[k] = 0;     /* containing last merge-step number of k-th obs. */
    }
  
  for(k = 1; k <= n-1; k++)
    {
      /* k-th merge, from n-k+1 to n-k atoms: (m1,m2) = merge[ k , ] */
      m1 = ia[k-1];
      m2 = ib[k-1];
      
      if(m1 < 0 && m2 < 0)
	{   
	  /* merging atoms [-m1] and [-m2] */
	  m_nr[-m1] = m_nr[-m2] = k;
	  sing[-m1] = sing[-m2] = 0;
	}
      else 
	if(m1 < 0 || m2 < 0)
	  {
	    /* the other >= 0 */
	    if(m1 < 0)
	      { 
		j = -m1;
		m1 = m2;
	      }
	    else
	      j = -m2;
	    
	    /* merging atom j & cluster m1 */
	    for(l=1; l<=n; l++)
	      if (m_nr[l] == m1)
		m_nr[l] = k;
	    
	    m_nr[j] = k;
	    sing[j] = 0;
	  }
	else
	  {
	    /* both m1, m2 >= 0 */
	    for(l=1; l<=n; l++)
	      if(m_nr[l]==m1 || m_nr[l]==m2)
		m_nr[l] = k;
	  }
     
      if(which == n-k)
	{
	  for(l = 1; l <= n; l++)
	    z[l] = 0;
	  
	  nclust = 0;
	  
	  for(l = 1, m1 = 0; l <= n; l++, m1++)
	    {
	      if(sing[l])
		ans[m1] = ++nclust;
	      else 
		{
		  if (z[m_nr[l]] == 0)
		    z[m_nr[l]] = ++nclust;
		  ans[m1] = z[m_nr[l]];
		}
	    }
	}
    }

  if(which == n)
    for(l = 1, m1 = 0; l <= n; l++, m1++)
      ans[m1] = l;
     
  free(sing+1);
  free(m_nr+1);
  free(z+1);
}


static PyObject *hc_linkage(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *dist = NULL; PyObject *dist_a = NULL;
  
  double *dist_c;
  int method;
  int n;
  
  PyObject *ia = NULL;
  PyObject *ib = NULL;
  PyObject *height = NULL; 
  PyObject *iorder = NULL; 

  npy_intp ia_dims[1];
  npy_intp ib_dims[1];
  npy_intp height_dims[1];
  npy_intp iorder_dims[1];
  
  long *ia_c, *ib_c;
  long *iorder_c;
  double *height_c;
  
  
  static char *kwlist[] = {"n", "dist", "method", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "iOi", kwlist,
				   &n, &dist, &method))
    return NULL;

  dist_a = PyArray_FROM_OTF(dist, NPY_DOUBLE, NPY_IN_ARRAY);
  if (dist_a == NULL) return NULL;

  dist_c = (double *) PyArray_DATA(dist_a);

  ia_dims[0] = n;
  ia = PyArray_SimpleNew(1, ia_dims, NPY_LONG);
  ia_c = (long *) PyArray_DATA(ia);

  ib_dims[0] = n;
  ib = PyArray_SimpleNew(1, ib_dims, NPY_LONG);
  ib_c = (long *) PyArray_DATA(ib);
  
  iorder_dims[0] = n;
  iorder = PyArray_SimpleNew(1, iorder_dims, NPY_LONG);
  iorder_c = (long *) PyArray_DATA(iorder);

  height_dims[0] = n;
  height = PyArray_SimpleNew(1, height_dims, NPY_DOUBLE);
  height_c = (double *) PyArray_DATA(height);
  
  hclust((long) n, method, dist_c, ia_c, ib_c, iorder_c, height_c);
     
  Py_DECREF(dist_a);

  return Py_BuildValue("(N, N, N, N)", ia, ib, iorder, height);
}


static PyObject *hc_cut(PyObject *self, PyObject *args, PyObject *keywds)
{
  /* Inputs */
  PyObject *ia = NULL; PyObject *ia_a = NULL;
  PyObject *ib = NULL; PyObject *ib_a = NULL;
  PyObject *height = NULL; PyObject *height_a = NULL;
  double t;
  
  npy_intp n;
   
  long *ia_c;
  long *ib_c;
  double *height_c;

  PyObject *cmap = NULL;
  npy_intp cmap_dims[1];
  long *cmap_c;
  

  static char *kwlist[] = {"ia", "ib", "height", "t", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOd", kwlist,
				   &ia, &ib, &height, &t))
    return NULL;

  ia_a = PyArray_FROM_OTF(ia, NPY_LONG, NPY_IN_ARRAY);
  if (ia_a == NULL) return NULL;
  
  ib_a = PyArray_FROM_OTF(ib, NPY_LONG, NPY_IN_ARRAY);
  if (ib_a == NULL) return NULL;
  
  height_a = PyArray_FROM_OTF(height, NPY_DOUBLE, NPY_IN_ARRAY);
  if (height_a == NULL) return NULL;
  
  n = PyArray_DIM(height_a, 0);
  
  ia_c = (long *) PyArray_DATA(ia_a);
  ib_c = (long *) PyArray_DATA(ib_a);
  height_c = (double *) PyArray_DATA(height_a);
 
  cmap_dims[0] = n;
  cmap = PyArray_SimpleNew(1, cmap_dims, NPY_LONG);
  cmap_c = (long *) PyArray_DATA(cmap);

  cutree((long) n, ia_c, ib_c, height_c, t, cmap_c);
  
  Py_DECREF(ia_a);
  Py_DECREF(ib_a);
  Py_DECREF(height_a);

  return Py_BuildValue("N", cmap);
}


static char module_doc[] = "";
static char hc_linkage_doc[] = "";
static char hc_cut_doc[] = "";

/* Method table */
static PyMethodDef hc_methods[] = {
  {"linkage",
   (PyCFunction)hc_linkage,
   METH_VARARGS | METH_KEYWORDS,
   hc_linkage_doc},
  {"cut",
   (PyCFunction)hc_cut,
   METH_VARARGS | METH_KEYWORDS,
   hc_cut_doc},
  {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "chc",
  module_doc,
  -1,
  hc_methods,
  NULL, NULL, NULL, NULL
};

PyObject *PyInit_chc(void)
{
  PyObject *m;
  m = PyModule_Create(&moduledef);
  if (!m) {
    return NULL;
  }

  import_array();

  return m;
}

#else

PyMODINIT_FUNC initchc(void)
{
  PyObject *m;
  
  m = Py_InitModule3("chc", hc_methods, module_doc);
  if (m == NULL) {
    return;
  }
  
  import_array();
}

#endif
