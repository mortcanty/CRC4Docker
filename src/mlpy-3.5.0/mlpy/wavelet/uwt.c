/*  
    This code derives from the R wavelets package and it is modified by Davide
    Albanese <albanese@fbk.it>. 

    The Python interface is written by  Davide Albanese <albanese@fbk.it>.
    (C) 2009 mlpy Developers.

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
#include <math.h>
#include <string.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_wavelet.h>
#include <gsl/gsl_math.h>


#define SQRT_2 1.4142135623730951


void
uwt_forward (double *V, int n, int j,
	      double *h, double *g, int l,
	      double *Wj, double *Vj)
{
  int t, k, z;
  double k_div;

  for(t = 0; t < n; t++)
    {
      k = t;
      Wj[t] = h[0] * V[k];
      Vj[t] = g[0] * V[k];
  
      for(z = 1; z < l; z++)
	{
	  k -= (int) pow(2, (j - 1));
	 
	  k_div = -k / (double) n;
	  
	  if(k < 0) 
	    k += (int) ceil(k_div) * n;
	  
	  Wj[t] += h[z] * V[k];
	  Vj[t] += g[z] * V[k];
	}     
    }
}


void
uwt_backward (double *W, double *V, int j,
	       int n, double *h, double *g,
	       int l, double *Vj)
{
  int t, k, z;
  double k_div;

  for(t = 0; t < n; t++)
    {
      k = t;
      Vj[t] = h[0] * W[k] + g[0] * V[k];
      
      for(z = 1; z < l; z++)
	{
	  k += (int) pow(2, (j - 1));
	  k_div = (double) k / (double) n;
	  
	  if(k >= n)
	    k -= (int) floor(k_div) * n;
	  
	  Vj[t] += h[z] * W[k] + g[z] * V[k];
	}
    }
}


static PyObject *uwt_uwt(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *x = NULL; PyObject *xa = NULL;
  PyObject *Xa = NULL;
  int levels = 0;
  char wf;

  int i, k, j, n, J;
  
  double *_x;
  double *_X;

  double *v;
  double *wj, *vj;
  double *h, *g;

  npy_intp Xa_dims[2];

  gsl_wavelet *wave;
  

  /* Parse Tuple*/
  static char *kwlist[] = {"x", "wf", "k", "levels", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "Oci|i", kwlist, &x, &wf, &k, &levels))
    return NULL;

  xa = PyArray_FROM_OTF(x, NPY_DOUBLE, NPY_IN_ARRAY);
  if (xa == NULL) return NULL;
  
  n = (int) PyArray_DIM(xa, 0);
  _x = (double *) PyArray_DATA(xa);
   
  switch (wf)
    {
    case 'd':
      wave = gsl_wavelet_alloc (gsl_wavelet_daubechies_centered, k);
      break;
      
    case 'h':
      wave = gsl_wavelet_alloc (gsl_wavelet_haar_centered, k);
      break;

    case 'b':
      wave = gsl_wavelet_alloc (gsl_wavelet_bspline_centered, k);
      break;

    default:
      PyErr_SetString(PyExc_ValueError, "wavelet family is not valid");
      return NULL;
    }

  h = (double *) malloc(wave->nc * sizeof(double));
  g = (double *) malloc(wave->nc * sizeof(double));
  
  for(i=0; i<wave->nc; i++)
    {
      h[i] = wave->h1[i] / SQRT_2;
      g[i] = wave->g1[i] / SQRT_2;
    }

  if (levels == 0)
    J = (int) floor(log(((n-1) / (wave->nc-1)) + 1) / log(2));
  else
    J = levels;

  Xa_dims[0] = (npy_intp) (2 * J);
  Xa_dims[1] = PyArray_DIM(xa, 0);
  Xa = PyArray_SimpleNew(2, Xa_dims, NPY_DOUBLE);
  _X = (double *) PyArray_DATA(Xa);

  v = _x;
  for(j=0; j<J; j++)
    {
      wj = _X +(j * n);
      vj = _X +((j + J) * n);
      uwt_forward(v, n, j+1, g, h, wave->nc, wj,  vj);
      v = vj;
    }
    
  gsl_wavelet_free(wave);
  free(h);
  free(g);
  Py_DECREF(xa);

  return Py_BuildValue("N", Xa);
}


static PyObject *uwt_iuwt(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *X = NULL; PyObject *Xa = NULL;
  
  PyObject *xa = NULL;
  
  char wf;
  int i, k, n, J;
  double *w1, *v1;

  double *_X;
  double *_x;
  double *h, *g;

  npy_intp xa_dims[1];

  gsl_wavelet *wave;
  
  /* Parse Tuple*/
  static char *kwlist[] = {"X", "wf", "k", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "Oci", kwlist, &X, &wf, &k))
    return NULL;

  Xa = PyArray_FROM_OTF(X, NPY_DOUBLE, NPY_IN_ARRAY);
  if (Xa == NULL) return NULL;
  
  n = (int) PyArray_DIM(Xa, 1);
  J = ((int) PyArray_DIM(Xa, 0)) / 2;
  
  _X = (double *) PyArray_DATA(Xa);
   
  switch (wf)
    {
    case 'd':
      wave = gsl_wavelet_alloc (gsl_wavelet_daubechies, k);
      break;
      
    case 'h':
      wave = gsl_wavelet_alloc (gsl_wavelet_haar, k);
      break;

    case 'b':
      wave = gsl_wavelet_alloc (gsl_wavelet_bspline, k);
      break;

    default:
      PyErr_SetString(PyExc_ValueError, "wavelet family is not valid");
      return NULL;
    }

  h = (double *) malloc(wave->nc * sizeof(double));
  g = (double *) malloc(wave->nc * sizeof(double));
  
  for(i=0; i<wave->nc; i++)
    {
      h[i] = wave->h2[i] / SQRT_2;
      g[i] = wave->g2[i] / SQRT_2;
    }

  w1 = _X;
  v1 = _X + (J * n);

  xa_dims[0] = (npy_intp) n;  
  xa = PyArray_SimpleNew(1, xa_dims, NPY_DOUBLE);
  _x = (double *) PyArray_DATA(xa);

  uwt_backward (w1, v1, 1, n, g, h, wave->nc, _x);
    
  gsl_wavelet_free(wave);
  free(h);
  free(g);
  Py_DECREF(Xa);

  return Py_BuildValue("N", xa);
}


/* Doc strings: */
static char module_doc[]  = "Undecimated Wavelet Transform Module";

static char uwt_uwt_doc[] =
  "Undecimated Wavelet Tranform\n\n"
  ":Parameters:\n"
  "   x : 1d array_like object (the length is restricted to powers of two)\n"
  "      data\n"
  "   wf : string ('d': daubechies, 'h': haar, 'b': bspline)\n"
  "      wavelet family\n"
  "   k : int\n"
  "      member of the wavelet family\n\n"
  "         * daubechies: k = 4, 6, ..., 20 with k even\n"
  "         * haar: the only valid choice of k is k = 2\n"
  "         * bspline: k = 103, 105, 202, 204, 206, 208, 301, 303, 305 307, 309\n\n"
  "   levels : int\n"
  "      level of the decomposition (J).\n"
  "      If levels = 0 this is the value J such that the length of X\n"
  "      is at least as great as the length of the level J wavelet filter,\n"
  "      but less than the length of the level J+1 wavelet filter.\n"
  "      Thus, j <= log_2((n-1)/(l-1)+1), where n is the length of x\n\n"
  ":Returns:\n"
  "   X : 2d numpy array (2J * len(x))\n"
  "      misaligned scaling and wavelet coefficients::\n\n"
  "         [[wavelet coefficients W_1]\n"
  "          [wavelet coefficients W_2]\n"
  "                        :\n"
  "          [wavelet coefficients W_J]\n"
  "          [scaling coefficients V_1]\n"
  "          [scaling coefficients V_2]\n"
  "                       :\n"
  "          [scaling coefficients V_J]]"
  ;

static char uwt_iuwt_doc[] =
  "Inverse Undecimated Wavelet Tranform\n\n"
  ":Parameters:\n"
  "   X : 2d array_like object (the length is restricted to powers of two)\n"
  "      misaligned scaling and wavelet coefficients\n"
  "   wf : string ('d': daubechies, 'h': haar, 'b': bspline)\n"
  "      wavelet family\n"
  "   k : int\n"
  "      member of the wavelet family\n\n"
  "         * daubechies: k = 4, 6, ..., 20 with k even\n"
  "         * haar: the only valid choice of k is k = 2\n"
  "         * bspline: k = 103, 105, 202, 204, 206, 208, 301, 303, 305 307, 309\n\n"
  ":Returns:\n"
  "   x : 1d numpy array\n"
  "      data"
  ;


/* Method table */
static PyMethodDef uwt_methods[] = {
  {"uwt",
   (PyCFunction)uwt_uwt,
   METH_VARARGS | METH_KEYWORDS,
   uwt_uwt_doc},
  {"iuwt",
   (PyCFunction)uwt_iuwt,
   METH_VARARGS | METH_KEYWORDS,
   uwt_iuwt_doc},
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "_uwt",
  module_doc,
  -1,
  uwt_methods,
  NULL, NULL, NULL, NULL
};

PyObject *PyInit__uwt(void)
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

PyMODINIT_FUNC init_uwt(void)
{
  PyObject *m;
  
  m = Py_InitModule3("_uwt", uwt_methods, module_doc);
  if (m == NULL) {
    return;
  }
  
  import_array();
}

#endif
