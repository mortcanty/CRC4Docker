/*  
    This code is written by <albanese@fbk.it>.
    (C) 2008 mlpy Developers.
    
    See DWT in the GSL Library.

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
#include <gsl/gsl_wavelet.h>
#include <gsl/gsl_math.h>


static PyObject *dwt_dwt(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *x = NULL; PyObject *xcopy = NULL;
  
  char wf;
  int k, n;
  double *_xcopy;
  PyObject *centered = Py_False;
  gsl_wavelet *w;
  gsl_wavelet_workspace *work;

  /* Parse Tuple*/
  static char *kwlist[] = {"x", "wf", "k", "centered", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "Oci|O", kwlist,
				   &x, &wf, &k, &centered))
    return NULL;

  /* Build xcopy */
  xcopy = PyArray_FROM_OTF(x, NPY_DOUBLE, NPY_OUT_ARRAY | NPY_ENSURECOPY);
  if (xcopy == NULL) return NULL;
  
  n = (int) PyArray_DIM(xcopy, 0);
  _xcopy = (double *) PyArray_DATA(xcopy);
  
  switch (wf)
    {
    case 'd':
      if (centered == Py_True)
	w = gsl_wavelet_alloc (gsl_wavelet_daubechies_centered, k);
      else
	w = gsl_wavelet_alloc (gsl_wavelet_daubechies, k);
      break;
      
    case 'h':
      if (centered == Py_True)
	w = gsl_wavelet_alloc (gsl_wavelet_haar_centered, k);
      else
	w = gsl_wavelet_alloc (gsl_wavelet_haar, k);
      break;

    case 'b':
      if (centered == Py_True)
	w = gsl_wavelet_alloc (gsl_wavelet_bspline_centered, k);
      else
	w = gsl_wavelet_alloc (gsl_wavelet_bspline, k);
      break;

    default:
      PyErr_SetString(PyExc_ValueError, "wavelet family is not valid");
      return NULL;
    }
  
  work = gsl_wavelet_workspace_alloc (n);
  
  gsl_wavelet_transform_forward (w, _xcopy, 1, n, work);
    
  gsl_wavelet_free (w);
  gsl_wavelet_workspace_free (work);
  
  return Py_BuildValue("N", xcopy);
}


static PyObject *dwt_idwt(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *x = NULL; PyObject *xcopy = NULL;
  
  char wf;
  int k, n;
  double *_xcopy;
  PyObject *centered = Py_False;
  gsl_wavelet *w;
  gsl_wavelet_workspace *work;

  /* Parse Tuple*/
  static char *kwlist[] = {"X", "wf", "k", "centered", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "Oci|O", kwlist,
				   &x, &wf, &k, &centered))
    return NULL;

   
  /* Build xcopy */
  xcopy = PyArray_FROM_OTF(x, NPY_DOUBLE, NPY_OUT_ARRAY | NPY_ENSURECOPY);
  if (xcopy == NULL) return NULL;
  
  n = (int) PyArray_DIM(xcopy, 0);
  _xcopy = (double *) PyArray_DATA(xcopy);
 
  switch (wf)
    {
    case 'd':
      if (centered == Py_True)
	w = gsl_wavelet_alloc (gsl_wavelet_daubechies_centered, k);
      else
	w = gsl_wavelet_alloc (gsl_wavelet_daubechies, k);
      break;
      
    case 'h':
      if (centered == Py_True)
	w = gsl_wavelet_alloc (gsl_wavelet_haar_centered, k);
      else
	w = gsl_wavelet_alloc (gsl_wavelet_haar, k);
      break;

    case 'b':
      if (centered == Py_True)
	w = gsl_wavelet_alloc (gsl_wavelet_bspline_centered, k);
      else
	w = gsl_wavelet_alloc (gsl_wavelet_bspline, k);
      break;

    default:
      PyErr_SetString(PyExc_ValueError, "wavelet family is not valid");
      return NULL;
    }
  
  work = gsl_wavelet_workspace_alloc (n);
  
  gsl_wavelet_transform_inverse (w, _xcopy, 1, n, work);
  
  gsl_wavelet_free (w);
  gsl_wavelet_workspace_free (work);
  
  return Py_BuildValue("N", xcopy);
}


/* Doc strings: */
static char module_doc[]  = "Discrete Wavelet Transform Module from GSL";

static char dwt_dwt_doc[] =
  "Discrete Wavelet Tranform\n\n"
  ":Parameters:\n"
  "   x : 1d array_like object (the length is restricted to powers of two)\n"
  "      data\n"
  "   wf : string ('d': daubechies, 'h': haar, 'b': bspline)\n"
  "      wavelet family\n"
  "   k : integer\n"
  "      member of the wavelet family\n\n"
  "         * daubechies : k = 4, 6, ..., 20 with k even\n"
  "         * haar : the only valid choice of k is k = 2\n"
  "         * bspline : k = 103, 105, 202, 204, 206, 208, 301, 303, 305 307, 309\n"
  "   centered : bool\n"
  "      align the coefficients of the various sub-bands on edges.\n"
  "      Thus the resulting visualization of the coefficients of the\n"
  "      wavelet transform in the phase plane is easier to understand.\n\n"
  ":Returns:\n"
  "  X : 1d numpy array\n"
  "    discrete wavelet transformed data\n\n"
  "Example\n\n"
  ">>> import numpy as np\n"
  ">>> import mlpy.wavelet as wave\n"
  ">>> x = np.array([1,2,3,4,3,2,1,0])\n"
  ">>> wave.dwt(x=x, wf='d', k=6)\n"
  "array([ 5.65685425,  3.41458985,  0.29185347, -0.29185347, -0.28310081,\n"
  "       -0.07045258,  0.28310081,  0.07045258])\n";

static char dwt_idwt_doc[] =
  "Inverse Discrete Wavelet Tranform\n\n"
  ":Parameters:\n"
  "   X : 1d array_like object\n"
  "      discrete wavelet transformed data\n"
  "   wf : string ('d': daubechies, 'h': haar, 'b': bspline)\n"
  "      wavelet type\n"
  "   k : integer\n"
  "      member of the wavelet family\n\n"
  "         * daubechies : k = 4, 6, ..., 20 with k even\n"
  "         * haar : the only valid choice of k is k = 2\n"
  "         * bspline : k = 103, 105, 202, 204, 206, 208, 301, 303, 305 307, 309\n\n"
  "   centered : bool\n"
  "      if the coefficients are aligned\n\n"
  ":Returns:\n"
  "   x : 1d numpy array\n"
  "      data\n\n"
  "Example:\n\n"
  ">>> import numpy as np\n"
  ">>> import mlpy.wavelet as wave\n"
  ">>> X = np.array([ 5.65685425,  3.41458985,  0.29185347, -0.29185347, -0.28310081,\n"
  "...               -0.07045258,  0.28310081,  0.07045258])\n"
  ">>> wave.idwt(X=X, wf='d', k=6)\n"
  "array([  1.00000000e+00,   2.00000000e+00,   3.00000000e+00,\n"
  "         4.00000000e+00,   3.00000000e+00,   2.00000000e+00,\n"
  "         1.00000000e+00,  -3.53954610e-09])\n";


/* Method table */
static PyMethodDef dwt_methods[] = {
  {"dwt",
   (PyCFunction)dwt_dwt,
   METH_VARARGS | METH_KEYWORDS,
   dwt_dwt_doc},
  {"idwt",
   (PyCFunction)dwt_idwt,
   METH_VARARGS | METH_KEYWORDS,
   dwt_idwt_doc},
  {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "_dwt",
  module_doc,
  -1,
  dwt_methods,
  NULL, NULL, NULL, NULL
};

PyObject *PyInit__dwt(void)
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

PyMODINIT_FUNC init_dwt(void)
{
  PyObject *m;
  
  m = Py_InitModule3("_dwt", dwt_methods, module_doc);
  if (m == NULL) {
    return;
  }
  
  import_array();
}

#endif

