/*  
    This code is written by Davide Albanese <albanese@fbk.it>.
    (C) 2010 mlpy Developers.
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
#include <float.h>


static PyObject *borda_core(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *x=NULL; PyObject *x_arr=NULL;
  int k;

  PyObject *ext_arr=NULL; 
  PyObject *pos_arr=NULL; 
  
  long *ext_c, *x_c;
  double *pos_c;

  int p, n;
  long idx;

  npy_intp nn, pp;
  npy_intp ext_dims[1];
  npy_intp pos_dims[1];

  /* Parse Tuple*/
  static char *kwlist[] = {"x", "k", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "Oi", kwlist,
				   &x, &k))
    return NULL;
  
  x_arr = PyArray_FROM_OTF(x, NPY_LONG, NPY_IN_ARRAY);
  if (x_arr == NULL) return NULL;
  
  if (PyArray_NDIM(x_arr) != 2)
    {
      PyErr_SetString(PyExc_ValueError, "x must be 2d array");
      return NULL;
    }

  x_c = (long *) PyArray_DATA(x_arr);
  nn = PyArray_DIM(x_arr, 0);
  pp = PyArray_DIM(x_arr, 1);

  ext_dims[0] = pp;
  ext_arr = PyArray_SimpleNew (1, ext_dims, NPY_LONG);
  ext_c = (long *) PyArray_DATA(ext_arr);
  
  pos_dims[0] = pp;
  pos_arr = PyArray_SimpleNew (1, pos_dims, NPY_DOUBLE);
  pos_c = (double *) PyArray_DATA(pos_arr);
  
  for(p=0; p<pp; p++)
    {
      ext_c[p] = 0;
      pos_c[p] = 0.0;
    }

  for(p=0; p<k; p++)
    {
      for(n=0; n<nn; n++)
	{
	  idx = x_c[p + (n * pp)];
	  if ((idx < 0) || (idx > pp-1))
	    {
	      PyErr_SetString(PyExc_ValueError, 
			      "ids values must be in [0, x.shape[1]-1]");
	      return NULL;
	    }

	  ext_c[idx] += 1;
	  pos_c[idx] += p + 1;
	}    
    }

  for(p=0; p<pp; p++)
    if (pos_c[p] != 0.0)
      {
	pos_c[p] /= ext_c[p];
	pos_c[p] -= 1;
      }
    else
      pos_c[p] = (double) pp - 1;
  
  Py_DECREF(x_arr);
  return Py_BuildValue("(N, N)", ext_arr, pos_arr);
}


/* Doc strings: */
static char module_doc[] = "";
static char borda_core_doc[] = "";

/* Method table */
static PyMethodDef borda_methods[] = {
  {"core",
   (PyCFunction)borda_core,
   METH_VARARGS | METH_KEYWORDS,
   borda_core_doc},
  {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "cborda",
  module_doc,
  -1,
  borda_methods,
  NULL, NULL, NULL, NULL
};

PyObject *PyInit_cborda(void)
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

PyMODINIT_FUNC initcborda(void)
{
  PyObject *m;
  
  m = Py_InitModule3("cborda", borda_methods, module_doc);
  if (m == NULL) {
    return;
  }
  
  import_array();
}

#endif
