/*
  fastcluster: Fast hierarchical clustering routines for R and Python

  Copyright © 2011 Daniel Müllner
  <http://math.stanford.edu/~muellner>
*/

// for INT32_MAX in fastcluster.cpp
// This must be defined here since Python.h loads the header file pyport.h,
//  and from this stdint.h. INT32_MAX is defined in stdint.h, but only if
// __STDC_LIMIT_MACROS is defined.
#define __STDC_LIMIT_MACROS

#include <Python.h>
#include <numpy/arrayobject.h>

#include <stddef.h>
#include "fastcluster.cpp"


// backwards compatibility
#ifndef NPY_ARRAY_CARRAY_RO
#define NPY_ARRAY_CARRAY_RO NPY_CARRAY_RO
#endif

/*
  Convenience class for the output array: automatic counter.
*/
class linkage_output {
private:
  t_float * Z;
  t_index pos;

public:
  linkage_output(t_float * const Z) {
    this->Z = Z;
    pos = 0;
  }

  void append(const t_index node1, const t_index node2, const t_float dist, const t_float size) {
    if (node1<node2) {
      Z[pos++] = static_cast<t_float>(node1);
      Z[pos++] = static_cast<t_float>(node2);
    }
    else {
      Z[pos++] = static_cast<t_float>(node2);
      Z[pos++] = static_cast<t_float>(node1);
    }
    Z[pos++] = dist;
    Z[pos++] = size;
  }
};

/*
  Generate the Scipy-specific output format for a dendrogram from the
  clustering output.

  The list of merging steps can be sorted or unsorted.
*/
// The size of a node is either 1 (a single point) or is looked up from
// one of the clusters.
#define size_(r_) ( ((r_<N) ? 1 : Z_(r_-N,3)) )

template <bool sorted>
static void generate_SciPy_dendrogram(t_float * const Z, cluster_result & Z2, const t_index N) {
  // The array "nodes" is a union-find data structure for the cluster
  // identites (only needed for unsorted cluster_result input).
  union_find nodes;
  if (!sorted) {
    std::stable_sort(Z2[0], Z2[N-1]);
    nodes.init(N);
  }

  linkage_output output(Z);
  t_index node1, node2;

  for (t_index i=0; i<N-1; i++) {
    // Get two data points whose clusters are merged in step i.
    if (sorted) {
      node1 = Z2[i]->node1;
      node2 = Z2[i]->node2;
    }
    else {
      // Find the cluster identifiers for these points.
      node1 = nodes.Find(Z2[i]->node1);
      node2 = nodes.Find(Z2[i]->node2);
      // Merge the nodes in the union-find data structure by making them
      // children of a new node.
      nodes.Union(node1, node2);
    }
    output.append(node1, node2, Z2[i]->dist, size_(node1)+size_(node2));
  }
}





// Tell Python about these methods.
/* 
PyMODINIT_FUNC init_fastcluster(void)  {
  (void) Py_InitModule("_fastcluster", _fastclusterWrapMethods);
  import_array();  // Must be present for NumPy. Called first after above line.
}
*/

/*
  Interface to Python, part 1:
  The input is a dissimilarity matrix.
*/
static PyObject *linkage_wrap(PyObject * const self, PyObject * const args) {
  PyArrayObject * D, * Z;
  long int N = 0;
  unsigned char method;

  try{
    // Parse the input arguments
    if (!PyArg_ParseTuple(args, "lO!O!b",
                          &N,                // signed long integer
                          &PyArray_Type, &D, // NumPy array
                          &PyArray_Type, &Z, // NumPy array
                          &method)) {        // unsigned char
      return NULL; // Error if the arguments have the wrong type.
    }
    if (N < 1 ) {
      // N must be at least 1.
      PyErr_SetString(PyExc_ValueError,
                      "At least one element is needed for clustering.");
      return NULL;
    }

    // (1)
    // The biggest index used below is 4*(N-2)+3, as an index to Z. This must fit
    // into the data type used for indices.
    // (2)
    // The largest representable integer, without loss of precision, by a floating
    // point number of type t_float is 2^T_FLOAT_MANT_DIG. Here, we make sure that
    // all cluster labels from 0 to 2N-2 in the output can be accurately represented
    // by a floating point number.
    if (N > MAX_INDEX/4 ||
        (N-1)>>(T_FLOAT_MANT_DIG-1) > 0) {
      PyErr_SetString(PyExc_ValueError,
                      "Data is too big, index overflow.");
      return NULL;
    }

    t_float * const D_ = reinterpret_cast<t_float *>(D->data);
    cluster_result Z2(N-1);
    auto_array_ptr<t_index> members;
    // For these methods, the distance update formula needs the number of
    // data points in a cluster.
    if (method==METHOD_METR_AVERAGE ||
        method==METHOD_METR_WARD ||
        method==METHOD_METR_CENTROID) {
      members.init(N, 1);
    }
    // Operate on squared distances for these methods.
    if (method==METHOD_METR_WARD ||
        method==METHOD_METR_CENTROID ||
        method==METHOD_METR_MEDIAN) {
      for (ptrdiff_t i=0; i < static_cast<ptrdiff_t>(N)*(N-1)/2; i++)
        D_[i] *= D_[i];
    }

    switch (method) {
    case METHOD_METR_SINGLE:
      MST_linkage_core(N, D_, Z2);
      break;
    case METHOD_METR_COMPLETE:
      NN_chain_core<METHOD_METR_COMPLETE, t_index>(N, D_, NULL, Z2);
      break;
    case METHOD_METR_AVERAGE:
      NN_chain_core<METHOD_METR_AVERAGE, t_index>(N, D_, members, Z2);
      break;
    case METHOD_METR_WEIGHTED:
      NN_chain_core<METHOD_METR_WEIGHTED, t_index>(N, D_, NULL, Z2);
      break;
    case METHOD_METR_WARD:
      NN_chain_core<METHOD_METR_WARD, t_index>(N, D_, members, Z2);
      break;
    case METHOD_METR_CENTROID:
      generic_linkage<METHOD_METR_CENTROID, t_index>(N, D_, members, Z2);
      break;
    case METHOD_METR_MEDIAN:
      generic_linkage<METHOD_METR_MEDIAN, t_index>(N, D_, NULL, Z2);
      break;
    default:
      PyErr_SetString(PyExc_IndexError, "Invalid method index.");
      return NULL;
    }

    if (method==METHOD_METR_WARD ||
        method==METHOD_METR_CENTROID ||
        method==METHOD_METR_MEDIAN) {
      Z2.sqrt();
    }

    t_float * const Z_ = reinterpret_cast<t_float *>(Z->data);
    if (method==METHOD_METR_CENTROID ||
        method==METHOD_METR_MEDIAN) {
      generate_SciPy_dendrogram<true>(Z_, Z2, N);
    }
    else {
      generate_SciPy_dendrogram<false>(Z_, Z2, N);
    }

  } // try
  catch (std::bad_alloc&) {
    return PyErr_NoMemory();
  }
  catch(std::exception& e){
    PyErr_SetString(PyExc_Exception, e.what());
    return NULL;
  }
  catch(...){
    PyErr_SetString(PyExc_Exception,
                    "C++ exception (unknown reason). Please send a bug report.");
    return NULL;
  }
  Py_RETURN_NONE;
}

/*
   Part 2: Clustering on vector data
*/

/*
   Helper class: Throw this if calling the Python interpreter from within
   C returned an error.
*/
class pythonerror {};

enum {
  // metrics
  METRIC_EUCLIDEAN       =  0,
  METRIC_MINKOWSKI       =  1,
  METRIC_CITYBLOCK       =  2,
  METRIC_SEUCLIDEAN      =  3,
  METRIC_SQEUCLIDEAN     =  4,
  METRIC_COSINE          =  5,
  METRIC_HAMMING         =  6,
  METRIC_JACCARD         =  7,
  METRIC_CHEBYCHEV       =  8,
  METRIC_CANBERRA        =  9,
  METRIC_BRAYCURTIS      = 10,
  METRIC_MAHALANOBIS     = 11,
  METRIC_YULE            = 12,
  METRIC_MATCHING        = 13,
  METRIC_DICE            = 14,
  METRIC_ROGERSTANIMOTO  = 15,
  METRIC_RUSSELLRAO      = 16,
  METRIC_SOKALSNEATH     = 17,
  METRIC_KULSINSKI       = 18,
  METRIC_USER            = 19,
  METRIC_INVALID         = 20, // sentinel
  METRIC_JACCARD_BOOL    = 21, // separate function for Jaccard metric on Boolean
};                             // input data

/*
  This class handles all the information about the dissimilarity
  computation.
*/

class python_dissimilarity {
private:
  t_float * Xa;
  auto_array_ptr<t_float> Xnew;
  ptrdiff_t dim; // size_t saves many statis_cast<> in products
  t_index N;
  t_index * members;
  void (cluster_result::*postprocessfn) (const t_float) const;
  t_float postprocessarg;

  t_float (python_dissimilarity::*distfn) (const t_index, const t_index) const;

  // for user-defined metrics
  PyObject * X_Python;
  PyObject * userfn;

  auto_array_ptr<t_float> precomputed;
  t_float * precomputed2;

  PyArrayObject * V;
  const t_float * V_data;

public:
  python_dissimilarity (PyArrayObject * const Xarg,
                        t_index * const members,
                        const unsigned char method,
                        const unsigned char metric,
                        PyObject * const extraarg,
                        bool temp_point_array)
    : Xa(reinterpret_cast<t_float *>(Xarg->data)),
      dim(Xarg->dimensions[1]),
      N(Xarg->dimensions[0]),
      members(members),
      postprocessfn(NULL),
      V(NULL)
  {
    switch (method) {
    case METHOD_METR_SINGLE:
      postprocessfn = NULL; // default
      switch (metric) {
      case METRIC_EUCLIDEAN:
        set_euclidean();
        break;
      case METRIC_SEUCLIDEAN:
        if (extraarg==NULL) {
          PyErr_SetString(PyExc_TypeError,
                          "The 'seuclidean' metric needs a variance parameter.");
          throw pythonerror();
        }
        V  = reinterpret_cast<PyArrayObject *>(PyArray_FromAny(extraarg,
                                               PyArray_DescrFromType(NPY_DOUBLE),
                                               1, 1,
                                               NPY_ARRAY_CARRAY_RO,
                                               NULL));
        if (PyErr_Occurred()) {
          throw pythonerror();
        }
        if (V->dimensions[0]!=dim) {
          PyErr_SetString(PyExc_ValueError,
          "The variance vector must have the same dimensionality as the data.");
          throw pythonerror();
        }
        V_data = reinterpret_cast<t_float *>(V->data);
        distfn = &python_dissimilarity::seuclidean;
        postprocessfn = &cluster_result::sqrt;
        break;
      case METRIC_SQEUCLIDEAN:
        distfn = &python_dissimilarity::sqeuclidean;
        break;
      case METRIC_CITYBLOCK:
        set_cityblock();
        break;
      case METRIC_CHEBYCHEV:
        set_chebychev();
        break;
      case METRIC_MINKOWSKI:
        set_minkowski(extraarg);
        break;
      case METRIC_COSINE:
        distfn = &python_dissimilarity::cosine;
        postprocessfn = &cluster_result::plusone;
        // precompute norms
        precomputed.init(N);
        for (t_index i=0; i<N; i++) {
          t_float sum=0;
          for (t_index k=0; k<dim; k++) {
            sum += X(i,k)*X(i,k);
          }
          precomputed[i] = 1/sqrt(sum);
        }
        break;
      case METRIC_HAMMING:
        distfn = &python_dissimilarity::hamming;
        postprocessfn = &cluster_result::divide;
        postprocessarg = static_cast<t_float>(dim);
        break;
      case METRIC_JACCARD:
        distfn = &python_dissimilarity::jaccard;
        break;
      case METRIC_CANBERRA:
        distfn = &python_dissimilarity::canberra;
        break;
      case METRIC_BRAYCURTIS:
        distfn = &python_dissimilarity::braycurtis;
        break;
      case METRIC_MAHALANOBIS:
        if (extraarg==NULL) {
          PyErr_SetString(PyExc_TypeError,
            "The 'mahalanobis' metric needs a parameter for the inverse covariance.");
          throw pythonerror();
        }
        V = reinterpret_cast<PyArrayObject *>(PyArray_FromAny(extraarg,
              PyArray_DescrFromType(NPY_DOUBLE),
              2, 2,
              NPY_ARRAY_CARRAY_RO,
              NULL));
        if (PyErr_Occurred()) {
          throw pythonerror();
        }
        if (V->dimensions[0]!=N || V->dimensions[1]!=dim) {
          PyErr_SetString(PyExc_ValueError,
            "The inverse covariance matrix has the wrong size.");
          throw pythonerror();
        }
        V_data = reinterpret_cast<t_float *>(V->data);
        distfn = &python_dissimilarity::mahalanobis;
        postprocessfn = &cluster_result::sqrt;
        break;
      case METRIC_YULE:
        distfn = &python_dissimilarity::yule;
        break;
      case METRIC_MATCHING:
        distfn = &python_dissimilarity::matching;
        postprocessfn = &cluster_result::divide;
        postprocessarg = static_cast<t_float>(dim);
        break;
      case METRIC_DICE:
        distfn = &python_dissimilarity::dice;
        break;
      case METRIC_ROGERSTANIMOTO:
        distfn = &python_dissimilarity::rogerstanimoto;
        break;
      case METRIC_RUSSELLRAO:
        distfn = &python_dissimilarity::russellrao;
        postprocessfn = &cluster_result::divide;
        postprocessarg = static_cast<t_float>(dim);
        break;
      case METRIC_SOKALSNEATH:
        distfn = &python_dissimilarity::sokalsneath;
        break;
      case METRIC_KULSINSKI:
        distfn = &python_dissimilarity::kulsinski;
        postprocessfn = &cluster_result::plusone;
        precomputed.init(N);
        for (t_index i=0; i<N; i++) {
          t_index sum=0;
          for (t_index k=0; k<dim; k++) {
            sum += Xb(i,k);
          }
          precomputed[i] = -.5/static_cast<t_float>(sum);
        }
        break;
      case METRIC_USER:
        X_Python = reinterpret_cast<PyObject *>(Xarg);
        this->userfn = extraarg;
        distfn = &python_dissimilarity::user;
        break;
      case METRIC_JACCARD_BOOL:
        distfn = &python_dissimilarity::jaccard_bool;
        break;
      default:
        throw 0;
      }
      break;

    case METHOD_METR_WARD:
      postprocessfn = &cluster_result::sqrtdouble;
      break;

    default:
      postprocessfn = &cluster_result::sqrt;
    }

    if (temp_point_array) {
      Xnew.init((N-1)*dim);
    }
  }

  ~python_dissimilarity() {
    Py_XDECREF(V);
  }

  inline t_float operator () (const t_index i, const t_index j) const {
    return (this->*distfn)(i,j);
  }

  inline t_float X (const t_index i, const t_index j) const {
    return Xa[i*dim+j];
  }

  inline bool Xb (const t_index i, const t_index j) const {
    return  reinterpret_cast<bool *>(Xa)[i*dim+j];
  }

  inline t_float * Xptr(const t_index i, const t_index j) const {
    return Xa+i*dim+j;
  }

  void merge(const t_index i, const t_index j, const t_index newnode) const {
    t_float const * Pi = i<N ? Xa+i*dim : Xnew+(i-N)*dim;
    t_float const * Pj = j<N ? Xa+j*dim : Xnew+(j-N)*dim;
    for(t_index k=0; k<dim; k++) {
      Xnew[(newnode-N)*dim+k] = (Pi[k]*static_cast<t_float>(members[i]) +
                             Pj[k]*static_cast<t_float>(members[j])) /
        static_cast<t_float>(members[i]+members[j]);
    }
    members[newnode] = members[i]+members[j];
  }

  void merge_weighted(const t_index i, const t_index j, const t_index newnode) const {
    t_float const * Pi = i<N ? Xa+i*dim : Xnew+(i-N)*dim;
    t_float const * Pj = j<N ? Xa+j*dim : Xnew+(j-N)*dim;
    for(t_index k=0; k<dim; k++) {
      Xnew[(newnode-N)*dim+k] = (Pi[k]+Pj[k])/2.;
    }
  }

  void postprocess(cluster_result & Z2) const {
    if (postprocessfn!=NULL) {
        (Z2.*postprocessfn)(postprocessarg);
    }
  }

  inline t_float ward(const t_index i, const t_index j) const {
    t_float mi = static_cast<t_float>(members[i]);
    t_float mj = static_cast<t_float>(members[j]);
    return sqeuclidean(i,j)*mi*mj/(mi+mj);
  }

  inline t_float ward_extended(const t_index i, const t_index j) const {
    t_float mi = static_cast<t_float>(members[i]);
    t_float mj = static_cast<t_float>(members[j]);
    return sqeuclidean_extended(i,j)*mi*mj/(mi+mj);
  }

  t_float sqeuclidean(const t_index i, const t_index j) const {
    t_float sum = 0;
    /*
    for (t_index k=0; k<dim; k++) {
        t_float diff = X(i,k) - X(j,k);
        sum += diff*diff;
    }
    */
    // faster
    t_float const * Pi = Xa+i*dim;
    t_float const * Pj = Xa+j*dim;
    for (t_index k=0; k<dim; k++) {
      t_float diff = Pi[k] - Pj[k];
      sum += diff*diff;
    }
    return sum;
  }

  t_float sqeuclidean_extended(const t_index i, const t_index j) const {
    t_float sum = 0;
    t_float const * Pi = i<N ? Xa+i*dim : Xnew+(i-N)*dim; // TBD
    t_float const * Pj = j<N ? Xa+j*dim : Xnew+(j-N)*dim;
    for (t_index k=0; k<dim; k++) {
      t_float diff = Pi[k] - Pj[k];
      sum += diff*diff;
    }
    return sum;
  }

private:
  void set_minkowski(PyObject * extraarg) {
    if (extraarg==NULL) {
      PyErr_SetString(PyExc_TypeError,
                      "The Minkowski metric needs a parameter.");
      throw pythonerror();
    }
    postprocessarg = PyFloat_AsDouble(extraarg);
    if (PyErr_Occurred()) {
      throw pythonerror();
    }

    if (postprocessarg==std::numeric_limits<t_float>::infinity()) {
      set_chebychev();
    }
    else if (postprocessarg==1.0){
      set_cityblock();
    }
    else if (postprocessarg==2.0){
      set_euclidean();
    }
    else {
      distfn = &python_dissimilarity::minkowski;
      postprocessfn = &cluster_result::power;
    }
  }

  void set_euclidean() {
    distfn = &python_dissimilarity::sqeuclidean;
    postprocessfn = &cluster_result::sqrt;
  }

  void set_cityblock() {
    distfn = &python_dissimilarity::cityblock;
  }

  void set_chebychev() {
    distfn = &python_dissimilarity::chebychev;
  }

  t_float seuclidean(const t_index i, const t_index j) const {
    t_float sum = 0;
    for (t_index k=0; k<dim; k++) {
      t_float diff = X(i,k)-X(j,k);
      sum += diff*diff/V_data[k];
    }
    return sum;
  }

  t_float cityblock(const t_index i, const t_index j) const {
    t_float sum = 0;
    for (t_index k=0; k<dim; k++) {
      sum += fabs(X(i,k)-X(j,k));
    }
    return sum;
  }

  t_float minkowski(const t_index i, const t_index j) const {
    t_float sum = 0;
    for (t_index k=0; k<dim; k++) {
      sum += pow(fabs(X(i,k)-X(j,k)),postprocessarg);
    }
    return sum;
  }

  t_float chebychev(const t_index i, const t_index j) const {
    t_float max = 0;
    for (t_index k=0; k<dim; k++) {
      t_float diff = fabs(X(i,k)-X(j,k));
      if (diff>max) {
        max = diff;
      }
    }
    return max;
  }

  t_float cosine(const t_index i, const t_index j) const {
    t_float sum = 0;
    for (t_index k=0; k<dim; k++) {
      sum -= X(i,k)*X(j,k);
    }
    return sum*precomputed[i]*precomputed[j];
  }

  t_float hamming(const t_index i, const t_index j) const {
    t_float sum = 0;
    for (t_index k=0; k<dim; k++) {
      sum += (X(i,k)!=X(j,k));
    }
    return sum;
  }

  // Differs from scipy.spatial.distance: equal vectors correctly
  // return distance 0.
  t_float jaccard(const t_index i, const t_index j) const {
    t_index sum1 = 0;
    t_index sum2 = 0;
    for (t_index k=0; k<dim; k++) {
      sum1 += (X(i,k)!=X(j,k));
      sum2 += ((X(i,k)!=0) || (X(j,k)!=0));
    }
    return sum1==0 ? 0 : static_cast<t_float>(sum1) / static_cast<t_float>(sum2);
  }

  t_float canberra(const t_index i, const t_index j) const {
    t_float sum = 0;
    for (t_index k=0; k<dim; k++) {
      t_float numerator = fabs(X(i,k)-X(j,k));
      sum += numerator==0 ? 0 : numerator / (fabs(X(i,k)) + fabs(X(j,k)));
    }
    return sum;
  }

  t_float user(const t_index i, const t_index j) const {
    PyObject * u = PySequence_ITEM(X_Python, i);
    PyObject * v = PySequence_ITEM(X_Python, j);
    PyObject * result = PyObject_CallFunctionObjArgs(userfn, u, v, NULL);
    Py_DECREF(u);
    Py_DECREF(v);
    if (result==NULL) {
      throw pythonerror();
    }
    const t_float C_result = PyFloat_AsDouble(result);
    Py_DECREF(result);
    if (PyErr_Occurred()) {
      throw pythonerror();
    }
    return C_result;
  }

  t_float braycurtis(const t_index i, const t_index j) const {
    t_float sum1 = 0;
    t_float sum2 = 0;
    for (t_index k=0; k<dim; k++) {
      sum1 += fabs(X(i,k)-X(j,k));
      sum2 += fabs(X(i,k)+X(j,k));
    }
    return sum1/sum2;
  }

  t_float mahalanobis(const t_index i, const t_index j) const {
    // V_data contains the product X*VI
    t_float sum = 0;
    for (t_index k=0; k<dim; k++) {
      sum += (V_data[i*dim+k]-V_data[j*dim+k])*(X(i,k)-X(j,k));
    }
    return sum;
  }

  t_index mutable NTT; // 'local' variables
  t_index mutable NXO;
  t_index mutable NTF;
  #define NTFFT NTF
  #define NFFTT NTT

  void nbool_correspond(const t_index i, const t_index j) const {
    NTT = 0;
    NXO = 0;
    for (t_index k=0; k<dim; k++) {
      NTT += (Xb(i,k) &  Xb(j,k)) ;
      NXO += (Xb(i,k) ^  Xb(j,k)) ;
    }
  }

  void nbool_correspond_tfft(const t_index i, const t_index j) const {
    NTT = 0;
    NXO = 0;
    NTF = 0;
    for (t_index k=0; k<dim; k++) {
      NTT += (Xb(i,k) &  Xb(j,k)) ;
      NXO += (Xb(i,k) ^  Xb(j,k)) ;
      NTF += (Xb(i,k) & ~Xb(j,k)) ;
    }
    NTF *= (NXO-NTF); // NTFFT
    NTT *= (dim-NTT-NXO); // NFFTT
  }

  void nbool_correspond_xo(const t_index i, const t_index j) const {
    NXO = 0;
    for (t_index k=0; k<dim; k++) {
      NXO += (Xb(i,k) ^  Xb(j,k)) ;
    }
  }

  void nbool_correspond_tt(const t_index i, const t_index j) const {
    NTT = 0;
    for (t_index k=0; k<dim; k++) {
      NTT += (Xb(i,k) &  Xb(j,k)) ;
    }
  }

  // Caution: zero denominators can happen here!
  t_float yule(const t_index i, const t_index j) const {
    nbool_correspond_tfft(i, j);
    return static_cast<t_float>(2*NTFFT) / static_cast<t_float>(NTFFT + NFFTT);
  }

  // Prevent a zero denominator for equal vectors.
  t_float dice(const t_index i, const t_index j) const {
    nbool_correspond(i, j);
    return (NXO==0) ? 0 :
      static_cast<t_float>(NXO) / static_cast<t_float>(NXO+2*NTT);
  }

  t_float rogerstanimoto(const t_index i, const t_index j) const {
    nbool_correspond_xo(i, j);
    return static_cast<t_float>(2*NXO) / static_cast<t_float>(NXO+dim);
  }

  t_float russellrao(const t_index i, const t_index j) const {
    nbool_correspond_tt(i, j);
    return static_cast<t_float>(dim-NTT);
  }

  // Prevent a zero denominator for equal vectors.
  t_float sokalsneath(const t_index i, const t_index j) const {
    nbool_correspond(i, j);
    return (NXO==0) ? 0 :
      static_cast<t_float>(2*NXO) / static_cast<t_float>(NTT+2*NXO);
  }

  t_float kulsinski(const t_index i, const t_index j) const {
    nbool_correspond_tt(i, j);
    return static_cast<t_float>(NTT) * (precomputed[i] + precomputed[j]);
  }

  // 'matching' distance = Hamming distance
  t_float matching(const t_index i, const t_index j) const {
    nbool_correspond_xo(i, j);
    return static_cast<t_float>(NXO);
  }

  // Prevent a zero denominator for equal vectors.
  t_float jaccard_bool(const t_index i, const t_index j) const {
    nbool_correspond(i, j);
    return (NXO==0) ? 0 :
      static_cast<t_float>(NXO) / static_cast<t_float>(NXO+NTT);
  }
};

static PyObject *linkage_vector_wrap(PyObject * const self, PyObject * const args) {
  PyArrayObject * X, * Z;
  unsigned char method, metric;
  PyObject * extraarg;

  try{
    // Parse the input arguments
    if (!PyArg_ParseTuple(args, "O!O!bbO",
                          &PyArray_Type, &X, // NumPy array
                          &PyArray_Type, &Z, // NumPy array
                          &method,           // unsigned char
                          &metric,           // unsigned char
                          &extraarg )) {     // Python object
      throw pythonerror(); // Error if the arguments have the wrong type.
    }

    if (X->nd != 2) {
      PyErr_SetString(PyExc_ValueError,
                      "The input array must be two-dimensional.");
    }
    npy_intp const N = X->dimensions[0];
    if (N < 1 ) {
      // N must be at least 1.
      PyErr_SetString(PyExc_ValueError,
                      "At least one element is needed for clustering.");
      throw pythonerror();
    }

    npy_intp const dim = X->dimensions[1];
    if (dim < 1 ) {
      PyErr_SetString(PyExc_ValueError,
                      "Invalid dimension of the data set.");
      throw pythonerror();
    }

    // (1)
    // The biggest index used below is 4*(N-2)+3, as an index to Z. This must fit
    // into the data type used for indices.
    // (2)
    // The largest representable integer, without loss of precision, by a floating
    // point number of type t_float is 2^T_FLOAT_MANT_DIG. Here, we make sure that
    // all cluster labels from 0 to 2N-2 in the output can be accurately represented
    // by a floating point number.
    if (N > MAX_INDEX/4 ||
        (N-1)>>(T_FLOAT_MANT_DIG-1) > 0) {
      PyErr_SetString(PyExc_ValueError,
                      "Data is too big, index overflow.");
      throw pythonerror();
    }

    cluster_result Z2(N-1);

    auto_array_ptr<t_index> members;
    if (method==METHOD_METR_WARD || method==METHOD_METR_CENTROID) {
      members.init(2*N-1, 1);
    }

    if ((method!=METHOD_METR_SINGLE && metric!=METRIC_EUCLIDEAN) ||
        metric>=METRIC_INVALID) {
      PyErr_SetString(PyExc_IndexError, "Invalid metric index.");
      throw pythonerror();
    }

    if (PyArray_ISBOOL(X)) {
      if (metric==METRIC_HAMMING) {
        metric = METRIC_MATCHING; // Alias
      }
      if (metric==METRIC_JACCARD) {
        metric = METRIC_JACCARD_BOOL;
      }
    }

    if (extraarg!=Py_None &&
        metric!=METRIC_MINKOWSKI &&
        metric!=METRIC_SEUCLIDEAN &&
        metric!=METRIC_MAHALANOBIS &&
        metric!=METRIC_USER) {
      PyErr_SetString(PyExc_TypeError,
                      "No extra parameter is allowed for this metric.");
      throw pythonerror();
    }

    bool temp_point_array = (method!=METHOD_METR_SINGLE);

    python_dissimilarity dist(X, members, method, metric, extraarg,
                              temp_point_array);

    switch (method) {
    case METHOD_METR_SINGLE:
      MST_linkage_core_vector(N, dist, Z2);
      break;
    case METHOD_METR_WARD:
      generic_linkage_vector<METHOD_METR_WARD>(N, dist, Z2);
      break;
    case METHOD_METR_CENTROID:
      generic_linkage_vector<METHOD_METR_CENTROID>(N, dist, Z2);
      break;
    case METHOD_METR_MEDIAN:
      generic_linkage_vector<METHOD_METR_MEDIAN>(N, dist, Z2);
      break;
    default:
      PyErr_SetString(PyExc_IndexError, "Invalid method index.");
      throw pythonerror();
    }

    if (method==METHOD_METR_WARD ||
        method==METHOD_METR_CENTROID) {
      members.free();
    }

    dist.postprocess(Z2);

    t_float * const Z_ = reinterpret_cast<t_float *>(Z->data);
    if (method!=METHOD_METR_SINGLE) {
      generate_SciPy_dendrogram<true>(Z_, Z2, N);
    }
    else {
      generate_SciPy_dendrogram<false>(Z_, Z2, N);
    }

  } // try
  catch (std::bad_alloc&) {
    return PyErr_NoMemory();
  }
  catch(std::exception& e){
    PyErr_SetString(PyExc_Exception, e.what());
    return NULL;
  }
  catch(pythonerror){
    return NULL;
  }
  catch(...){
    PyErr_SetString(PyExc_Exception,
                    "C++ exception (unknown reason). Please send a bug report.");
    return NULL;
  }
  Py_RETURN_NONE;
}


#ifdef __cplusplus
extern "C" {
#endif 

// List the C++ methods that this extension provides.
static PyMethodDef fastclusterWrapMethods[] = {
  {"linkage_wrap",
   (PyCFunction)linkage_wrap,
   METH_VARARGS,
   ""},
  {"linkage_vector_wrap", 
   (PyCFunction)linkage_vector_wrap, 
   METH_VARARGS,
   ""},
  {NULL, NULL, 0, NULL}     /* Sentinel - marks the end of this structure */
};


#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "_fastcluster",
  "",
  -1,
  fastclusterWrapMethods,
  NULL, NULL, NULL, NULL
};

PyObject *PyInit__fastcluster(void)
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

PyMODINIT_FUNC init_fastcluster(void)
{
  PyObject *m;
  
  m = Py_InitModule3("_fastcluster", fastclusterWrapMethods, "");
  if (m == NULL) {
    return;
  }
  
  import_array();
}

#endif

#ifdef __cplusplus
}  // extern "C"
#endif 
