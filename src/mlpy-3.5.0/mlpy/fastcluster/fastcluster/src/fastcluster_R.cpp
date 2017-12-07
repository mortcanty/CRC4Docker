/*
  fastcluster: Fast hierarchical clustering routines for R and Python

  Copyright © 2011 Daniel Müllner
  <http://math.stanford.edu/~muellner>
*/
#include <Rdefines.h>
#include <R_ext/Rdynload.h>
#include <Rmath.h> // for R_pow

#include "fastcluster.cpp"

/*
  Helper function: order the nodes so that they can be displayed nicely
  in a dendrogram.

  This is used for the 'order' field in the R output.
*/

struct node_pos {
  int node;
  t_index pos;
};

void order_nodes(const int N, const int * const merge, const t_index * const node_size, int * const order) {
  auto_array_ptr<node_pos> queue(N/2);

  int parent;
  int child;
  t_index pos = 0;

  queue[0].node = N-2;
  queue[0].pos = 0;
  t_index idx = 1;

  do {
    idx --;
    parent = queue[idx].node;
    pos = queue[idx].pos;

    // First child
    child = merge[parent];
    if (child<0) { // singleton node, write this into the 'order' array
      order[pos] = -child;
      pos++;
    }
    else {
      queue[idx].node = child-1;
      queue[idx].pos = pos;
      idx++;
      pos += node_size[child-1];
    }
    // Second child
    child = merge[parent+N-1];
    if (child<0) {
      order[pos] = -child;
    }
    else {
      queue[idx].node = child-1;
      queue[idx].pos = pos;
      idx++;
    }
  } while (idx>0);
}

#define size_(r_) ( ((r_<N) ? 1 : node_size[r_-N]) )

template <bool sorted>
void generate_R_dendrogram(int * const merge, double * const height, int * const order, cluster_result & Z2, const int N) {
  // The array "nodes" is a union-find data structure for the cluster
  // identites (only needed for unsorted cluster_result input).
  union_find nodes;

  if (!sorted) {
    std::stable_sort(Z2[0], Z2[N-1]);
    nodes.init(N);
  }

  t_index node1, node2;
  auto_array_ptr<t_index> node_size(N-1);

  for (t_index i=0; i<N-1; i++) {
    // Get two data points whose clusters are merged in step i.
    // Find the cluster identifiers for these points.
    if (sorted) {
      node1 = Z2[i]->node1;
      node2 = Z2[i]->node2;
    }
    else {
      node1 = nodes.Find(Z2[i]->node1);
      node2 = nodes.Find(Z2[i]->node2);
      // Merge the nodes in the union-find data structure by making them
      // children of a new node.
      nodes.Union(node1, node2);
    }
    // Sort the nodes in the output array.
    if (node1>node2) {
      t_index tmp = node1;
      node1 = node2;
      node2 = tmp;
    }
    merge[i]     = (node1<N) ? -static_cast<int>(node1)-1
                              : static_cast<int>(node1)-N+1;
    merge[i+N-1] = (node2<N) ? -static_cast<int>(node2)-1
                              : static_cast<int>(node2)-N+1;
    height[i] = Z2[i]->dist;
    node_size[i] = size_(node1) + size_(node2);
  }

  order_nodes(N, merge, node_size, order);
}

/*
  R interface code
*/

enum {
  METRIC_R_EUCLIDEAN = 0,
  METRIC_R_MAXIMUM   = 1,
  METRIC_R_MANHATTAN = 2,
  METRIC_R_CANBERRA  = 3,
  METRIC_R_BINARY    = 4,
  METRIC_R_MINKOWSKI = 5
};

class R_dissimilarity {
private:
  t_float * Xa;
  ptrdiff_t dim; // ptrdiff_t saves many statis_cast<> in products
  int * members;
  void (cluster_result::*postprocessfn) (const t_float) const;
  t_float postprocessarg;

  t_float (R_dissimilarity::*distfn) (const t_index, const t_index) const;
  int N;

  auto_array_ptr<t_index> row_repr;

  // derived constructor
public:
  R_dissimilarity (t_float * const X,
                   const int N,
                   const int dim,
                   int * const members,
                   const unsigned char method,
                   const unsigned char metric,
                   const t_float p,
                   bool make_row_repr)
    : Xa(X),
      dim(dim),
      members(members),
      postprocessfn(NULL),
      postprocessarg(p),
      N(N)
  {
    switch (method) {
    case METHOD_VECTOR_SINGLE:
      switch (metric) {
      case METRIC_R_EUCLIDEAN:
        distfn = &R_dissimilarity::sqeuclidean;
        postprocessfn = &cluster_result::sqrt;
        break;
      case METRIC_R_MAXIMUM:
        distfn = &R_dissimilarity::maximum;
        break;
      case METRIC_R_MANHATTAN:
        distfn = &R_dissimilarity::manhattan;
        break;
      case METRIC_R_CANBERRA:
        distfn = &R_dissimilarity::canberra;
        break;
      case METRIC_R_BINARY:
        distfn = &R_dissimilarity::dist_binary;
        break;
      case METRIC_R_MINKOWSKI:
        distfn = &R_dissimilarity::minkowski;
        postprocessfn = &cluster_result::power;
        break;
      default:
        throw 0;
      }
      break;

    case METHOD_VECTOR_WARD:
      postprocessfn = &cluster_result::sqrtdouble;
      break;

    default:
      postprocessfn = &cluster_result::sqrt;
    }

    if (make_row_repr) {
      row_repr.init(2*N-1);
      for (t_index i=0; i<N; i++) {
        row_repr[i] = i;
      }
    }

  }

  inline t_float operator () (const t_index i, const t_index j) const {
    return (this->*distfn)(i,j);
  }

  inline t_float X (const t_index i, const t_index j) const {
    // "C-style" array alignment
    return Xa[i*dim+j];
  }

  inline t_float * Xptr(const t_index i, const t_index j) const {
    // "C-style" array alignment
    return Xa+i*dim+j;
  }

  void merge2(const t_index i, const t_index j) const {
    for(t_index k=0; k<dim; k++) {
      *(Xptr(j,k)) = (X(i,k)*static_cast<t_float>(members[i]) +
                      X(j,k)*static_cast<t_float>(members[j])) /
        static_cast<t_float>(members[i]+members[j]);
    }
    members[j] += members[i];
  }

  void merge(const t_index i, const t_index j, const t_index newnode) const {
    merge2(row_repr[i], row_repr[j]);
    row_repr[newnode] = row_repr[j];
  }

  void merge_weighted(const t_index i, const t_index j, const t_index newnode) const {
    t_float * const Pi = Xa+row_repr[i]*dim;
    t_float * const Pj = Xa+row_repr[j]*dim;
    for(t_index k=0; k<dim; k++) {
      Pj[k] = (Pi[k]+Pj[k])/2.;
    }
    row_repr[newnode] =row_repr[j];
  }

  void postprocess(cluster_result & Z2) const {
    if (postprocessfn!=NULL) {
        (Z2.*postprocessfn)(postprocessarg);
    }
  }

  double ward(t_index const i1, t_index const i2) const {
    t_float const mi = static_cast<t_float>(members[i1]);
    t_float const mj = static_cast<t_float>(members[i2]);
    return sqeuclidean(i1,i2)*mi*mj/(mi+mj);
  }

  double ward_extended(t_index i1, t_index i2) const {
    return ward(row_repr[i1], row_repr[i2]);
  }

  /*
    The following definitions and methods are taken directly from
    the R source file

      /src/library/stats/src/distance.c

    in the R release 2.13.0. The code has only been adapted very slightly.
    (Unfortunately, the methods cannot be called directly in the R libraries
    since the functions are declared "static" in the above file.)

    Note to maintainers: If the code in distance.c changes in future R releases
    compared to 2.13.0, please update the definitions here, if necessary.
  */

  // translation of variable names
  #define nc dim
  #define nr N
  #define x Xa
  #define p postprocessarg

  // The code from distance.c starts here

  #define both_FINITE(a,b) (R_FINITE(a) && R_FINITE(b))
  #ifdef R_160_and_older
  #define both_non_NA both_FINITE
  #else
  #define both_non_NA(a,b) (!ISNAN(a) && !ISNAN(b))
  #endif

  // still public
  double sqeuclidean(t_index const i1, t_index const i2) const {
    double dev, dist;
    int count, j;

    count = 0;
    dist = 0;
    double * p1 = x+i1*nc;
    double * p2 = x+i2*nc;
    for(j = 0 ; j < nc ; j++) {
      if(both_non_NA(*p1, *p2)) {
        dev = (*p1 - *p2);
        if(!ISNAN(dev)) {
          dist += dev * dev;
          count++;
        }
      }
      p1++;
      p2++;
    }
    if(count == 0) return NA_REAL;
    if(count != nc) dist /= (static_cast<double>(count)/static_cast<double>(nc));
    //return sqrt(dist);
    // we take the square root later
    return dist;
  }

  inline double sqeuclidean_extended(t_index const i1, t_index const i2) const {
    return sqeuclidean(row_repr[i1], row_repr[i2]);
  }

private:
  double maximum(t_index i1, t_index i2) const {
    double dev, dist;
    int count, j;

    count = 0;
    dist = -DBL_MAX;
    double * p1 = x+i1*nc;
    double * p2 = x+i2*nc;
    for(j = 0 ; j < nc ; j++) {
      if(both_non_NA(*p1, *p2)) {
        dev = fabs(*p1 - *p2);
        if(!ISNAN(dev)) {
          if(dev > dist)
            dist = dev;
          count++;
        }
      }
      p1++;
      p2++;
    }
    if(count == 0) return NA_REAL;
    return dist;
  }

  double manhattan(t_index i1, t_index i2) const {
    double dev, dist;
    int count, j;

    count = 0;
    dist = 0;
    double * p1 = x+i1*nc;
    double * p2 = x+i2*nc;
    for(j = 0 ; j < nc ; j++) {
      if(both_non_NA(*p1, *p2)) {
        dev = fabs(*p1 - *p2);
        if(!ISNAN(dev)) {
          dist += dev;
          count++;
        }
      }
      p1++;
      p2++;
    }
    if(count == 0) return NA_REAL;
    if(count != nc) dist /= (static_cast<double>(count)/static_cast<double>(nc));
    return dist;
  }

  double canberra(t_index i1, t_index i2) const {
    double dev, dist, sum, diff;
    int count, j;

    count = 0;
    dist = 0;
    double * p1 = x+i1*nc;
    double * p2 = x+i2*nc;
    for(j = 0 ; j < nc ; j++) {
      if(both_non_NA(*p1, *p2)) {
        sum = fabs(*p1 + *p2);
        diff = fabs(*p1 - *p2);
        if (sum > DBL_MIN || diff > DBL_MIN) {
          dev = diff/sum;
          if(!ISNAN(dev) ||
             (!R_FINITE(diff) && diff == sum &&
              /* use Inf = lim x -> oo */ (dev = 1.))) {
            dist += dev;
            count++;
          }
        }
      }
      p1++;
      p2++;
    }
    if(count == 0) return NA_REAL;
    if(count != nc) dist /= (static_cast<double>(count)/static_cast<double>(nc));
    return dist;
  }

  double dist_binary(t_index i1, t_index i2) const {
    int total, count, dist;
    int j;

    total = 0;
    count = 0;
    dist = 0;
    double * p1 = x+i1*nc;
    double * p2 = x+i2*nc;
    for(j = 0 ; j < nc ; j++) {
      if(both_non_NA(*p1, *p2)) {
        if(!both_FINITE(*p1, *p2)) {
          //          warning(_("treating non-finite values as NA"));
        }
        else {
          if(*p1 || *p2) {
            count++;
            if( ! (*p1 && *p2) ) {
              dist++;
            }
          }
          total++;
        }
      }
      p1++;
      p2++;
    }

    if(total == 0) return NA_REAL;
    if(count == 0) return 0;
    return static_cast<double>(dist) / static_cast<double>(count);
  }

  double minkowski(t_index i1, t_index i2) const {
    double dev, dist;
    int count, j;

    count= 0;
    dist = 0;
    double * p1 = x+i1*nc;
    double * p2 = x+i2*nc;
    for(j = 0 ; j < nc ; j++) {
      if(both_non_NA(*p1, *p2)) {
        dev = (*p1 - *p2);
        if(!ISNAN(dev)) {
          dist += R_pow(fabs(dev), p);
          count++;
        }
      }
      p1++;
      p2++;
    }
    if(count == 0) return NA_REAL;
    if(count != nc) dist /= (static_cast<double>(count)/static_cast<double>(nc));
    //return R_pow(dist, 1.0/p);
    // raise to the (1/p)-th power later
    return dist;
  }

};

extern "C" {
  SEXP fastcluster(SEXP const N_, SEXP const method_, SEXP D_, SEXP members_) {
    SEXP r = NULL; // return value

    try{
      /*
        Input checks
      */
      // Parameter N: number of data points
      PROTECT(N_);
      if (!IS_INTEGER(N_) || LENGTH(N_)!=1)
        Rf_error("'N' must be a single integer.");
      const int N = *INTEGER_POINTER(N_);
      if (N<2)
        Rf_error("N must be at least 2.");
      const ptrdiff_t NN = static_cast<ptrdiff_t>(N)*(N-1)/2;
      UNPROTECT(1); // N_

      // Parameter method: dissimilarity index update method
      PROTECT(method_);
      if (!IS_INTEGER(method_) || LENGTH(method_)!=1)
        Rf_error("'method' must be a single integer.");
      const int method = *INTEGER_POINTER(method_) - 1; // index-0 based;
      if (method<METHOD_METR_SINGLE || method>METHOD_METR_MEDIAN) {
        Rf_error("Invalid method index.");
      }
      UNPROTECT(1); // method_

      // Parameter members: number of members in each node
      auto_array_ptr<int> members;
      if (method==METHOD_METR_AVERAGE ||
          method==METHOD_METR_WARD ||
          method==METHOD_METR_CENTROID) {
        members.init(N);
        if (Rf_isNull(members_)) {
          for (t_index i=0; i<N; i++) members[i] = 1;
        }
        else {
          PROTECT(members_ = AS_INTEGER(members_));
          if (LENGTH(members_)!=N)
            Rf_error("'members' must have length N.");
          const int * const m = INTEGER_POINTER(members_);
          for (t_index i=0; i<N; i++) members[i] = m[i];
          UNPROTECT(1); // members
        }
      }

      // Parameter D_: dissimilarity matrix
      PROTECT(D_ = AS_NUMERIC(D_));
      if (LENGTH(D_)!=NN)
        Rf_error("'D' must have length (N \\choose 2).");
      const double * const D = NUMERIC_POINTER(D_);
      // Make a working copy of the dissimilarity array
      // for all methods except "single".
      auto_array_ptr<double> D__;
      if (method!=METHOD_METR_SINGLE) {
        D__.init(NN);
        for (ptrdiff_t i=0; i<NN; i++)
          D__[i] = D[i];
      }
      UNPROTECT(1); // D_

      /*
        Clustering step
      */
      cluster_result Z2(N-1);
      switch (method) {
      case METHOD_METR_SINGLE:
        MST_linkage_core(N, D, Z2);
        break;
      case METHOD_METR_COMPLETE:
        NN_chain_core<METHOD_METR_COMPLETE, int>(N, D__, NULL, Z2);
        break;
      case METHOD_METR_AVERAGE:
        NN_chain_core<METHOD_METR_AVERAGE, int>(N, D__, members, Z2);
        break;
      case METHOD_METR_WEIGHTED:
        NN_chain_core<METHOD_METR_WEIGHTED, int>(N, D__, NULL, Z2);
        break;
      case METHOD_METR_WARD:
        NN_chain_core<METHOD_METR_WARD, int>(N, D__, members, Z2);
        break;
      case METHOD_METR_CENTROID:
        generic_linkage<METHOD_METR_CENTROID, int>(N, D__, members, Z2);
        break;
      case METHOD_METR_MEDIAN:
        generic_linkage<METHOD_METR_MEDIAN, int>(N, D__, NULL, Z2);
        break;
      }

      D__.free();     // Free the memory now
      members.free(); // (not strictly necessary).

      SEXP m; // return field "merge"
      PROTECT(m = NEW_INTEGER(2*(N-1)));
      int * const merge = INTEGER_POINTER(m);

      SEXP dim_m; // Specify that m is an (N-1)×2 matrix
      PROTECT(dim_m = NEW_INTEGER(2));
      INTEGER(dim_m)[0] = N-1;
      INTEGER(dim_m)[1] = 2;
      SET_DIM(m, dim_m);

      SEXP h; // return field "height"
      PROTECT(h = NEW_NUMERIC(N-1));
      double * const height = NUMERIC_POINTER(h);

      SEXP o; // return fiels "order'
      PROTECT(o = NEW_INTEGER(N));
      int * const order = INTEGER_POINTER(o);

      if (method==METHOD_METR_CENTROID ||
          method==METHOD_METR_MEDIAN)
        generate_R_dendrogram<true>(merge, height, order, Z2, N);
      else
        generate_R_dendrogram<false>(merge, height, order, Z2, N);

      SEXP n; // names
      PROTECT(n = NEW_CHARACTER(3));
      SET_STRING_ELT(n, 0, COPY_TO_USER_STRING("merge"));
      SET_STRING_ELT(n, 1, COPY_TO_USER_STRING("height"));
      SET_STRING_ELT(n, 2, COPY_TO_USER_STRING("order"));

      PROTECT(r = NEW_LIST(3)); // field names in the output list
      SET_ELEMENT(r, 0, m);
      SET_ELEMENT(r, 1, h);
      SET_ELEMENT(r, 2, o);
      SET_NAMES(r, n);

      UNPROTECT(6); // m, dim_m, h, o, r, n
    } // try
    catch (std::bad_alloc&) {
      Rf_error( "Memory overflow.");
    }
    catch(std::exception& e){
      Rf_error( e.what() );
    }
    catch(...){
      Rf_error( "C++ exception (unknown reason)." );
    }

    return r;
  }

  SEXP fastcluster_vector(SEXP const method_,
                          SEXP const metric_,
                          SEXP X_,
                          SEXP members_,
                          SEXP p_) {
    SEXP r = NULL; // return value
    try{

      /*
        Input checks
      */

      // Parameter method: dissimilarity index update method
      PROTECT(method_);
      if (!IS_INTEGER(method_) || LENGTH(method_)!=1)
        Rf_error("'method' must be a single integer.");
      int method = *INTEGER_POINTER(method_) - 1; // index-0 based;
      if (method<METHOD_VECTOR_SINGLE || method>METHOD_VECTOR_MEDIAN) {
        Rf_error("Invalid method index.");
      }
      UNPROTECT(1); // method_

      // Parameter metric
      PROTECT(metric_);
      if (!IS_INTEGER(metric_) || LENGTH(metric_)!=1)
        Rf_error("'metric' must be a single integer.");
      int metric = *INTEGER_POINTER(metric_) - 1; // index-0 based;
      if (metric<0 || metric>5 ||
          (method!=METHOD_VECTOR_SINGLE && metric!=0) ) {
        Rf_error("Invalid metric index.");
      }
      UNPROTECT(1); // metric_

      // data array
      PROTECT(X_ = AS_NUMERIC(X_));
      SEXP dims_ = PROTECT( Rf_getAttrib( X_, R_DimSymbol ) ) ;
      if( dims_ == R_NilValue || LENGTH(dims_) != 2 ) {
        Rf_error( "Argument is not a matrix.");
      }
      const int * const dims = INTEGER(dims_);
      const int N = dims[0];
      const int dim = dims[1];
      if (N<2)
        Rf_error("There must be at least two data points.");
      // Make a working copy of the dissimilarity array
      // for all methods except "single".
      double * X__ = NUMERIC_POINTER(X_);
      // Copy the input array and change it from Fortran-contiguous  style
      // to C-contiguous style
      // (Waste of memory for 'single'; the other methods need a copy
      auto_array_ptr<double> X(LENGTH(X_));
      for (ptrdiff_t i=0; i<N; i++)
        for (ptrdiff_t j=0; j<dim; j++)
          X[i*dim+j] = X__[i+j*N];

      UNPROTECT(2); // dims_, X_

      // Parameter members: number of members in each node
      auto_array_ptr<int> members;
      if (method==METHOD_VECTOR_WARD ||
          method==METHOD_VECTOR_CENTROID) {
        members.init(N);
        if (Rf_isNull(members_)) {
          for (t_index i=0; i<N; i++) members[i] = 1;
        }
        else {
          PROTECT(members_ = AS_INTEGER(members_));
          if (LENGTH(members_)!=N)
            Rf_error("The length of 'members' must be the same as the number of data points.");
          const int * const m = INTEGER_POINTER(members_);
          for (t_index i=0; i<N; i++) members[i] = m[i];
          UNPROTECT(1); // members
        }
      }

      // Parameter p
      PROTECT(p_);
      double p = 0;
      if (metric==METRIC_R_MINKOWSKI) {
        if (!IS_NUMERIC(p_) || LENGTH(p_)!=1)
          Rf_error("'p' must be a single floating point number.");
        p = *NUMERIC_POINTER(p_);
      }
      else {
        if (p_ != R_NilValue) {
          Rf_error("No metric except 'minkowski' allows a 'p' parameter.");
        }
      }
      UNPROTECT(1); // p_

      // The generic_linkage_vector algorithm uses labels N,N+1,... for the
      // new nodes, so we need a table which node is stored in which row.
      bool make_row_repr =
        (method!=METHOD_VECTOR_SINGLE);

      R_dissimilarity dist(X, N, dim, members,
                           static_cast<unsigned char>(method),
                           static_cast<unsigned char>(metric),
                           p,
                           make_row_repr);
      cluster_result Z2(N-1);

      /*
        Clustering step
      */
      switch (method) {
      case METHOD_VECTOR_SINGLE:
        MST_linkage_core_vector(N, dist, Z2);
        break;

      case METHOD_VECTOR_WARD:
        generic_linkage_vector<METHOD_METR_WARD>(N, dist, Z2);
        break;

      case METHOD_VECTOR_CENTROID:
        generic_linkage_vector<METHOD_METR_CENTROID>(N, dist, Z2);
        break;

      case METHOD_VECTOR_MEDIAN:
        generic_linkage_vector<METHOD_METR_MEDIAN>(N, dist, Z2);
        break;
      }

      X.free();     // Free the memory now
      members.free(); // (not strictly necessary).

      dist.postprocess(Z2);

      SEXP m; // return field "merge"
      PROTECT(m = NEW_INTEGER(2*(N-1)));
      int * const merge = INTEGER_POINTER(m);

      SEXP dim_m; // Specify that m is an (N-1)×2 matrix
      PROTECT(dim_m = NEW_INTEGER(2));
      INTEGER(dim_m)[0] = N-1;
      INTEGER(dim_m)[1] = 2;
      SET_DIM(m, dim_m);

      SEXP h; // return field "height"
      PROTECT(h = NEW_NUMERIC(N-1));
      double * const height = NUMERIC_POINTER(h);

      SEXP o; // return fiels "order'
      PROTECT(o = NEW_INTEGER(N));
      int * const order = INTEGER_POINTER(o);

      if (method==METHOD_VECTOR_SINGLE)
        generate_R_dendrogram<false>(merge, height, order, Z2, N);
      else
        generate_R_dendrogram<true>(merge, height, order, Z2, N);

      SEXP n; // names
      PROTECT(n = NEW_CHARACTER(3));
      SET_STRING_ELT(n, 0, COPY_TO_USER_STRING("merge"));
      SET_STRING_ELT(n, 1, COPY_TO_USER_STRING("height"));
      SET_STRING_ELT(n, 2, COPY_TO_USER_STRING("order"));

      PROTECT(r = NEW_LIST(3)); // field names in the output list
      SET_ELEMENT(r, 0, m);
      SET_ELEMENT(r, 1, h);
      SET_ELEMENT(r, 2, o);
      SET_NAMES(r, n);

      UNPROTECT(6); // m, dim_m, h, o, r, n
    } // try
    catch (std::bad_alloc&) {
      Rf_error( "Memory overflow.");
    }
    catch(std::exception& e){
      Rf_error( e.what() );
    }
    catch(...){
      Rf_error( "C++ exception (unknown reason)." );
    }

    return r;
  }

  void R_init_fastcluster(DllInfo * const info)
  {
    R_CallMethodDef callMethods[]  = {
      {"fastcluster", (DL_FUNC) &fastcluster, 4},
      {"fastcluster_vector", (DL_FUNC) &fastcluster_vector, 5},
      {NULL, NULL, 0}
    };
    R_registerRoutines(info, NULL, callMethods, NULL, NULL);
  }

} // extern "C"
