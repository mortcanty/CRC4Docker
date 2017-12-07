cdef extern from "gsl/gsl_sf.h":

    double gsl_sf_gamma (double x)
    double gsl_sf_fact (unsigned int n)

cdef extern from "gsl/gsl_statistics_double.h":
    double gsl_stats_quantile_from_sorted_data(double sorted_data[], size_t stride, size_t n, double f)
