cdef extern from "liblinear/linear.h":
    
    cdef struct feature_node:
        int index
        double value
    
    cdef struct problem:
        int l, n # l samples, n features
        int *y
        feature_node **x
        double bias
    
    cdef struct parameter:
        int solver_type

        # these are for training only
        double eps  # stopping criteria
        double C
        int nr_weight
        int *weight_label
        double* weight

    cdef struct model:
        parameter param	# parameter
        int nr_class	# number of classes
        int nr_feature
        double *w
        int *label    # label of each class
        double bias

    void set_print_string_function(void (*print_func)(char *))
        
    model* train(problem *prob, parameter *param)
    int predict(model *model_, feature_node *x)
    int predict_values(model *model_, feature_node *x, double* dec_values)
    int predict_probability(model *model_, feature_node *x, double* prob_estimates)

    int save_model(char *model_file_name, model *model_)
    model *load_model(char *model_file_name)

    void free_and_destroy_model(model **model_ptr_ptr)
    void destroy_param(parameter* param)

    char *check_parameter(problem *prob, parameter *param)
