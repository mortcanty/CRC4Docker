cdef extern from "libsvm/svm.h":
    
    cdef struct svm_node:
        int index
        double value
    
    cdef struct svm_problem:
        int l
        double *y
        svm_node **x
    
    cdef struct svm_parameter:
        int svm_type
        int kernel_type
        int degree	# for poly
        double gamma	# for poly/rbf/sigmoid
        double coef0	# for poly/sigmoid

        # these are for training only
        double cache_size # in MB
        double eps	# stopping criteria
        double C	# for C_SVC, EPSILON_SVR, and NU_SVR
        int nr_weight	      # for C_SVC
        int *weight_label     # for C_SVC
        double* weight	      # for C_SVC
        double nu	# for NU_SVC, ONE_CLASS, and NU_SVR
        double p	# for EPSILON_SVR
        int shrinking	# use the shrinking heuristics
        int probability # do probability estimates

    cdef struct svm_model:
        svm_parameter param	# parameter
        int nr_class	# number of classes, = 2 in regression/one class svm
        int l	        # total SV
        svm_node **SV	# SVs (SV[l])
        # coefficients for SVs in decision functions (sv_coef[k-1][l])
        double **sv_coef 
        double *rho     # constants in decision functions (rho[k*(k-1)/2])
        # pairwise probability information
        double *probA
        double *probB
        
        # for classification only
        int *label    # label of each class (label[k])
        int *nSV      # number of SVs for each class (nSV[k])
                      # nSV[0] + nSV[1] + ... + nSV[k-1] = l
        
        # XXX
        int free_sv  # 1 if svm_model is created by svm_load_model
                     # 0 if svm_model is created by svm_train

    void svm_set_print_string_function(void (*print_func)(char *))
        
    svm_model *svm_train(svm_problem *prob, svm_parameter *param)

    double svm_predict(svm_model *model, svm_node *x)
    double svm_predict_probability(svm_model *model, 
        svm_node *x, double* prob_estimates)
    double svm_predict_values(svm_model *model, 
        svm_node *x, double* dec_values)
    
    void svm_free_and_destroy_model(svm_model **model_ptr_ptr)
    void svm_destroy_param(svm_parameter* param)
    
    int svm_save_model(char *model_file_name, svm_model *model)
    svm_model *svm_load_model(char *model_file_name)
    
    char *svm_check_parameter(svm_problem *prob, svm_parameter *param)
    int svm_check_probability_model(svm_model *model)

    
