from c_hmm cimport *

cpdef log_multivariate_normal_density(x, mean, variance):
    print(log_multivariate_normal_density(x, mean, variance))