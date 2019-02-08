cdef extern from "hmm.h":

    cdef log_multivariate_normal_density(float X, float means, float covars)