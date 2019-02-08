import numpy as np
cimport numpy as np
np.import_array

cdef extern from "hmm/gmm_c.h" namespace "ML":

    cdef void make_ptr_gmm()

    cdef void gmm_fit()

    cdef void gmm_predict()

