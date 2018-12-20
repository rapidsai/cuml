import numpy as np
cimport numpy as np
from libcpp cimport bool
np.import_array


cdef extern from "mean/mean_c.h" namespace "ML":

    cdef void mean(float* mu,
                   float* data,
                   int D,
                   int N,
                   bool sample,
                   bool rowMajor)

    cdef void mean(double* mu,
                   double* data,
                   int D,
                   int N,
                   bool sample,
                   bool rowMajor)