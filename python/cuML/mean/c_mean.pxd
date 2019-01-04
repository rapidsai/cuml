import numpy as np
cimport numpy as np
from libcpp cimport bool
np.import_array


cdef extern from "mg_descriptor.h" namespace "ML":
    cdef cppclass MGDescriptorFloat:
        MGDescriptorFloat(float *data, int n_rows, int n_cols) except +

    cdef cppclass MGDescriptorDouble:
        MGDescriptorDouble(double *data, int n_rows, int n_cols) except +


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

    cdef void meanMG(MGDescriptorFloat *mu,
                     MGDescriptorFloat *data,
                     int n_gpus,
                     bool sample,
                     bool rowMajor,
                     bool rowSplit)

    cdef void meanMG(MGDescriptorDouble *mu,
                     MGDescriptorDouble *data,
                     int n_gpus,
                     bool sample,
                     bool rowMajor,
                     bool rowSplit)

