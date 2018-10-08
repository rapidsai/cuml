import numpy as np
cimport numpy as np
from libcpp cimport bool
np.import_array

cdef extern from "ml_utils.h" namespace "ML":

    enum solver:
        COV_EIG_DQ, COV_EIG_JACOBI, RANDOMIZED

    cdef cppclass params:
        int n_rows
        int n_cols
        int gpu_id

    cdef cppclass paramsSolver(params):
        int n_rows
        int n_cols
        double tol
        int n_iterations
        int random_state
        int verbose

    cdef cppclass paramsTSVD(paramsSolver):
        int n_components
        int max_sweeps
        solver algorithm #= solver::COV_EIG_DQ
        bool trans_input

    cdef cppclass paramsPCA(paramsTSVD):
        bool copy
        bool whiten

cdef extern from "tsvd/tsvd_c.h" namespace "ML":

    cdef void tsvdFit(float *input,
                      float *components,
                      float *explained_var,
                      float *explained_var_ratio,
                      float *singular_vals,
                      paramsTSVD prms)

    cdef void tsvdFit(double *input,
                      double *components,
                      double *explained_var,
                      double *explained_var_ratio,
                      double *singular_vals,
                      paramsTSVD prms)

    cdef void tsvdFitTransform(float *input,
                               float *trans_input,
                               float *components,
                               float *explained_var,
                               float *explained_var_ratio,
                               float *singular_vals,
                               paramsTSVD prms)

    cdef void tsvdFitTransform(double *input,
                               double *trans_input,
                               double *components,
                               double *explained_var,
                               double *explained_var_ratio,
                               double *singular_vals,
                               paramsTSVD prms)

    cdef void tsvdInverseTransform(float *trans_input,
                                   float *components,
                                   float *input,
                                   paramsTSVD prms)

    cdef void tsvdInverseTransform(double *trans_input,
                                   double *components,
                                   double *input,
                                   paramsTSVD prms)

    cdef void tsvdTransform(float *input,
                            float *components,
                            float *trans_input,
                            paramsTSVD prms)

    cdef void tsvdTransform(double *input,
                            double *components,
                            double *trans_input,
                            paramsTSVD prms)
