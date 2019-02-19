import numpy as np
from libcpp cimport bool

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
        solver algorithm
        bool trans_input

    cdef cppclass paramsPCA(paramsTSVD):
        bool copy
        bool whiten


cdef extern from "pca/pca_c.h" namespace "ML":

    cdef void pcaFit(float *input,
                     float *components,
                     float *explained_var,
                     float *explained_var_ratio,
                     float *singular_vals,
                     float *mu,
                     float *noise_vars,
                     paramsPCA prms)

    cdef void pcaFit(double *input,
                     double *components,
                     double *explained_var,
                     double *explained_var_ratio,
                     double *singular_vals,
                     double *mu,
                     double *noise_vars,
                     paramsPCA prms)

    cdef void pcaFitTransform(float *input,
                              float *trans_input,
                              float *components,
                              float *explained_var,
                              float *explained_var_ratio,
                              float *singular_vals,
                              float *mu,
                              float *noise_vars,
                              paramsPCA prms)

    cdef void pcaFitTransform(double *input,
                              double *trans_input,
                              double *components,
                              double *explained_var,
                              double *explained_var_ratio,
                              double *singular_vals,
                              double *mu,
                              double *noise_vars,
                              paramsPCA prms)

    cdef void pcaInverseTransform(float *trans_input,
                                  float *components,
                                  float *singular_vals,
                                  float *mu,
                                  float *input,
                                  paramsPCA prms)

    cdef void pcaInverseTransform(double *trans_input,
                                  double *components,
                                  double *singular_vals,
                                  double *mu,
                                  double *input,
                                  paramsPCA prms)

    cdef void pcaTransform(float *input,
                           float *components,
                           float *trans_input,
                           float *singular_vals,
                           float *mu,
                           paramsPCA prms)

    cdef void pcaTransform(double *input,
                           double *components,
                           double *trans_input,
                           double *singular_vals,
                           double *mu,
                           paramsPCA prms)

