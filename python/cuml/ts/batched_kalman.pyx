# cython: language_level = 3

import numpy as np

import ctypes
cimport numpy as np
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from libcpp cimport bool

cdef extern from "ts/batched_kalman.h":

  void batched_kalman_filter(double* ptr_ys_b,
                             int nobs,
                             # double* h_Zb,
                             # double* h_Rb,
                             # double* h_Tb,
                             const vector[double*]& ptr_Zb,
                             const vector[double*]& ptr_Rb,
                             const vector[double*]& ptr_Tb,
                             const vector[double*]& ptr_P0,
                             int r,
                             int num_batches,
                             vector[double]& vec_loglike_b,
                             vector[vector[double]]& vec_vs_b,
                             bool initP_with_kalman_iterations)

  void batched_kalman_filter_cpu(const vector[double*]& ptr_ys_b,
                                 int nobs,
                                 const vector[double*]& ptr_Zb,
                                 const vector[double*]& ptr_Rb,
                                 const vector[double*]& ptr_Tb,
                                 int r,
                                 vector[double]& vec_loglike_b,
                                 vector[vector[double]]& vec_vs_b,
                                 bool initP_with_kalman_iterations)


def batched_kfilter(np.ndarray[double, ndim=2] y,
                    Z_b, # list of numpy arrays
                    R_b,
                    T_b,
                    P0,
                    int r,
                    gpu=True,
                    initP_with_kalman_iterations=True):

    cdef vector[double] vec_loglike_b
    
    cdef int nobs = y.shape[0]
    cdef int num_batches = y.shape[1]

    # cuDF wasn't working well, comment out for now
    # # Extract device pointer from DataFrame. Careful: `y_mat` temporary is to
    # # avoid the "gpu_matrix" object from getting garbage collected. `ytmp`
    # # simply satisfies the Cython compiler.
    # y_mat = y.as_gpu_matrix()
    # cdef unsigned long long ytmp = y_mat.gpu_data.device_pointer.value
    # cdef double* y_ptr = <double*>ytmp

    cdef np.ndarray[double, ndim=2, mode="fortran"] Z_bi
    cdef np.ndarray[double, ndim=2, mode="fortran"] R_bi
    cdef np.ndarray[double, ndim=2, mode="fortran"] T_bi
    cdef np.ndarray[double, ndim=2, mode="fortran"] P0_bi
    cdef vector[double*] vec_Zb
    cdef vector[double*] vec_Rb
    cdef vector[double*] vec_Tb
    cdef vector[double*] vec_P0

    cdef vector[double*] vec_ys_b

    cdef vector[vector[double]] vec_vs_b

    for i in range(num_batches):
        Z_bi = Z_b[i]
        R_bi = R_b[i]
        T_bi = T_b[i]
        P0_bi = P0[i]
        vec_Zb.push_back(&Z_bi[0,0])
        vec_Rb.push_back(&R_bi[0,0])
        vec_Tb.push_back(&T_bi[0,0])
        if not initP_with_kalman_iterations:
            # invImTT = np.linalg.pinv(np.eye(r**2) - np.kron(T_bi, T_bi))
            # P0 = np.reshape(invImTT @ (R_bi @ R_bi.T).ravel(), (r, r), order="F")
            vec_P0.push_back(&P0_bi[0,0])
            # print("P0=",P0[0,0], P0[0,1], P0[1,0], P0[1,1])

    if gpu:
        batched_kalman_filter(&y[0,0],
                              nobs,
                              # &Z_dense[0], &R_dense[0], &T_dense[0],
                              vec_Zb, vec_Rb, vec_Tb,
                              vec_P0,
                              r,
                              num_batches,
                              vec_loglike_b,
                              vec_vs_b,
                              initP_with_kalman_iterations)
    else:
        
        # initialize cpu input
        vec_ys_b.resize(num_batches)
        for i in range(num_batches):
            vec_ys_b[i] = &y[0,i]

        batched_kalman_filter_cpu(vec_ys_b,
                                  nobs,
                                  # &Z_dense[0], &R_dense[0], &T_dense[0],
                                  vec_Zb, vec_Rb, vec_Tb,
                                  r,
                                  vec_loglike_b,
                                  vec_vs_b,
                                  initP_with_kalman_iterations)

    # convert C++-results to numpy arrays
    ll_b = np.zeros(num_batches)
    vs = np.zeros((nobs, num_batches))
    for i in range(num_batches):
        ll_b[i] = vec_loglike_b[i]
        for j in range(nobs):
            vs[j,i] = vec_vs_b[i][j]

    return ll_b, vs
