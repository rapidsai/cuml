# cython: language_level = 3

import numpy as np

import ctypes
cimport numpy as np
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free



cdef extern from "ts/batched_kalman.h":
  void batched_kalman_filter(const vector[double*]& ptr_ys_b,
                             int nobs,
                             const vector[double*]& ptr_Zb,
                             const vector[double*]& ptr_Rb,
                             const vector[double*]& ptr_Tb,
                             int r,
                             vector[double]& ptr_loglike_b,
                             vector[double]& ptr_sigma2_b)

  void batched_kalman_filter_cudf(double* ptr_ys_b,
                                  int nobs,
                                  # double* h_Zb,
                                  # double* h_Rb,
                                  # double* h_Tb,
                                  const vector[double*]& ptr_Zb,
                                  const vector[double*]& ptr_Rb,
                                  const vector[double*]& ptr_Tb,
                                  int r,
                                  int num_batches,
                                  vector[double]& ptr_loglike_b)

  void batched_kalman_filter_cpu(const vector[double*]& ptr_ys_b,
                                 int nobs,
                                 const vector[double*]& ptr_Zb,
                                 const vector[double*]& ptr_Rb,
                                 const vector[double*]& ptr_Tb,
                                 int r,
                                 vector[double]& ptr_loglike_b,
                                 vector[double]& ptr_sigma2_b)


def cudf_kfilter(np.ndarray[double, ndim=2] y,
                 Z_b, R_b, T_b,
                 # np.ndarray[double, ndim=1] Z_dense,
                 # np.ndarray[double, ndim=1] R_dense,
                 # np.ndarray[double, ndim=1] T_dense,
                 int r):

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
    cdef vector[double*] vec_Zb
    cdef vector[double*] vec_Rb
    cdef vector[double*] vec_Tb

    for i in range(num_batches):
        Z_bi = Z_b[i]
        R_bi = R_b[i]
        T_bi = T_b[i]
        vec_Zb.push_back(&Z_bi[0,0])
        vec_Rb.push_back(&R_bi[0,0])
        vec_Tb.push_back(&T_bi[0,0])


    batched_kalman_filter_cudf(&y[0,0],
                               nobs,
                               # &Z_dense[0], &R_dense[0], &T_dense[0],
                               vec_Zb, vec_Rb, vec_Tb,
                               r,
                               num_batches,
                               vec_loglike_b)

    # convert C++-results to numpy arrays
    ll_b = np.zeros(num_batches)
    for i in range(num_batches):
        ll_b[i] = vec_loglike_b[i]

    return ll_b

def batched_kfilter(ys_b,
                    Z_b,
                    R_b,
                    T_b,
                    int r,
                    gpu=True):

    cdef vector[double*] vec_ys_b
    cdef vector[double*] vec_Zb
    cdef vector[double*] vec_Rb
    cdef vector[double*] vec_Tb

    cdef vector[double] vec_loglike_b
    cdef vector[double] vec_sigma2_b

    cdef np.ndarray[double, ndim=1, mode="fortran"] ysi

    cdef np.ndarray[double, ndim=1, mode="fortran"] ys_bi
    cdef np.ndarray[double, ndim=2, mode="fortran"] Z_bi
    cdef np.ndarray[double, ndim=2, mode="fortran"] R_bi
    cdef np.ndarray[double, ndim=2, mode="fortran"] T_bi

    num_batches = len(Z_b)
    cdef int nobs = len(ys_b[0])

    # initialize input/output
    vec_ys_b.resize(num_batches)
    for i in range(num_batches):
        num_samples_i = len(ys_b[i])
        ysi = ys_b[i]
        vec_ys_b[i] = &ysi[0]

    vec_loglike_b.resize(num_batches)
    vec_sigma2_b.resize(num_batches)
    
    for i in range(num_batches):
        Z_bi = Z_b[i]
        R_bi = R_b[i]
        T_bi = T_b[i]
        vec_Zb.push_back(&Z_bi[0,0])
        vec_Rb.push_back(&R_bi[0,0])
        vec_Tb.push_back(&T_bi[0,0])

    if gpu:    
        batched_kalman_filter(vec_ys_b,
                              nobs,
                              vec_Zb, vec_Rb, vec_Tb,
                              r,
                              vec_loglike_b, vec_sigma2_b)
    else:
        batched_kalman_filter_cpu(vec_ys_b,
                                  nobs,
                                  vec_Zb, vec_Rb, vec_Tb,
                                  r,
                                  vec_loglike_b, vec_sigma2_b)

    # convert C-arrays to numpy arrays
    ll_b = np.zeros(num_batches)
    sigma2_b = np.zeros(num_batches)
    for i in range(num_batches):
        ll_b[i] = vec_loglike_b[i]
        sigma2_b[i] = vec_sigma2_b[i]

    # Convert C++ vectors of pointers to Python List[ndarray]
    for (i, ysi) in enumerate(ys_b):
        ys_len = len(ysi)
    
    return ll_b, sigma2_b
