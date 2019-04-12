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
                             vector[double*]& ptr_vs_b,
                             vector[double*]& ptr_Fs_b,
                             vector[double]& ptr_loglike_b,
                             vector[double]& ptr_sigma2_b)


def batched_kfilter(ys_b,
                    Z_b,
                    R_b,
                    T_b,
                    int r):

    cdef vector[double*] vec_ys_b
    cdef vector[double*] vec_Zb
    cdef vector[double*] vec_Rb
    cdef vector[double*] vec_Tb

    cdef vector[double*] vec_vs_b
    cdef vector[double*] vec_Fs_b
    cdef vector[double] vec_loglike_b
    cdef vector[double] vec_sigma2_b

    cdef np.ndarray[double, ndim=1, mode="fortran"] ysi
    cdef np.ndarray[double, ndim=1, mode="fortran"] vsi
    cdef np.ndarray[double, ndim=1, mode="fortran"] Fsi

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
        vsi = np.zeros(num_samples_i)
        Fsi = np.zeros(num_samples_i)
        vec_vs_b.push_back(&vsi[0])
        vec_Fs_b.push_back(&Fsi[0])

    vec_loglike_b.resize(num_batches)
    vec_sigma2_b.resize(num_batches)
    
    for i in range(num_batches):
        Z_bi = Z_b[i]
        R_bi = R_b[i]
        T_bi = T_b[i]
        vec_Zb.push_back(&Z_bi[0,0])
        vec_Rb.push_back(&R_bi[0,0])
        vec_Tb.push_back(&T_bi[0,0])

    batched_kalman_filter(vec_ys_b,
                          nobs,
                          vec_Zb, vec_Rb, vec_Tb,
                          r,
                          vec_vs_b, vec_Fs_b,
                          vec_loglike_b, vec_sigma2_b)

    # convert C-arrays to numpy arrays
    ll_b = np.zeros(num_batches)
    sigma2_b = np.zeros(num_batches)
    for i in range(num_batches):
        ll_b[i] = vec_loglike_b[i]
        sigma2_b[i] = vec_sigma2_b[i]

    # Convert C++ vectors of pointers to Python List[ndarray]
    vs_b = []
    Fs_b = []
    for (i, ysi) in enumerate(ys_b):
        ys_len = len(ysi)
        vsi = np.zeros(ys_len)
        Fsi = np.zeros(ys_len)
        for j in range(ys_len):
            vsi[j] = vec_vs_b[i][j]
            Fsi[j] = vec_Fs_b[i][j]

        vs_b.append(vsi)
        Fs_b.append(Fsi)
    
    return vs_b, Fs_b, ll_b, sigma2_b
    
    

