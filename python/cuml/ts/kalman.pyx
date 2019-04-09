import numpy as np

import ctypes
cimport numpy as np
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free

cdef extern from "ts/kalman.h":
  cdef void kalman_filter(double* ptr_ys, int ys_len, double* ptr_Z, double* ptr_R, double* ptr_T, int r,
                          double* ptr_vs, double* ptr_Fs, double* ptr_loglike, double* ptr_sigma2)

cdef extern from "ts/batched_kalman.h":
  cdef void batched_kalman_filter(const vector[double*]& ptr_ys_b,
                           const vector[double*]& ptr_Zb,
                           const vector[double*]& ptr_Rb,
                           const vector[double*]& ptr_Tb,
                           int r,
                           vector[double*]& ptr_vs_b,
                           vector[double*]& ptr_Fs_b,
                           vector[double]& ptr_loglike_b,
                           vector[double]& ptr_sigma2_b)



def kfilter(np.ndarray[double, ndim=1, mode="fortran"] ys,
            np.ndarray[double, ndim=2, mode="fortran"] Z,
            np.ndarray[double, ndim=2, mode="fortran"] R,
            np.ndarray[double, ndim=2, mode="fortran"] T,
            int r):

    cdef int nobs = len(ys)
    cdef np.ndarray[double, ndim=1, mode="fortran"] vs = np.zeros(nobs)
    cdef np.ndarray[double, ndim=1, mode="fortran"] Fs = np.zeros(nobs)
    cdef double loglike = 0.0
    cdef double sigma2 = 0.0

    kalman_filter(&ys[0], len(ys), &Z[0,0], &R[0,0], &T[0,0], r, &vs[0], &Fs[0], &loglike, &sigma2)

    # # Test batched version
    # cdef vector[double*] vec_ys_b
    # cdef vector[double*] vec_Zb
    # cdef vector[double*] vec_Rb
    # cdef vector[double*] vec_Tb

    # cdef vector[double*] vec_vs_b
    # cdef vector[double*] vec_Fs_b
    # cdef vector[double] vec_loglike_b
    # cdef vector[double] vec_sigma2_b

    # cdef np.ndarray[double, ndim=1, mode="fortran"] vsi
    # cdef np.ndarray[double, ndim=1, mode="fortran"] Fsi

    # num_batches = 1
    # for i in range(len(ys)):
    #     vec_ys_b.push_back(<double*>malloc(num_batches*sizeof(double)))
    #     for j in range(num_batches):
    #         vec_ys_b[i][j] = ys[i]

    # vec_Zb.push_back(&Z[0,0])
    # vec_Rb.push_back(&R[0,0])
    # vec_Tb.push_back(&T[0,0])

    # # initialize output
    # for i in range(num_batches):
    #     num_samples_i = len(ys)
    #     vsi = np.zeros(num_samples_i)
    #     Fsi = np.zeros(num_samples_i)
    #     vec_vs_b.push_back(&vsi[0])
    #     vec_Fs_b.push_back(&Fsi[0])

    # vec_loglike_b.resize(num_batches)
    # vec_sigma2_b.resize(num_batches)
    # batched_kalman_filter(vec_ys_b,
    #                       vec_Zb, vec_Rb, vec_Tb,
    #                       r,
    #                       vec_vs_b, vec_Fs_b,
    #                       vec_loglike_b, vec_sigma2_b)

    
    # # print("1st five vs:{}".format(vs[0:5]))
    # vs2 = np.zeros(len(ys))
    # for i in range(len(ys)):
    #     vs2[i] = vec_vs_b[0][i]

    # Fs2 = np.zeros(len(ys))
    # for i in range(len(ys)):
    #     Fs2[i] = vec_Fs_b[0][i]
    # print("error Fs=", np.linalg.norm(Fs-Fs2))
    #     # print(vec_vs_b[0][i])

    # # print("1st five Fs:{}".format(Fs[0:5]))
    # for i in range(5):
    #     print(vec_Fs_b[0][i])

    return vs, Fs, loglike, sigma2


def init_kalman_matrices(ar_params, ma_params):
    p = len(ar_params)
    q = len(ma_params)
    r = max(p, q+1)  # see (3.18) in TSA by D&K

    Z = np.zeros((1, r), order="F")
    Z[0, 0] = 1.0

    R = np.zeros((r, 1), order="F")
    # for (i, ma_i) in enumerate(ma_params):
    R[1:q + 1, 0] = ma_params[:]

    R[0] = 1.0

    T = np.zeros((r, r), order="F")
    params_padded = np.zeros(r)
    # handle zero coefficients if necessary
    params_padded[:p] = ar_params[:]
    T[:, 0] = params_padded
    T[:-1, 1:] = np.eye(r - 1)

    return Z, R, T, r
