import numpy as np

import ctypes
cimport numpy as np
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free

cdef extern from "ts/kalman.h":
  cdef void kalman_filter(double* ptr_ys, int ys_len, double* ptr_Z, double* ptr_R, double* ptr_T, int r,
                          double* ptr_vs, double* ptr_loglike)


def kfilter(np.ndarray[double, ndim=1, mode="fortran"] ys,
            np.ndarray[double, ndim=2, mode="fortran"] Z,
            np.ndarray[double, ndim=2, mode="fortran"] R,
            np.ndarray[double, ndim=2, mode="fortran"] T,
            int r):

    cdef int nobs = len(ys)
    cdef np.ndarray[double, ndim=1, mode="fortran"] vs = np.zeros(nobs)
    cdef double loglike = 0.0

    kalman_filter(&ys[0], len(ys), &Z[0,0], &R[0,0], &T[0,0], r, &vs[0], &loglike)

    return vs, loglike


def init_kalman_matrices(ar_params, ma_params, r=None):
    p = len(ar_params)
    q = len(ma_params)

    # for batched case, we input the maximum `r` to zero-pad some matrices.
    if r is None:
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
