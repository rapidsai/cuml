import numpy as np

import ctypes
cimport numpy as np
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from libcpp cimport bool
from libcpp.string cimport string
cimport cython
from .batched_kalman import pynvtx_range_push, pynvtx_range_pop

cdef extern from "ts/batched_arima.h":
  void batched_loglike(double* y, int num_batches, int nobs, int p,
                       int d, int q, double* params, vector[double]& vec_loglike, bool trans)


def batched_loglike_cuda(np.ndarray[double, ndim=2] y, int num_batches, int nobs,
                         int p, int d, int q, np.ndarray[double] x, bool trans):

    cdef vector[double] vec_loglike
    cdef vector[double] vec_y_cm
    cdef vector[double] vec_x

    pynvtx_range_push("batched_loglike_cuda")

    num_params = (p+d+q)

    vec_loglike.resize(num_batches)

    # ensure Column major layout
    cdef np.ndarray[double, ndim=2] y_cm = np.asfortranarray(y)

    batched_loglike(&y_cm[0,0], num_batches, nobs, p, d, q, &x[0], vec_loglike, trans)

    # copy results into numpy array
    loglike = np.zeros(num_batches)
    for i in range(num_batches):
        loglike[i] = vec_loglike[i]

    pynvtx_range_pop()

    return loglike
