#
# Copyright (c) 2019, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np

import ctypes
cimport numpy as np
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from libcpp cimport bool
from libcpp.string cimport string
cimport cython
from cuml.ts.batched_kalman import pynvtx_range_push, pynvtx_range_pop

cdef extern from "ts/batched_arima.h" namespace "ML":
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
