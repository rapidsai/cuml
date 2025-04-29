#
# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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
from pylibraft.common.handle import Handle

import cuml.internals
from cuml.common import input_to_cuml_array

from libc.stdint cimport uintptr_t
from pylibraft.common.handle cimport handle_t


cdef extern from "cuml/metrics/metrics.hpp" namespace "ML::Metrics" nogil:
    double c_kl_divergence "ML::Metrics::kl_divergence"(
        const handle_t &handle,
        const double *y,
        const double *y_hat,
        int n) except +
    float c_kl_divergence "ML::Metrics::kl_divergence"(
        const handle_t &handle,
        const float *y,
        const float *y_hat,
        int n) except +


@cuml.internals.api_return_any()
def kl_divergence(P, Q, handle=None, convert_dtype=True):
    """
    Calculates the "Kullback-Leibler" Divergence
    The KL divergence tells us how well the probability distribution Q
    approximates the probability distribution P
    It is often also used as a 'distance metric' between two probability
    distributions (not symmetric)

    Parameters
    ----------
    P : Dense array of probabilities corresponding to distribution P
        shape = (n_samples, 1)
        Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
        ndarray, cuda array interface compliant array like CuPy.

    Q : Dense array of probabilities corresponding to distribution Q
        shape = (n_samples, 1)
        Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
        ndarray, cuda array interface compliant array like CuPy.

    handle : cuml.Handle

    convert_dtype : bool, optional (default = True)
        When set to True, the method will, convert P and
        Q to be the same data type: float32. This
        will increase memory used for the method.

    Returns
    -------
    float
        The KL Divergence value
    """
    handle = Handle() if handle is None else handle
    cdef handle_t *handle_ = <handle_t*> <size_t> handle.getHandle()

    P_m, n_features_p, _, dtype_p = \
        input_to_cuml_array(P, check_cols=1,
                            convert_to_dtype=(np.float32 if convert_dtype
                                              else None),
                            check_dtype=[np.float32, np.float64])
    Q_m, n_features_q, _, _ = \
        input_to_cuml_array(Q, check_cols=1,
                            convert_to_dtype=(dtype_p if convert_dtype
                                              else None),
                            check_dtype=[dtype_p])

    if n_features_p != n_features_q:
        raise ValueError("Incompatible dimension for P and Q arrays: \
                         P.shape == ({}) while Q.shape == ({})"
                         .format(n_features_p, n_features_q))

    cdef uintptr_t d_P_ptr = P_m.ptr
    cdef uintptr_t d_Q_ptr = Q_m.ptr

    if (dtype_p == np.float32):
        res = c_kl_divergence(handle_[0],
                              <float*> d_P_ptr,
                              <float*> d_Q_ptr,
                              <int> n_features_p)
    else:
        res = c_kl_divergence(handle_[0],
                              <double*> d_P_ptr,
                              <double*> d_Q_ptr,
                              <int> n_features_p)

    return res
