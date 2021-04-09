#
# Copyright (c) 2021, NVIDIA CORPORATION.
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

import cupy as cp
import numpy as np
import cuml.internals
from cuml.common.input_utils import determine_array_type
from cuml.common import (input_to_cuml_array, CumlArray, logger)
from libc.stdint cimport uintptr_t
from cuml.raft.common.handle cimport handle_t
from cuml.raft.common.handle import Handle

cdef extern from "cuml/metrics/metrics.hpp" namespace "ML::Metrics":
    double kl_divergence(const handle_t &handle,
                         const double *y,
                         const double *y_hat,
                         int n) except +
    float kl_divergence(const handle_t &handle,
                        const float *y,
                        const float *y_hat,
                        int n) except +

@cuml.internals.api_return_array(get_output_type=True)
def kl_divergence(P, Q, handle=None, convert_dtype=True):
    """
    Calculates the "Kullback-Leibler" Divergence
    The KL divergence tells us how well the probability distribution Q
    approximates the probability distribution P
    It is often also used as a 'distance metric' between two probablity
    ditributions (not symmetric)

        Parameters
        ----------
        handle : cuml.Handle
        P : NumPy ndarray or Numba device
           Array of probabilities corresponding to distribution P
        Q : NumPy ndarray, Numba device
           Array of probabilities corresponding to distribution Q

        Returns
        -------
        float
           The KL Divergence value
    """
    handle = Handle() if handle is None else handle
    cdef handle_t *handle_ = <handle_t*> <size_t> handle.getHandle()

    P_m, n_samples_p, n_features_p, dtype_p = \
        input_to_cuml_array(P, order="K", check_dtype=[np.float32, np.float64])
    Q_m, n_samples_q, n_features_q, dtype_q = \
        input_to_cuml_array(Q, order="K", check_dtype=[np.float32, np.float64])

    if (n_samples_p != n_samples_q) or (n_features_p != n_features_q):
        raise ValueError("Incompatible dimension for Y and Y_hat arrays: \
                         P.shape == ({},{}) while Q.shape == ({},{})"
                         .format(n_samples_p, n_features_p, n_samples_q,
                                 n_features_q))

    
    cdef uintptr_t d_P_ptr = P_m.ptr
    cdef uintptr_t d_Q_ptr = Q_m.ptr

    '''n_elements = n_samples_p * n_features_p
    if (dtype_p == np.float32):
        res = kl_divergence(handle_[0],
                            <float*> d_P_ptr,
                            <float*> d_Q_ptr,
                            <int> n_elements)
    elif (dtype_p == np.float64):
        res = kl_divergence(handle_[0],
                            <double*> d_P_ptr,
                            <double*> d_Q_ptr,
                            <int> n_elements)
    else:
        raise NotImplementedError("Unsupported dtype: {}".format(dtype_p))
    
    del P_m
    del Q_m
    return res'''