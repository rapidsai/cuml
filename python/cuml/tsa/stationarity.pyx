# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import ctypes
import numpy as np
from libc.stdint cimport uintptr_t
from libcpp cimport bool

import cuml
from cuml.common.array import CumlArray as cumlArray
from cuml.common.handle cimport cumlHandle
from cuml.utils.input_utils import input_to_cuml_array


cdef extern from "cuml/tsa/stationarity.h" namespace "ML":
    int cpp_kpss "ML::Stationarity::kpss_test" (
        const cumlHandle& handle,
        const float* d_y,
        bool* results,
        int batch_size,
        int n_obs,
        int d, int D, int s,
        float pval_threshold)

    int cpp_kpss "ML::Stationarity::kpss_test" (
        const cumlHandle& handle,
        const double* d_y,
        bool* results,
        int batch_size,
        int n_obs,
        int d, int D, int s,
        double pval_threshold)


def kpss_test(y, d=1, D=0, s=0, pval_threshold=0.05, handle=None):
    """
    Perform the KPSS stationarity test on the data differenced according
    to the given order

    Parameters
    ----------
    y : dataframe or array-like (device or host)
        The time series data, assumed to have each time series in columns.
        Acceptable formats: cuDF DataFrame, cuDF Series, NumPy ndarray,
        Numba device ndarray, cuda array interface compliant array like CuPy.
    d: integer
        Order of simple differencing
    D: integer
        Order of seasonal differencing
    s: integer
        Seasonal period if D > 0
    pval_threshold : float
        The p-value threshold above which a series is considered stationary.
    handle : cuml.Handle (default=None)
        If it is None, a new one is created just for this function call.

    Returns
    -------
    stationarity : List[bool]
        A list of the stationarity test result for each series in the batch
    """
    if(isinstance(y, cumlArray)):
        d_y = y
        n_obs = d_y.shape[0]
        batch_size = d_y.shape[1] if len(d_y.shape) > 1 else 1
        dtype = d_y.dtype
    else:
        d_y, n_obs, batch_size, dtype = \
            input_to_cuml_array(y, check_dtype=[np.float32, np.float64])
    cdef uintptr_t d_y_ptr = d_y.ptr

    if handle is None:
        handle = cuml.common.handle.Handle()
    cdef cumlHandle* handle_ = <cumlHandle*><size_t>handle.getHandle()

    results = cumlArray.empty(batch_size, dtype=np.bool)
    cdef uintptr_t d_results = results.ptr

    # Call C++ function
    if dtype == np.float32:
        cpp_kpss(handle_[0],
                 <float*> d_y_ptr,
                 <bool*> d_results,
                 <int> batch_size,
                 <int> n_obs,
                 <int> d, <int> D, <int> s,
                 <float> pval_threshold)
    elif dtype == np.float64:
        cpp_kpss(handle_[0],
                 <double*> d_y_ptr,
                 <bool*> d_results,
                 <int> batch_size,
                 <int> n_obs,
                 <int> d, <int> D, <int> s,
                 <double> pval_threshold)

    return results
