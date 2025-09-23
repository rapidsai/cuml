# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

# distutils: language = c++

import numpy as np

from libc.stdint cimport uintptr_t
from libcpp cimport bool as boolcpp

import cuml.internals
from cuml.internals.array import CumlArray

from pylibraft.common.handle cimport handle_t

from pylibraft.common.handle import Handle

from cuml.internals.input_utils import input_to_cuml_array


cdef extern from "cuml/tsa/stationarity.h" namespace "ML" nogil:
    int cpp_kpss "ML::Stationarity::kpss_test" (
        const handle_t& handle,
        const float* d_y,
        boolcpp* results,
        int batch_size,
        int n_obs,
        int d, int D, int s,
        float pval_threshold) except +

    int cpp_kpss "ML::Stationarity::kpss_test" (
        const handle_t& handle,
        const double* d_y,
        boolcpp* results,
        int batch_size,
        int n_obs,
        int d, int D, int s,
        double pval_threshold) except +


@cuml.internals.api_return_array(input_arg="y", get_output_type=True)
def kpss_test(y, d=0, D=0, s=0, pval_threshold=0.05,
              handle=None, convert_dtype=True) -> CumlArray:
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
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.

    Returns
    -------
    stationarity : List[bool]
        A list of the stationarity test result for each series in the batch
    """
    d_y, n_obs, batch_size, dtype = \
        input_to_cuml_array(y,
                            convert_to_dtype=(np.float32 if convert_dtype
                                              else None),
                            check_dtype=[np.float32, np.float64])
    cdef uintptr_t d_y_ptr = d_y.ptr

    if handle is None:
        handle = Handle()
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

    results = CumlArray.empty(batch_size, dtype=bool)
    cdef uintptr_t d_results = results.ptr

    # Call C++ function
    if dtype == np.float32:
        cpp_kpss(handle_[0],
                 <float*> d_y_ptr,
                 <boolcpp*> d_results,
                 <int> batch_size,
                 <int> n_obs,
                 <int> d, <int> D, <int> s,
                 <float> pval_threshold)
    elif dtype == np.float64:
        cpp_kpss(handle_[0],
                 <double*> d_y_ptr,
                 <boolcpp*> d_results,
                 <int> batch_size,
                 <int> n_obs,
                 <int> d, <int> D, <int> s,
                 <double> pval_threshold)

    return results
