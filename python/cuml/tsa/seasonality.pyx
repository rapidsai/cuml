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

# TODO: the following tests would be nice to have:
# - CH (Canova & Hansen, 1995)
# - OCSB (Osborn, Chui, Smith & Birchenhall, 1988)
# - HEGY? (Hylleberg, Engle, Granger & Yoo, 1990)


# cdef extern from "cuml/tsa/seasonality.h" namespace "ML":
#     int cpp_seas_test "ML::Seasonality::seas_test" (
#         const cumlHandle& handle,
#         const float* d_y,
#         bool* results,
#         int batch_size,
#         int n_obs,
#         int s)

#     int cpp_seas_test "ML::Seasonality::seas_test" (
#         const cumlHandle& handle,
#         const double* d_y,
#         bool* results,
#         int batch_size,
#         int n_obs,
#         int s)


def python_seas_test(y, batch_size, n_obs, s, threshold=0.64):
    """Python prototype to be ported later in CUDA
    """
    # TODO: our own implementation of STL
    from statsmodels.tsa.seasonal import STL

    results = []
    for i in range(batch_size):
        stlfit = STL(y[:,i], s).fit()
        seasonal = stlfit.seasonal
        residual = stlfit.resid
        heuristics = max(0, min(1, 1 - np.var(residual) / np.var(residual + seasonal)))
        results.append(heuristics > threshold)

    return results


def seas_test(y, s, handle=None):
    """
    Perform Wang, Smith & Hyndman's test to decide whether seasonal
    differencing is needed

    Parameters
    ----------
    y : dataframe or array-like (device or host)
        The time series data, assumed to have each time series in columns.
        Acceptable formats: cuDF DataFrame, cuDF Series, NumPy ndarray,
        Numba device ndarray, cuda array interface compliant array like CuPy.
    s: integer
        Seasonal period (s > 1)
    handle : cuml.Handle (default=None)
        If it is None, a new one is created just for this function call.

    Returns
    -------
    stationarity : List[bool]
        For each series in the batch, whether it needs seasonal differencing
    """
    if s <= 1:
        raise ValueError(
            "ERROR: Invalid period for the seasonal differencing test: {}"
            .format(s))

    d_y, n_obs, batch_size, dtype = \
        input_to_cuml_array(y, check_dtype=[np.float32, np.float64])
    cdef uintptr_t d_y_ptr = d_y.ptr

    h_y = d_y.to_output("numpy")

    # Temporary: Python implementation
    python_res = python_seas_test(h_y, batch_size, n_obs, s)
    d_res, *_ = input_to_cuml_array(np.array(python_res), check_dtype=np.bool)
    return d_res

    # if handle is None:
    #     handle = cuml.common.handle.Handle()
    # cdef cumlHandle* handle_ = <cumlHandle*><size_t>handle.getHandle()

    # results = cumlArray.empty(batch_size, dtype=np.bool)
    # cdef uintptr_t d_results = results.data()

    # # Call C++ function
    # if dtype == np.float32:
    #     cpp_seas_test(handle_[0],
    #                   <float*> d_y_ptr,
    #                   <bool*> d_results,
    #                   <int> batch_size,
    #                   <int> n_obs,
    #                   <int> s)
    # elif dtype == np.float64:
    #     cpp_seas_test(handle_[0],
    #                   <double*> d_y_ptr,
    #                   <bool*> d_results,
    #                   <int> batch_size,
    #                   <int> n_obs,
    #                   <int> s)

    # return results
