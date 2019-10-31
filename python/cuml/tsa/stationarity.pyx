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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import cudf
import ctypes
import numpy as np
from libcpp.vector cimport vector
from libc.stdint cimport uintptr_t

import cuml
from cuml.common.handle cimport cumlHandle
from cuml.utils.input_utils import input_to_dev_array


cdef extern from "cuml/tsa/stationarity.h" namespace "ML":
    int cpp_stationarity "ML::Stationarity::stationarity" (
        const cumlHandle& handle,
        const float* y_d,
        int* d,
        int n_batches,
        int n_samples,
        float pval_threshold)

    int cpp_stationarity "ML::Stationarity::stationarity" (
        const cumlHandle& handle,
        const double* y_d,
        int* d,
        int n_batches,
        int n_samples,
        double pval_threshold)


def stationarity(y, pval_threshold=0.05, handle=None):
    """
    Compute recommended trend parameter (d=0 or 1) for a batched series

    Example
    -------
    .. code-block:: python

        import numpy as np
        from cuml.tsa.stationarity import stationarity

        num_samples = 200
        xs = np.linspace(0, 1, num_samples)
        np.random.seed(12)
        noise = np.random.normal(scale=0.1, size=num_samples)
        ys1 = noise + 0.5*xs  # d = 1
        ys2 = noise           # d = 0

        num_batches = 2
        ys_df = np.zeros((num_samples, num_batches), order="F")
        ys_df[:, 0] = ys1
        ys_df[:, 1] = ys2

        d_b = stationarity(ys_df)
        print(d_b)

    Output:

    .. code-block:: none

        [1, 0]

    Parameters
    ----------
    y : array-like (device or host)
        Batched series to compute the trend parameters of.
        Acceptable formats: cuDF DataFrame, cuDF Series, NumPy ndarray,
        numba device ndarray, cuda array interface compliant array like CuPy.
        Note: cuDF.DataFrame types assumes data is in columns, while all other
        datatypes assume data is in rows.
    pval_threshold : float
                     The p-value threshold above which a series is considered
                     stationary.
    handle : cuml.Handle (default=None)
             If it is None, a new one is created just for this function call.

    Returns
    -------
    stationarity : list[int]
                   The recommended `d` for each series

    """
    cdef uintptr_t y_d_ptr
    y_d, y_d_ptr, n_samples, n_batches, dtype = \
        input_to_dev_array(y, check_dtype=[np.float32, np.float64])

    if handle is None:
        handle = cuml.common.handle.Handle()
    cdef cumlHandle* handle_ = <cumlHandle*><size_t>handle.getHandle()

    cdef vector[int] d
    d.resize(n_batches)

    # Call C++ function
    if dtype == np.float32:
        ret_value = cpp_stationarity(handle_[0],
                                     <float*> y_d_ptr,
                                     <int*> d.data(),
                                     <int> n_batches,
                                     <int> n_samples,
                                     <float> pval_threshold)
    elif dtype == np.float64:
        ret_value = cpp_stationarity(handle_[0],
                                     <double*> y_d_ptr,
                                     <int*> d.data(),
                                     <int> n_batches,
                                     <int> n_samples,
                                     <double> pval_threshold)

    if ret_value < 0:
        raise ValueError("Stationarity test failed for d=0 or 1.")

    return d
