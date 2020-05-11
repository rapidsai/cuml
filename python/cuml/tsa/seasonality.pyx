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
from cuml.common.base import _input_to_type
from cuml.common.handle cimport cumlHandle
from cuml.common.input_utils import input_to_host_array, input_to_cuml_array

# TODO: #2234 and #2235


def python_seas_test(y, batch_size, n_obs, s, threshold=0.64):
    """Python prototype to be ported later in CUDA
    """
    # TODO: our own implementation of STL
    from statsmodels.tsa.seasonal import STL

    results = []
    for i in range(batch_size):
        stlfit = STL(y[:, i], s).fit()
        seasonal = stlfit.seasonal
        residual = stlfit.resid
        heuristics = max(
            0, min(1, 1 - np.var(residual)/ np.var(residual + seasonal)))
        results.append(heuristics > threshold)

    return results


def seas_test(y, s, output_type="input", handle=None):
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

    if output_type == "input":
        output_type = _input_to_type(y)

    # At the moment we use a host array
    h_y, _, n_obs, batch_size, dtype = \
        input_to_host_array(y, check_dtype=[np.float32, np.float64])

    # Temporary: Python implementation
    python_res = python_seas_test(h_y, batch_size, n_obs, s)
    d_res, *_ = input_to_cuml_array(np.array(python_res), check_dtype=np.bool)
    return d_res.to_output(output_type)
