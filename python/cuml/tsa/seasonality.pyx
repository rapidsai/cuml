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

# distutils: language = c++

import ctypes
import numpy as np
from libc.stdint cimport uintptr_t
from libcpp cimport bool

import cuml.internals
from cuml.common.array import CumlArray
from cuml.raft.common.handle cimport handle_t
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


@cuml.internals.api_return_array(input_arg="y", get_output_type=True)
def seas_test(y, s, handle=None) -> CumlArray:
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
        For each series in the batch, whether it needs seasonal differencing
    """
    if s <= 1:
        raise ValueError(
            "ERROR: Invalid period for the seasonal differencing test: {}"
            .format(s))

    # At the moment we use a host array
    h_y, _, n_obs, batch_size, dtype = \
        input_to_host_array(y, check_dtype=[np.float32, np.float64])

    # Temporary: Python implementation
    python_res = python_seas_test(h_y, batch_size, n_obs, s)
    d_res, *_ = input_to_cuml_array(np.array(python_res), check_dtype=np.bool)
    return d_res
