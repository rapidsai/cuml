#
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

from random import randint

import numpy as np
from pylibraft.common.handle import Handle

import cuml.internals
from cuml.internals.array import CumlArray as cumlArray

from libc.stdint cimport uint64_t, uintptr_t
from pylibraft.common.handle cimport handle_t

from cuml.tsa.arima cimport ARIMAOrder


cdef extern from "cuml/datasets/make_arima.hpp" namespace "ML" nogil:
    void cpp_make_arima "ML::Datasets::make_arima" (
        const handle_t& handle,
        float* out,
        int batch_size,
        int n_obs,
        ARIMAOrder order,
        float scale,
        float noise_scale,
        float intercept_scale,
        uint64_t seed
    ) except +

    void cpp_make_arima "ML::Datasets::make_arima" (
        const handle_t& handle,
        double* out,
        int batch_size,
        int n_obs,
        ARIMAOrder order,
        double scale,
        double noise_scale,
        double intercept_scale,
        uint64_t seed
    ) except +


inp_to_dtype = {
    'single': np.float32,
    'float': np.float32,
    'double': np.float64,
    np.float32: np.float32,
    np.float64: np.float64
}


@cuml.internals.api_return_array()
def make_arima(batch_size=1000, n_obs=100, order=(1, 1, 1),
               seasonal_order=(0, 0, 0, 0), intercept=False,
               random_state=None, dtype='double',
               handle=None):
    """Generates a dataset of time series by simulating an ARIMA process
    of a given order.

    Examples
    --------
    .. code-block:: python

        from cuml.datasets import make_arima
        y = make_arima(1000, 100, (2,1,2), (0,1,2,12), 0)

    Parameters
    ----------
    batch_size: int
        Number of time series to generate
    n_obs: int
        Number of observations per series
    order : Tuple[int, int, int]
        Order (p, d, q) of the simulated ARIMA process
    seasonal_order: Tuple[int, int, int, int]
        Seasonal ARIMA order (P, D, Q, s) of the simulated ARIMA process
    intercept: bool or int
        Whether to include a constant trend mu in the simulated ARIMA process
    random_state: int, RandomState instance or None (default)
        Seed for the random number generator for dataset creation.
    dtype: string or numpy dtype (default: 'single')
        Type of the data. Possible values: float32, float64, 'single', 'float'
        or 'double'

    handle: cuml.Handle
        If it is None, a new one is created just for this function call

    Returns
    -------
    out: array-like, shape (n_obs, batch_size)
        Array of the requested type containing the generated dataset
    """

    cdef ARIMAOrder cpp_order
    cpp_order.p, cpp_order.d, cpp_order.q = order
    cpp_order.P, cpp_order.D, cpp_order.Q, cpp_order.s = seasonal_order
    cpp_order.k = <int>intercept
    cpp_order.n_exog = 0

    # Set the default output type to "cupy". This will be ignored if the user
    # has set `cuml.global_settings.output_type`. Only necessary for array
    # generation methods that do not take an array as input
    cuml.internals.set_api_output_type("cupy")

    # Define some parameters based on the order
    scale = 1.0
    noise_scale = 0.2
    intercept_scale = [1.0, 0.2, 0.01][cpp_order.d + cpp_order.D]

    if dtype not in ['single', 'float', 'double', np.float32, np.float64]:
        raise TypeError("dtype must be either 'float' or 'double'")
    else:
        dtype = inp_to_dtype[dtype]

    handle = Handle() if handle is None else handle
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

    out = cumlArray.empty((n_obs, batch_size), dtype=dtype, order='F')
    cdef uintptr_t out_ptr = <uintptr_t> out.ptr

    if random_state is None:
        random_state = randint(0, 10**18)

    if dtype == np.float32:
        cpp_make_arima(handle_[0], <float*> out_ptr, <int> batch_size,
                       <int> n_obs, cpp_order, <float> scale,
                       <float> noise_scale, <float> intercept_scale,
                       <uint64_t> random_state)

    else:
        cpp_make_arima(handle_[0], <double*> out_ptr, <int> batch_size,
                       <int> n_obs, cpp_order, <double> scale,
                       <double> noise_scale, <double> intercept_scale,
                       <uint64_t> random_state)

    return out
