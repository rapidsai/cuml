#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from random import randint

import cupy as cp

from cuml.internals import get_handle, mlfunc
from cuml.tsa._deprecation import deprecated_tsa_api

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


@deprecated_tsa_api("cuml.datasets.make_arima")
@mlfunc(array_arg=None)
def make_arima(batch_size=1000, n_obs=100, order=(1, 1, 1),
               seasonal_order=(0, 0, 0, 0), intercept=False,
               random_state=None, dtype="float64"):
    """Generates a dataset of time series by simulating an ARIMA process
    of a given order.

    .. deprecated:: 26.08
        ``cuml.datasets.make_arima`` is deprecated and will be removed
        in the cuML 26.12 release.

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
    dtype: string or numpy dtype (default: 'float64')
        The output dtype. Only float32 or float64 supported.

    Returns
    -------
    out: array-like, shape (n_obs, batch_size)
        Array of the requested type containing the generated dataset
    """
    dtype = cp.dtype(dtype)
    if dtype not in ["float32", "float64"]:
        raise ValueError(f"Expected dtype in ['float32', 'float64'], got `{dtype!s}`")

    cdef ARIMAOrder cpp_order
    cpp_order.p, cpp_order.d, cpp_order.q = order
    cpp_order.P, cpp_order.D, cpp_order.Q, cpp_order.s = seasonal_order
    cpp_order.k = <int>intercept
    cpp_order.n_exog = 0

    # Define some parameters based on the order
    scale = 1.0
    noise_scale = 0.2
    intercept_scale = [1.0, 0.2, 0.01][cpp_order.d + cpp_order.D]

    handle = get_handle()
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

    out = cp.empty((n_obs, batch_size), dtype=dtype, order='F')
    cdef uintptr_t out_ptr = <uintptr_t> out.data.ptr

    if random_state is None:
        random_state = randint(0, 10**18)

    if dtype == "float32":
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
