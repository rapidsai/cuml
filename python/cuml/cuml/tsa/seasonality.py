# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from cuml.internals import get_handle, reflect
from cuml.internals.array import CumlArray
from cuml.internals.input_utils import input_to_cuml_array, input_to_host_array


def python_seas_test(y, batch_size, n_obs, s, threshold=0.64):
    """Python prototype to be ported later in CUDA"""
    # TODO: our own implementation of STL
    from statsmodels.tsa.seasonal import STL

    results = []
    for i in range(batch_size):
        stlfit = STL(y[:, i], s).fit()
        seasonal = stlfit.seasonal
        residual = stlfit.resid
        heuristics = max(
            0, min(1, 1 - np.var(residual) / np.var(residual + seasonal))
        )
        results.append(heuristics > threshold)

    return results


@reflect
def seas_test(y, s, handle=None, convert_dtype=True) -> CumlArray:
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
            "ERROR: Invalid period for the seasonal differencing test: {}".format(
                s
            )
        )
    # `handle` is fully unused in this function - calling `get_handle` here just
    # to raise the uniform deprecation warning
    get_handle(handle=handle)

    # At the moment we use a host array
    h_y, n_obs, batch_size, _ = input_to_host_array(
        y,
        convert_to_dtype=(np.float32 if convert_dtype else None),
        check_dtype=[np.float32, np.float64],
    )

    python_res = python_seas_test(h_y, batch_size, n_obs, s)
    d_res, *_ = input_to_cuml_array(
        np.array(python_res),
        convert_to_dtype=(bool if convert_dtype else None),
        check_dtype=bool,
    )
    return d_res
