#
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

import numpy as np

from libc.stdint cimport uintptr_t

import cuml.common.handle
from cuml.common.handle cimport cumlHandle
from cuml.metrics cimport regression
from cuml.utils import input_to_dev_array


def r2_score(y, y_hat, convert_dtype=False, handle=None):
    """
    Calculates r2 score between y and y_hat

    Parameters
    ----------
        y : array-like (device or host) shape = (n_samples, 1)
            Dense vector (floats or doubles) of shape (n_samples, 1).
            Acceptable formats: cuDF Series, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        y_hat : array-like (device or host) shape = (n_samples, 1)
            Dense vector (floats or doubles) of shape (n_samples, 1).
            Acceptable formats: cuDF Series, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        convert_dtype : bool (default = False)
            When set to True, the fit method will automatically convert
            y_hat to be the same data type as y if they differ. This
            will increase memory used for the method.

    Returns
    -------
        trustworthiness score : double
            Trustworthiness of the low-dimensional embedding
    """
    handle = cuml.common.handle.Handle() if handle is None else handle
    cdef cumlHandle* handle_ = <cumlHandle*><size_t>handle.getHandle()

    cdef uintptr_t y_ptr
    cdef uintptr_t y_hat_ptr

    y_m, y_ptr, n_rows, _, ytype = \
        input_to_dev_array(y, check_dtype=[np.float32, np.float64],
                           check_cols=1)

    y_m2, y_hat_ptr, _, _, _ = \
        input_to_dev_array(y_hat, check_dtype=ytype,
                           convert_to_dtype=(ytype if convert_dtype
                                             else None),
                           check_rows=n_rows, check_cols=1)

    cdef float result_f32
    cdef double result_f64

    n = len(y)

    if y.dtype == 'float32':

        result_f32 = regression.r2_score_py(handle_[0],
                                            <float*> y_ptr,
                                            <float*> y_hat_ptr,
                                            <int> n)

        result = result_f32

    else:
        result_f64 = regression.r2_score_py(handle_[0],
                                            <double*> y_ptr,
                                            <double*> y_hat_ptr,
                                            <int> n)

        result = result_f64

    del y_m
    del y_m2

    return result
