#
# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
import cupy as cp

from libc.stdint cimport uintptr_t

import cuml.internals
from cuml.internals.array import CumlArray
from pylibraft.common.handle import Handle
from pylibraft.common.handle cimport handle_t
from cuml.metrics cimport regression
from cuml.internals.input_utils import input_to_cuml_array


@cuml.internals.api_return_any()
def r2_score(y, y_hat, convert_dtype=True, handle=None) -> double:
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

        convert_dtype : bool, optional (default = False)
            When set to True, the fit method will, when necessary, convert
            y_hat to be the same data type as y if they differ. This
            will increase memory used for the method.

    Returns
    -------
        trustworthiness score : double
            Trustworthiness of the low-dimensional embedding
    """
    handle = Handle() if handle is None else handle
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

    y_m, n_rows, _, ytype = \
        input_to_cuml_array(y, check_dtype=[np.float32, np.float64],
                            check_cols=1)
    cdef uintptr_t y_ptr = y_m.ptr

    y_m2, *_ = \
        input_to_cuml_array(y_hat, check_dtype=ytype,
                            convert_to_dtype=(ytype if convert_dtype
                                              else None),
                            check_rows=n_rows, check_cols=1)
    cdef uintptr_t y_hat_ptr = y_m2.ptr

    cdef float result_f32
    cdef double result_f64

    n = len(y)

    if y_m.dtype == 'float32':

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


def _prepare_input_reg(y_true, y_pred, sample_weight, multioutput):
    """
    Helper function to avoid code duplication for regression metrics.
    Converts inputs to CumlArray and check multioutput parameter validity.
    """
    allowed_d_types = [np.float32, np.float64, np.int32, np.int64]
    y_true = y_true.squeeze() if len(y_true.shape) > 1 else y_true
    y_true, n_rows, n_cols, ytype = \
        input_to_cuml_array(y_true, check_dtype=allowed_d_types)

    y_pred = y_pred.squeeze() if len(y_pred.shape) > 1 else y_pred
    y_pred, _, _, _ = \
        input_to_cuml_array(y_pred, check_dtype=allowed_d_types,
                            check_rows=n_rows, check_cols=n_cols)

    if sample_weight is not None:
        sample_weight, _, _, _ = \
            input_to_cuml_array(sample_weight, check_dtype=allowed_d_types,
                                check_rows=n_rows, check_cols=n_cols)

    raw_multioutput = False
    allowed_multioutput_str = ('raw_values', 'uniform_average',
                               'variance_weighted')
    if isinstance(multioutput, str):
        if multioutput not in allowed_multioutput_str:
            raise ValueError("Allowed 'multioutput' string values are {}. "
                             "You provided multioutput={!r}"
                             .format(allowed_multioutput_str, multioutput))
        elif multioutput == 'raw_values':
            raw_multioutput = True
        elif multioutput == 'uniform_average':
            # pass None as weights to np.average: uniform mean
            multioutput = None
    elif multioutput is not None:
        multioutput, _, _, _ = \
            input_to_cuml_array(multioutput, check_dtype=allowed_d_types)
        if n_cols == 1:
            raise ValueError("Custom weights are useful only in "
                             "multi-output cases.")

    return y_true, y_pred, sample_weight, multioutput, raw_multioutput


def _mse(y_true, y_pred, sample_weight, multioutput, squared, raw_multioutput):
    """Helper to compute the mean squared error"""
    output_errors = cp.subtract(y_true, y_pred)
    output_errors = cp.multiply(output_errors, output_errors)

    output_errors = cp.average(output_errors, axis=0, weights=sample_weight)

    if raw_multioutput:
        return output_errors

    mse = cp.average(output_errors, weights=multioutput)
    return mse if squared else cp.sqrt(mse)


@cuml.internals.api_return_any()
def mean_squared_error(y_true, y_pred,
                       sample_weight=None,
                       multioutput='uniform_average',
                       squared=True):
    """Mean squared error regression loss

    Be careful when using this metric with float32 inputs as the result can be
    slightly incorrect because of floating point precision if the input is
    large enough. float64 will have lower numerical error.

    Parameters
    ----------
    y_true : array-like (device or host) shape = (n_samples,)
        or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like (device or host) shape = (n_samples,)
        or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like (device or host) shape = (n_samples,), optional
        Sample weights.
    multioutput : string in ['raw_values', 'uniform_average'] \
            (default='uniform_average')
        or array-like of shape (n_outputs)
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        'raw_values' :
        Returns a full set of errors in case of multioutput input.
        'uniform_average' :
        Errors of all outputs are averaged with uniform weight.
    squared : boolean value, optional (default = True)
        If True returns MSE value, if False returns RMSE value.

    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.
    """
    y_true, y_pred, sample_weight, multioutput, raw_multioutput = \
        _prepare_input_reg(y_true, y_pred, sample_weight, multioutput)

    return _mse(y_true, y_pred, sample_weight, multioutput, squared,
                raw_multioutput)


@cuml.internals.api_return_any()
def mean_absolute_error(y_true, y_pred,
                        sample_weight=None,
                        multioutput='uniform_average'):
    """Mean absolute error regression loss

    Be careful when using this metric with float32 inputs as the result can be
    slightly incorrect because of floating point precision if the input is
    large enough. float64 will have lower numerical error.

    Parameters
    ----------
    y_true : array-like (device or host) shape = (n_samples,)
        or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like (device or host) shape = (n_samples,)
        or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like (device or host) shape = (n_samples,), optional
        Sample weights.
    multioutput : string in ['raw_values', 'uniform_average']
        or array-like of shape (n_outputs)
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        'raw_values' :
        Returns a full set of errors in case of multioutput input.
        'uniform_average' :
        Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float or ndarray of floats
        If multioutput is ‘raw_values’, then mean absolute error is returned
        for each output separately. If multioutput is ‘uniform_average’ or an
        ndarray of weights, then the weighted average of all output errors is
        returned.

        MAE output is non-negative floating point. The best value is 0.0.
    """
    y_true, y_pred, sample_weight, multioutput, raw_multioutput = \
        _prepare_input_reg(y_true, y_pred, sample_weight, multioutput)

    output_errors = cp.abs(cp.subtract(y_pred, y_true))
    output_errors = cp.average(output_errors, axis=0, weights=sample_weight)

    if raw_multioutput:
        return output_errors

    return cp.average(output_errors, weights=multioutput)


@cuml.internals.api_return_any()
def mean_squared_log_error(y_true, y_pred,
                           sample_weight=None,
                           multioutput='uniform_average',
                           squared=True):
    """Mean squared log error regression loss

    Be careful when using this metric with float32 inputs as the result can be
    slightly incorrect because of floating point precision if the input is
    large enough. float64 will have lower numerical error.

    Parameters
    ----------
    y_true : array-like (device or host) shape = (n_samples,)
        or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like (device or host) shape = (n_samples,)
        or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like (device or host) shape = (n_samples,), optional
        Sample weights.
    multioutput : string in ['raw_values', 'uniform_average']
        or array-like of shape (n_outputs)
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        'raw_values' :
        Returns a full set of errors in case of multioutput input.
        'uniform_average' :
        Errors of all outputs are averaged with uniform weight.
    squared : boolean value, optional (default = True)
        If True returns MSE value, if False returns RMSE value.

    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.
    """
    y_true, y_pred, sample_weight, multioutput, raw_multioutput = \
        _prepare_input_reg(y_true, y_pred, sample_weight, multioutput)

    if cp.less(y_true, 0).any() or cp.less(y_pred, 0).any():
        raise ValueError("Mean Squared Logarithmic Error cannot be used when "
                         "targets contain negative values.")

    return _mse(cp.log1p(y_true), cp.log1p(y_pred), sample_weight, multioutput,
                squared, raw_multioutput)
