#
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

import warnings

import cuml.internals
from cuml.internals.input_utils import input_to_cupy_array
from cuml.internals.safe_imports import cpu_only_import, gpu_only_import

np = cpu_only_import("numpy")
cp = gpu_only_import("cupy")


def _normalize_regression_metric_args(
    y_true, y_pred, sample_weight, multioutput
):
    """
    Helper function to normalize inputs to a regression metric.

    Validates inputs and coerces all arrays to cupy of proper shape and dtype.
    """
    # Coerce inputs to cupy arrays
    float_or_int = [np.float32, np.float64, np.int32, np.int64]
    y_true, n_rows, n_cols, _ = input_to_cupy_array(
        y_true, check_dtype=float_or_int
    )
    y_pred, _, _, _ = input_to_cupy_array(
        y_pred, check_dtype=float_or_int, check_rows=n_rows, check_cols=n_cols
    )
    if sample_weight is not None:
        sample_weight, _, _, _ = input_to_cupy_array(
            sample_weight,
            check_dtype=float_or_int,
            check_rows=n_rows,
            check_cols=1,
        )

    # Ensure y_true & y_pred are 2D and sample_weight is 1D
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))

    if sample_weight is not None:
        sample_weight = sample_weight.reshape(-1)

    # Validate multioutput, and maybe coerce to a cupy array
    valid_multioutput = ("raw_values", "uniform_average", "variance_weighted")
    if isinstance(multioutput, str):
        if multioutput not in valid_multioutput:
            raise ValueError(
                f"Valid `multioutput` values are {valid_multioutput}, got {multioutput=}"
            )
    elif multioutput is not None:
        if n_cols == 1:
            raise ValueError(
                "Custom weights are useful only in multi-output cases."
            )
        multioutput, _, _, _ = input_to_cupy_array(
            multioutput, check_rows=n_cols
        )

    return y_true, y_pred, sample_weight, multioutput


@cuml.internals.api_return_any()
def r2_score(
    y_true,
    y_pred,
    *,
    sample_weight=None,
    multioutput="uniform_average",
    force_finite=True,
    **kwargs,
):
    """:math:`R^2` (coefficient of determination) regression score function.

    Best possible score is 1.0 and it can be negative (because the
    model can be arbitrarily worse). In the general case when the true y is
    non-constant, a constant model that always predicts the average y
    disregarding the input features would get a :math:`R^2` score of 0.0.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,)
        Sample weights.
    multioutput : {'raw_values', 'uniform_average', 'variance_weighted'} or array-like of shape (n_outputs,)
        How to aggregate multioutput scores. One of:

        - 'uniform_average': Scores of all outputs are averaged with uniform weight.
          This is the default.
        - 'variance_weighted': Scores of all outputs are averaged, weighted by the
          variances of each individual output.
        - 'raw_values': Full set of scores in case of multioutput input.
        - array-like: Weights to use when averaging scores of all outputs.

    force_finite : bool, default=True
        Flag indicating if ``NaN`` and ``-Inf`` scores resulting from constant
        data should be replaced with real numbers (``1.0`` if prediction is
        perfect, ``0.0`` otherwise). Default is ``True``.

    Returns
    -------
    z : float or ndarray of floats
        The :math:`R^2` score or ndarray of scores if 'multioutput' is
        'raw_values'.
    """
    if kwargs:
        warnings.warn(
            "`convert_dtype` and `handle` were deprecated from `r2_score` in version "
            "25.02.01 and will be removed in 25.06.",
            FutureWarning,
        )

    (
        y_true,
        y_pred,
        sample_weight,
        multioutput,
    ) = _normalize_regression_metric_args(
        y_true, y_pred, sample_weight, multioutput
    )

    weight = 1.0 if sample_weight is None else sample_weight[:, None]
    numerator = cp.sum(weight * (y_true - y_pred) ** 2, axis=0)
    denominator = cp.sum(
        weight
        * (y_true - cp.average(y_true, axis=0, weights=sample_weight)) ** 2,
        axis=0,
    )

    nonzero_denominator = denominator != 0

    if not force_finite:
        output_scores = 1 - (numerator / denominator)
    else:
        # numerator == 0 -> 1
        # denominator == 0 -> 0
        # else -> 1 - (numerator / denominator)
        nonzero_numerator = numerator != 0
        output_scores = cp.ones([y_true.shape[1]], dtype=numerator.dtype)
        valid_score = nonzero_denominator & nonzero_numerator
        output_scores[valid_score] = 1 - (
            numerator[valid_score] / denominator[valid_score]
        )
        output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_scores
        elif multioutput == "uniform_average":
            avg_weights = None
        elif multioutput == "variance_weighted":
            avg_weights = denominator
            if not cp.any(nonzero_denominator):
                # All weights are zero, _average would raise a ZeroDiv error.
                # This only happens when all y are constant (or 1-element long)
                # Since weights are all equal, fall back to uniform weights.
                avg_weights = None
    else:
        avg_weights = multioutput

    result = cp.average(output_scores, weights=avg_weights)
    if result.size == 1:
        return float(result)
    return result


def _mse(y_true, y_pred, sample_weight, multioutput, squared):
    """Helper to compute the mean squared error"""
    output_errors = cp.average(
        (y_true - y_pred) ** 2, axis=0, weights=sample_weight
    )

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            multioutput = None

    out = cp.average(output_errors, weights=multioutput)
    return float(out if squared else cp.sqrt(out))


@cuml.internals.api_return_any()
def mean_squared_error(
    y_true,
    y_pred,
    sample_weight=None,
    multioutput="uniform_average",
    squared=True,
):
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
    (
        y_true,
        y_pred,
        sample_weight,
        multioutput,
    ) = _normalize_regression_metric_args(
        y_true, y_pred, sample_weight, multioutput
    )
    return _mse(y_true, y_pred, sample_weight, multioutput, squared)


@cuml.internals.api_return_any()
def mean_absolute_error(
    y_true, y_pred, sample_weight=None, multioutput="uniform_average"
):
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
    (
        y_true,
        y_pred,
        sample_weight,
        multioutput,
    ) = _normalize_regression_metric_args(
        y_true, y_pred, sample_weight, multioutput
    )

    output_errors = cp.average(
        cp.abs(y_pred - y_true), axis=0, weights=sample_weight
    )
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            multioutput = None

    out = cp.average(output_errors, weights=multioutput)
    return float(out)


@cuml.internals.api_return_any()
def mean_squared_log_error(
    y_true,
    y_pred,
    sample_weight=None,
    multioutput="uniform_average",
    squared=True,
):
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
    (
        y_true,
        y_pred,
        sample_weight,
        multioutput,
    ) = _normalize_regression_metric_args(
        y_true, y_pred, sample_weight, multioutput
    )

    if cp.less(y_true, 0).any() or cp.less(y_pred, 0).any():
        raise ValueError(
            "Mean Squared Logarithmic Error cannot be used when "
            "targets contain negative values."
        )

    return _mse(
        cp.log1p(y_true),
        cp.log1p(y_pred),
        sample_weight,
        multioutput,
        squared,
    )
