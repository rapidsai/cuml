# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import numbers
import warnings

import cupyx.scipy.sparse
import numpy as np
import scipy.sparse
from sklearn.exceptions import DataConversionWarning
from sklearn.utils.validation import check_is_fitted

from cuml.internals.array import CumlArray
from cuml.internals.array_sparse import SparseCumlArray
from cuml.internals.input_utils import input_to_cuml_array

__all__ = (
    "check_X",
    "check_consistent_length",
    "check_inputs",
    "check_is_fitted",
    "check_sample_weight",
    "check_y",
)


def check_consistent_length(*arrays):
    """Check whether all inputs have the same number of samples"""
    lengths = [X.shape[0] for X in arrays if X is not None]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"Found input variables with inconsistent numbers of samples: "
            f"{[int(n) for n in lengths]}"
        )


def check_X(
    estimator,
    X,
    *,
    dtype=(np.float32, np.float64),
    convert_dtype=True,
    order="K",
    min_samples=1,
    min_features=1,
    accept_sparse=False,
    reset=False,
):
    if (shape := getattr(X, "shape", None)) is not None:
        ndim = len(shape)
        if ndim == 0:
            raise ValueError("Expected 2D array, got scalar array instead.")
        elif ndim == 1:
            warnings.warn(
                "Passing a 1D array for X was deprecated in 26.04 "
                "and will be removed in 26.06. Please reshape X to "
                "have 2 dimensions",
                FutureWarning,
            )
        elif ndim > 2:
            raise ValueError(
                f"Found array with dim {ndim}, while dim <= 2 is required"
            )

    if not isinstance(dtype, (list, tuple)):
        dtype = [dtype]

    if scipy.sparse.issparse(X) or cupyx.scipy.sparse.issparse(X):
        if not accept_sparse:
            raise TypeError(
                "Sparse data was passed, but dense data is required. "
                "Use '.toarray()' to convert to a dense numpy array."
            )
        out = SparseCumlArray(
            X, convert_to_dtype=(X.dtype if X.dtype in dtype else dtype[0])
        )
    else:
        out = input_to_cuml_array(
            X,
            convert_to_dtype=(dtype[0] if convert_dtype else None),
            check_dtype=dtype,
            order=order,
        ).array

    n_samples, n_features = out.shape if out.ndim == 2 else (out.shape[0], 1)

    if n_samples < min_samples:
        raise ValueError(
            f"Found array with {n_samples} sample(s) (shape={out.shape}) while a "
            f"minimum of {n_samples} is required."
        )
    if n_features < min_features:
        raise ValueError(
            f"Found array with {n_features} feature(s) (shape={out.shape}) while a "
            f"minimum of {n_features} is required."
        )

    if reset:
        estimator._set_output_type(X)
        estimator._set_features(X)
    else:
        estimator._check_features(X)

    return out


def check_y(
    y,
    *,
    dtype=(np.float32, np.float64),
    convert_dtype=True,
    order="K",
    accept_multi_output=False,
):
    if not isinstance(dtype, (list, tuple)):
        dtype = [dtype]

    out = input_to_cuml_array(
        y,
        check_dtype=dtype,
        convert_to_dtype=(dtype[0] if convert_dtype else None),
        order=order,
    ).array

    if out.ndim == 1:
        return out
    elif out.ndim == 2:
        if accept_multi_output:
            return out
        elif out.shape[1] == 1:
            warnings.warn(
                "A column-vector y was passed when a 1d array was "
                "expected. Please change the shape of y to "
                "(n_samples,), for example using ravel().",
                DataConversionWarning,
            )
            return out
    raise ValueError(
        f"y should be a {'1d or 2d' if accept_multi_output else '1d'} array, "
        f"got an array of shape {out.shape} instead."
    )


def check_sample_weight(
    sample_weight,
    n_samples,
    *,
    dtype=(np.float32, np.float64),
    convert_dtype=True,
):
    if not isinstance(dtype, (list, tuple)):
        dtype = [dtype]

    if isinstance(sample_weight, numbers.Number):
        return CumlArray.full(n_samples, sample_weight, dtype=dtype[0])
    else:
        out = input_to_cuml_array(
            sample_weight,
            check_dtype=dtype,
            convert_to_dtype=(dtype[0] if convert_dtype else None),
        ).array

        if out.ndim != 1:
            raise ValueError(
                f"Sample weights must be 1D array or scalar, got "
                f"{out.ndim}D array. Expected either a scalar value "
                f"or a 1D array of length {n_samples}."
            )
        return out


def check_inputs(
    estimator,
    X,
    y=...,
    sample_weight=...,
    *,
    dtype=(np.float32, np.float64),
    convert_dtype=True,
    order="K",
    min_samples=1,
    min_features=1,
    accept_sparse=False,
    accept_multi_output=False,
    reset=False,
):
    X = check_X(
        estimator,
        X,
        dtype=dtype,
        convert_dtype=convert_dtype,
        order=order,
        min_samples=min_samples,
        min_features=min_features,
        accept_sparse=accept_sparse,
        reset=reset,
    )
    arrays = [X]

    if y is not ...:
        if y is None:
            raise ValueError(
                f"This {type(estimator).__name__} requires y to be passed, "
                f"but the target y is None."
            )
        y = check_y(
            y,
            dtype=X.dtype,
            convert_dtype=convert_dtype,
            order=order,
            accept_multi_output=accept_multi_output,
        )
        arrays.append(y)

    if sample_weight is not ...:
        if sample_weight is not None:
            sample_weight = check_sample_weight(
                sample_weight,
                n_samples=X.shape[0],
                dtype=X.dtype,
                convert_dtype=convert_dtype,
            )
        arrays.append(sample_weight)

    check_consistent_length(*arrays)

    return tuple(arrays) if len(arrays) > 1 else arrays[0]
