# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import warnings

import cudf
import cupy as cp
import numpy as np
import pandas as pd

from cuml.internals.array import CumlArray
from cuml.internals.input_utils import input_to_cuml_array, input_to_cupy_array
from cuml.internals.memory_utils import cuda_ptr
from cuml.internals.output_utils import cudf_to_pandas

is_integral = cp.ReductionKernel(
    "T x",
    "bool out",
    "ceilf(x) == x",
    "a && b",
    "out = a",
    "true",
    "is_integral",
)


def check_classification_targets(y):
    """Check if `y` is composed of valid class labels"""
    if y.dtype.kind == "f" and not is_integral(y):
        raise ValueError(
            "Unknown label type: continuous. Maybe you are trying to fit a "
            "classifier, which expects discrete classes on a regression target "
            "with continuous values."
        )


def preprocess_labels(
    y, dtype=None, order="C", n_samples=None, allow_multitarget=False
):
    """Preprocess the `y` input to a classifier.

    Parameters
    ----------
    y : array-like
        The labels for fitting, may be any type cuml supports as input.
    dtype : dtype, optional
        The output dtype to use for the encoded labels. If not provided,
        a data-dependent integral type will be used.
    order : {"C", "F"}, optional
        The array order to use for the encoded labels.
    n_samples : int, optional
        If provided, will raise an error if the number of samples in `y`
        doesn't match.
    allow_multitarget : bool, optional
        Whether to allow multi-target labels.

    Returns
    -------
    y_encoded : cp.ndarray
        The labels, encoded as integers in [0, n_classes - 1].
    classes : np.ndarray or list[np.ndarray]
        The classes as a numpy array, or a list of numpy arrays if
        y is multi-target.
    """
    # cudf may coerce the dtype, store the original so we can cast back later
    y_dtype = y.dtype if isinstance(y, np.ndarray) else None

    # No cuda container supports all dtypes. Here we coerce to cupy when
    # possible, falling back to cudf Series/DataFrame otherwise.
    if isinstance(y, np.ndarray) and y.dtype.kind in "iufb":
        y = cp.asarray(y)
    elif isinstance(y, pd.DataFrame):
        y = cudf.DataFrame(y)
    elif isinstance(y, pd.Series):
        y = cudf.Series(y)
    elif not isinstance(y, (cp.ndarray, cudf.DataFrame, cudf.Series)):
        # Non-numeric dtype, always go through cudf
        y = input_to_cuml_array(y, convert_to_mem_type=False).array
        if y.dtype.kind in "iufb":
            y = y.to_output("cupy")
        else:
            y = (cudf.DataFrame if y.ndim == 2 else cudf.Series)(
                y, dtype=(np.dtype("O") if y.dtype.kind in "U" else None)
            )

    # Validate dimensionality, ensuring 1D/2D y is as expected
    if y.ndim == 2 and y.shape[1] == 1:
        warnings.warn(
            "A column-vector y was passed when a 1d array was expected. Please "
            "change the shape of y to (n_samples,), for example using ravel()."
        )
        y = y.iloc[:, 0] if isinstance(y, cudf.DataFrame) else y.ravel()
    elif allow_multitarget and y.ndim not in (1, 2):
        raise ValueError(
            f"y should be a 1d or 2d array, got an array of shape {y.shape} instead."
        )
    elif not allow_multitarget and y.ndim != 1:
        raise ValueError(
            f"y should be a 1d array, got an array of shape {y.shape} instead."
        )

    # Validate correct number of samples
    if n_samples is not None and y.shape[0] != n_samples:
        raise ValueError(
            f"Expected `y` with {n_samples} samples, got {y.shape[0]}"
        )

    def _encode(y):
        """Encode `y` to codes and classes"""
        check_classification_targets(y)
        if isinstance(y, cudf.Series):
            y = y.astype("category")
            codes = cp.asarray(y.cat.codes)
            classes = y.cat.categories.to_numpy()
            # cudf will sometimes translate non-numeric dtypes. Coerce back to
            # the input dtype if the input was originally a numpy array.
            if y_dtype is not None:
                classes = classes.astype(y_dtype, copy=False)
        else:
            classes, codes = cp.unique(y, return_inverse=True)
            classes = classes.get()
        return codes, classes

    if y.ndim == 1:
        y_encoded, classes = _encode(y)
        if dtype is not None:
            y_encoded = y_encoded.astype(dtype, copy=False)
    else:
        getter = y.iloc if isinstance(y, cudf.DataFrame) else y
        encoded_cols, classes = zip(
            *(_encode(getter[:, i]) for i in range(y.shape[1]))
        )
        classes = list(classes)
        if dtype is None:
            dtype = cp.result_type(*(c.dtype for c in encoded_cols))
        y_encoded = cp.empty(shape=y.shape, dtype=dtype, order=order)
        for i, col in enumerate(encoded_cols):
            y_encoded[:, i] = col

    return y_encoded, classes


def decode_labels(y_encoded, classes, output_type="cupy"):
    """Convert encoded labels back into their original classes.

    Parameters
    ----------
    y_encoded : cp.ndarray
        The labels, encoded as integers in [0, n_classes - 1].
    classes : np.ndarray or list[np.ndarray]
        The array of classes, or a list of arrays if multi-target.
    output_type : str, optional
        The type to output. May be any of the output types cuml supports.

    Returns
    -------
    labels
        The decoded labels, as output type ``output_type``.
    """
    if isinstance(classes, list):
        # Multi-target output
        dtype = (
            classes[0].dtype
            if len(set(c.dtype for c in classes)) == 1
            else None
        )
        if dtype is not None and dtype.kind in "iufb":
            # All dtypes are identical and numeric, we can use cupy here
            if all((c == np.arange(len(c))).all() for c in classes):
                # Fast path for common case of monotonically increasing numeric classes
                labels = y_encoded.astype(dtype, copy=False)
            else:
                # Need to transform y_encoded back to classes
                labels = cp.empty(shape=y_encoded.shape, dtype=dtype)
                for i, c in enumerate(classes):
                    labels[:, i] = cp.asarray(c).take(y_encoded[:, i])

            out = CumlArray(labels)
        else:
            # At least one class is non-numeric, we need to use cudf
            out = cudf.DataFrame(
                {
                    i: cudf.Series(c)
                    .take(y_encoded[:, i])
                    .reset_index(drop=True)
                    for i, c in enumerate(classes)
                }
            )
    else:
        # Single-target output
        dtype = classes.dtype
        if classes.dtype.kind in "iufb":
            # Numeric dtype, we can use cupy here
            if (classes == np.arange(len(classes))).all():
                # Fast path for common case of monotonically increasing numeric classes
                labels = y_encoded.astype(classes.dtype, copy=False)
            else:
                # Need to transform y_encoded back to classes
                labels = cp.asarray(classes).take(y_encoded)

            out = CumlArray(labels)
        else:
            # Non-numeric classes. We use cudf since it supports all types, and will
            # error appropriately later on when converting to outputs like `cupy`
            # that don't support strings.
            out = cudf.Series(classes).take(y_encoded).reset_index(drop=True)

    # Coerce result to requested output_type
    if isinstance(out, CumlArray):
        # Common numeric case, can just rely on CumlArray here
        return out.to_output(output_type)
    elif (
        output_type in ("cudf", "df_obj")
        or (output_type == "dataframe" and isinstance(out, cudf.DataFrame))
        or (output_type == "series" and isinstance(out, cudf.Series))
    ):
        return out
    elif output_type == "pandas":
        return cudf_to_pandas(out)
    elif output_type in ("numpy", "array"):
        return out.to_numpy(dtype=dtype)
    else:
        raise TypeError(
            f"{output_type=!r} doesn't support outputs of dtype "
            f"{dtype or 'object'} and shape {y_encoded.shape}"
        )


def process_class_weight(
    classes,
    y_ind,
    class_weight,
    sample_weight=None,
    float64=False,
    balanced_with_sample_weight=True,
):
    """Processes the `class_weight` argument to classifiers.

    Parameters
    ----------
    classes : array-like
        An array of classes for this classifier.
    y_ind : cp.ndarray
        An integral array of the transformed labels, where values (in [0,
        n_classes - 1]) are indices into `classes` mapping `y_ind` back to the
        original `y`.
    class_weight : dict, 'balanced', or None
        If `"balanced"`, classes are weighted by the inverse of their
        (weighted) counts. If a dict, keys are classes and values are
        corresponding weights. If `None`, the class weights will be uniform.
    sample_weight : array-like, optional
        An optional array of weights assigned to individual samples. May
        be unvalidated user-provided data.
    float64 : bool, optional
        Whether to use float64 for the weights, default False.
    balanced_with_sample_weight : bool, optional
        Whether to incorporate `sample_weight` when handling
        `class_weight='balanced'`. Statistically it makes sense to do this, but
        some sklearn and cuml estimators (e.g. `SVC`) weren't doing this and we
        may need to maintain this bug for a bit.

    Returns
    -------
    class_weight: np.ndarray, shape (n_classes,)
        Array of the applied weights, with `class_weight[i]` being the weight
        for the i-th class.
    sample_weight: CumlArray or None
        The resulting sample weights, or None if uniformly weighted.
    """
    dtype = np.float64 if float64 else np.float32
    n_samples = len(y_ind)
    n_classes = len(classes)

    sample_weight_cp = (
        None
        if sample_weight is None
        else input_to_cupy_array(
            sample_weight,
            check_cols=1,
            check_rows=n_samples,
            check_dtype=dtype,
            convert_to_dtype=dtype,
        ).array
    )

    if class_weight is None:
        # Uniform class weights
        weights = np.ones(n_classes, dtype=np.float64)
    elif class_weight == "balanced":
        counts = cp.bincount(
            y_ind,
            weights=(
                sample_weight_cp if balanced_with_sample_weight else None
            ),
        ).get()
        weights = (counts.sum() / (n_classes * counts)).astype(
            dtype, copy=False
        )
    else:
        weights = np.ones(n_classes, dtype=np.float64)
        unweighted = []
        for i, c in enumerate(cp.asnumpy(classes)):
            if c in class_weight:
                weights[i] = class_weight[c]
            else:
                unweighted.append(c)

        if unweighted and (n_classes - len(unweighted)) != len(class_weight):
            raise ValueError(
                f"The classes, {np.array(unweighted).tolist()}, are not in class_weight"
            )

    if (weights != 1).any():
        if sample_weight_cp is None:
            sample_weight_cp = cp.asarray(weights, dtype=dtype).take(y_ind)
        else:
            if cuda_ptr(sample_weight) == cuda_ptr(sample_weight_cp):
                # Need to make a copy
                sample_weight_cp = sample_weight_cp.copy()
            for ind, weight in enumerate(weights):
                sample_weight_cp[y_ind == ind] *= weight

    if sample_weight_cp is not None:
        sample_weight = CumlArray(data=sample_weight_cp)

    return weights, sample_weight
