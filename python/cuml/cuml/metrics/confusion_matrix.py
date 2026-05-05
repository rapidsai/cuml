#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import cupyx
import numpy as np

from cuml.internals.array import CumlArray
from cuml.internals.validation import (
    check_array,
    check_consistent_length,
    check_sample_weight,
)
from cuml.metrics.utils import sorted_unique_labels
from cuml.prims.label import make_monotonic

_LABEL_DTYPES = (np.int32, np.int64)
_WEIGHT_DTYPES = (np.float32, np.float64, np.int32, np.int64)


def confusion_matrix(
    y_true,
    y_pred,
    labels=None,
    sample_weight=None,
    normalize=None,
    convert_dtype=False,
) -> CumlArray:
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Parameters
    ----------
    y_true : array-like (device or host) shape = (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like (device or host) shape = (n_samples,)
        Estimated target values.
    labels : array-like (device or host) shape = (n_classes,), optional
        List of labels to index the matrix. This may be used to reorder or
        select a subset of labels. If None is given, those that appear at least
        once in y_true or y_pred are used in sorted order.
    sample_weight : array-like (device or host) shape = (n_samples,), optional
        Sample weights.
    normalize : string in ['true', 'pred', 'all'] or None (default=None)
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be
        normalized.
    convert_dtype : bool, optional (default=False)
        When set to True, the confusion matrix method will automatically
        convert the predictions, ground truth, and labels arrays to np.int32.

    Returns
    -------
    C : array-like (device or host) shape = (n_classes, n_classes)
        Confusion matrix.
    """
    y_true = check_array(
        y_true,
        ensure_2d=False,
        dtype=_LABEL_DTYPES,
        convert_dtype=convert_dtype,
        input_name="y_true",
    )
    y_pred = check_array(
        y_pred,
        ensure_2d=False,
        dtype=_LABEL_DTYPES,
        convert_dtype=convert_dtype,
        input_name="y_pred",
    )
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(
            f"y_true and y_pred must be 1D arrays, got shapes "
            f"{y_true.shape} and {y_pred.shape}"
        )
    check_consistent_length(y_true, y_pred)
    n_rows = y_true.shape[0]

    if labels is None:
        labels = sorted_unique_labels(y_true, y_pred)
    else:
        labels = check_array(
            labels,
            ensure_2d=False,
            dtype=_LABEL_DTYPES,
            convert_dtype=convert_dtype,
            input_name="labels",
        )
        if labels.ndim != 1:
            raise ValueError(
                f"labels must be a 1D array, got shape {labels.shape}"
            )
    n_labels = labels.shape[0]

    if (
        sample_weight := check_sample_weight(
            sample_weight, dtype=_WEIGHT_DTYPES
        )
    ) is not None:
        check_consistent_length(y_true, sample_weight)
    else:
        sample_weight = cp.ones(n_rows, dtype=y_true.dtype)

    if normalize not in ("true", "pred", "all", None):
        raise ValueError(
            "normalize must be one of "
            f"{{'true', 'pred', 'all', None}}, got {normalize}."
        )

    y_true, _ = make_monotonic(y_true, labels, copy=True)
    y_pred, _ = make_monotonic(y_pred, labels, copy=True)

    # intersect y_pred, y_true with labels, eliminate items not in labels
    ind = cp.logical_and(y_pred < n_labels, y_true < n_labels)
    y_pred = y_pred[ind]
    y_true = y_true[ind]
    sample_weight = sample_weight[ind]

    cm = cupyx.scipy.sparse.coo_matrix(
        (sample_weight, (y_true, y_pred)),
        shape=(n_labels, n_labels),
        dtype=np.float64,
    ).toarray()

    # Choose the accumulator dtype to always have high precision
    if sample_weight.dtype.kind in {"i", "u", "b"}:
        cm = cm.astype(np.int64)

    with np.errstate(all="ignore"):
        if normalize == "true":
            cm = cp.divide(cm, cm.sum(axis=1, keepdims=True))
        elif normalize == "pred":
            cm = cp.divide(cm, cm.sum(axis=0, keepdims=True))
        elif normalize == "all":
            cm = cp.divide(cm, cm.sum())
        cm = cp.nan_to_num(cm)

    return cm
