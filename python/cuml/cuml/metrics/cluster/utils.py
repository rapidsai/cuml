#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np

from cuml.internals.validation import check_array, check_consistent_length
from cuml.metrics.utils import sorted_unique_labels
from cuml.prims.label import make_monotonic


def prepare_cluster_metric_inputs(labels_true, labels_pred):
    """Helper function to avoid code duplication for homogeneity score, mutual
    info score and completeness score.

    Returns ``(y_true, y_pred, n_rows, lower_class_range, upper_class_range)``
    where ``y_true`` and ``y_pred`` are C-contiguous int32 ``cupy.ndarray``
    arrays whose label values have been remapped to the contiguous range
    ``[0, len(classes) - 1]``.
    """
    y_true = check_array(
        labels_true,
        ensure_2d=False,
        ensure_min_samples=0,
        order="C",
        dtype=np.int32,
        input_name="labels_true",
    )
    y_pred = check_array(
        labels_pred,
        ensure_2d=False,
        ensure_min_samples=0,
        order="C",
        dtype=np.int32,
        input_name="labels_pred",
    )
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(
            "labels_true and labels_pred must be 1D arrays, got shapes "
            f"{y_true.shape} and {y_pred.shape}"
        )
    check_consistent_length(y_true, y_pred)
    n_rows = y_true.shape[0]

    classes = sorted_unique_labels(y_true, y_pred)

    # Make copies so that we never mutate the caller's input arrays.
    # ``make_monotonic`` with ``copy=True`` returns new cupy arrays; we
    # use those for the downstream Cython callers.
    y_true, _ = make_monotonic(y_true, classes=classes, copy=True)
    y_pred, _ = make_monotonic(y_pred, classes=classes, copy=True)

    # Those values are only correct because we used make_monotonic
    lower_class_range = 0
    upper_class_range = len(classes) - 1

    return y_true, y_pred, n_rows, lower_class_range, upper_class_range
