#
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp

from cuml.common import input_to_cuml_array
from cuml.metrics.utils import sorted_unique_labels
from cuml.prims.label import make_monotonic


def prepare_cluster_metric_inputs(labels_true, labels_pred):
    """Helper function to avoid code duplication for homogeneity score, mutual
    info score and completeness score.
    """
    y_true, n_rows, _, dtype = input_to_cuml_array(
        labels_true,
        check_dtype=[cp.int32, cp.int64],
        check_cols=1,
        deepcopy=True,  # deepcopy because we call make_monotonic inplace below
    )

    y_pred, _, _, _ = input_to_cuml_array(
        labels_pred,
        check_dtype=dtype,
        check_rows=n_rows,
        check_cols=1,
        deepcopy=True,  # deepcopy because we call make_monotonic inplace below
    )

    classes = sorted_unique_labels(y_true, y_pred)

    make_monotonic(y_true, classes=classes, copy=False)
    make_monotonic(y_pred, classes=classes, copy=False)

    # Those values are only correct because we used make_monotonic
    lower_class_range = 0
    upper_class_range = len(classes) - 1

    return y_true, y_pred, n_rows, lower_class_range, upper_class_range
