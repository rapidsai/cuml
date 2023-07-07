#
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

from cuml.prims.label import make_monotonic
from cuml.metrics.utils import sorted_unique_labels
from cuml.internals.input_utils import input_to_cupy_array
from cuml.internals.array import CumlArray
from cuml.common import using_output_type
from cuml.common import input_to_cuml_array
import cuml.internals
from cuml.internals.safe_imports import gpu_only_import
from cuml.internals.safe_imports import cpu_only_import

np = cpu_only_import("numpy")
cp = gpu_only_import("cupy")
cupyx = gpu_only_import("cupyx")


@cuml.internals.api_return_any()
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
        or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like (device or host) shape = (n_samples,)
        or (n_samples, n_outputs)
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
    y_true, n_rows, n_cols, dtype = input_to_cuml_array(
        y_true,
        check_dtype=[cp.int32, cp.int64],
        convert_to_dtype=(cp.int32 if convert_dtype else None),
    )

    y_pred, _, _, _ = input_to_cuml_array(
        y_pred,
        check_dtype=[cp.int32, cp.int64],
        check_rows=n_rows,
        check_cols=n_cols,
        convert_to_dtype=(cp.int32 if convert_dtype else None),
    )

    if labels is None:
        labels = sorted_unique_labels(y_true, y_pred)
        n_labels = len(labels)
    else:
        labels, n_labels, _, _ = input_to_cupy_array(
            labels,
            check_dtype=[cp.int32, cp.int64],
            convert_to_dtype=(cp.int32 if convert_dtype else None),
            check_cols=1,
        )
    if sample_weight is None:
        sample_weight = cp.ones(n_rows, dtype=dtype)
    else:
        sample_weight, _, _, _ = input_to_cupy_array(
            sample_weight,
            check_dtype=[cp.float32, cp.float64, cp.int32, cp.int64],
            check_rows=n_rows,
            check_cols=n_cols,
        )

    if normalize not in ["true", "pred", "all", None]:
        msg = (
            "normalize must be one of "
            f"{{'true', 'pred', 'all', None}}, got {normalize}."
        )
        raise ValueError(msg)

    with using_output_type("cupy"):
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
