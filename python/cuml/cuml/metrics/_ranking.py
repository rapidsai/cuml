#
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

import math
import typing

import cupy as cp
import numpy as np

import cuml.internals
from cuml.internals.array import CumlArray
from cuml.internals.input_utils import input_to_cupy_array


@cuml.internals.api_return_generic(get_output_type=True)
def precision_recall_curve(
    y_true, probs_pred
) -> typing.Tuple[CumlArray, CumlArray, CumlArray]:
    """
    Compute precision-recall pairs for different probability thresholds

    .. note:: this implementation is restricted to the binary classification
        task. The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the
        number of true positives and ``fp`` the number of false positives. The
        precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative.

        The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number
        of true positives and ``fn`` the number of false negatives. The recall
        is intuitively the ability of the classifier to find all the positive
        samples. The last precision and recall values are 1. and 0.
        respectively and do not have a corresponding threshold. This ensures
        that the graph starts on the y axis.

        Read more in the scikit-learn's `User Guide
        <https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics>`_.


    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True binary labels, {0, 1}.
    probas_pred : array, shape = [n_samples]
        Estimated probabilities or decision function.

    Returns
    -------
    precision : array, shape = [n_thresholds + 1]
        Precision values such that element i is the precision of
        predictions with score >= thresholds[i] and the last element is 1.
    recall : array, shape = [n_thresholds + 1]
        Decreasing recall values such that element i is the recall of
        predictions with score >= thresholds[i] and the last element is 0.
    thresholds : array, shape = [n_thresholds <= len(np.unique(probas_pred))]
        Increasing thresholds on the decision function used to compute
        precision and recall.

    Examples
    --------

    .. code-block:: python

        >>> import cupy as cp
        >>> from cuml.metrics import precision_recall_curve
        >>> y_true = cp.array([0, 0, 1, 1])
        >>> y_scores = cp.array([0.1, 0.4, 0.35, 0.8])
        >>> precision, recall, thresholds = precision_recall_curve(
        ...     y_true, y_scores)
        >>> print(precision)
        [0.666... 0.5  1.  1. ]
        >>> print(recall)
        [1. 0.5 0.5 0. ]
        >>> print(thresholds)
        [0.35 0.4 0.8 ]

    """
    y_true, n_rows, n_cols, ytype = input_to_cupy_array(
        y_true, check_dtype=[np.int32, np.int64, np.float32, np.float64]
    )

    y_score, _, _, _ = input_to_cupy_array(
        probs_pred,
        check_dtype=[np.int32, np.int64, np.float32, np.float64],
        check_rows=n_rows,
        check_cols=n_cols,
    )

    if cp.any(y_true) == 0:
        raise ValueError(
            "precision_recall_curve cannot be used when " "y_true is all zero."
        )

    fps, tps, thresholds = _binary_clf_curve(y_true, y_score)
    precision = cp.flip(tps / (tps + fps), axis=0)
    recall = cp.flip(tps / tps[-1], axis=0)
    n = (recall == 1).sum()

    if n > 1:
        precision = precision[n - 1 :]
        recall = recall[n - 1 :]
        thresholds = thresholds[n - 1 :]
    precision = cp.concatenate([precision, cp.ones(1)])
    recall = cp.concatenate([recall, cp.zeros(1)])

    return precision, recall, thresholds


@cuml.internals.api_return_any()
def roc_auc_score(y_true, y_score):
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    from prediction scores.

    .. note:: this implementation can only be used with binary classification.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels. The binary cases
        expect labels with shape (n_samples,)
    y_score : array-like of shape (n_samples,)
        Target scores. In the binary cases, these can be either
        probability estimates or non-thresholded decision values (as returned
        by `decision_function` on some classifiers). The binary
        case expects a shape (n_samples,), and the scores must be the scores of
        the class with the greater label.

    Returns
    -------
        auc : float

    Examples
    --------
    >>> import numpy as np
    >>> from cuml.metrics import roc_auc_score
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> print(roc_auc_score(y_true, y_scores))
    0.75

    """
    y_true, n_rows, n_cols, ytype = input_to_cupy_array(
        y_true, check_dtype=[np.int32, np.int64, np.float32, np.float64]
    )

    y_score, _, _, _ = input_to_cupy_array(
        y_score,
        check_dtype=[np.int32, np.int64, np.float32, np.float64],
        check_rows=n_rows,
        check_cols=n_cols,
    )
    return _binary_roc_auc_score(y_true, y_score)


def _binary_clf_curve(y_true, y_score):

    if y_true.dtype.kind == "f" and np.any(y_true != y_true.astype(int)):
        raise ValueError("Continuous format of y_true  " "is not supported.")

    ids = cp.argsort(-y_score)
    sorted_score = y_score[ids]

    ones = y_true[ids].astype("float32")  # for calculating true positives
    zeros = 1 - ones  # for calculating predicted positives

    # calculate groups
    group = _group_same_scores(sorted_score)
    num = int(group[-1])

    tps = cp.zeros(num, dtype="float32")
    fps = cp.zeros(num, dtype="float32")

    tps = _addup_x_in_group(group, ones, tps)
    fps = _addup_x_in_group(group, zeros, fps)

    tps = cp.cumsum(tps)
    fps = cp.cumsum(fps)
    thresholds = cp.unique(y_score)
    return fps, tps, thresholds


def _binary_roc_auc_score(y_true, y_score):
    """Compute binary roc_auc_score using cupy"""

    if cp.unique(y_true).shape[0] == 1:
        raise ValueError(
            "roc_auc_score cannot be used when "
            "only one class present in y_true. ROC AUC score "
            "is not defined in that case."
        )

    if cp.unique(y_score).shape[0] == 1:
        return 0.5

    fps, tps, thresholds = _binary_clf_curve(y_true, y_score)
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]

    return _calculate_area_under_curve(fpr, tpr).item()


def _addup_x_in_group(group, x, result):
    addup_x_in_group_kernel = cp.RawKernel(
        r"""
        extern "C" __global__
        void addup_x_in_group(const int* group, const float* x,
            float* result, int N)
        {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            if(tid<N){
                atomicAdd(result + group[tid] - 1, x[tid]);
            }
        }
    """,
        "addup_x_in_group",
    )

    N = x.shape[0]
    tpb = 256
    bpg = math.ceil(N / tpb)
    addup_x_in_group_kernel((bpg,), (tpb,), (group, x, result, N))
    return result


def _group_same_scores(sorted_score):
    mask = cp.empty(sorted_score.shape, dtype=cp.bool_)
    mask[0] = True
    mask[1:] = sorted_score[1:] != sorted_score[:-1]
    group = cp.cumsum(mask, dtype=cp.int32)
    return group


def _calculate_area_under_curve(fpr, tpr):
    """helper function to calculate area under curve given fpr & tpr arrays"""
    return (
        cp.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1])) / 2
        + tpr[0] * fpr[0] / 2
    )
