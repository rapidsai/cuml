#
# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

import cudf
import cupy as cp

import cuml.internals
from cuml.internals.input_utils import determine_array_type
from cuml.preprocessing import LabelBinarizer, LabelEncoder


@cuml.internals.api_return_any()
def hinge_loss(
    y_true, pred_decision, labels=None, sample_weights=None
) -> float:
    """
    Calculates non-regularized hinge loss. Adapted from scikit-learn hinge loss

    Parameters
    ----------
    y_true: cuDF Series or cuPy array of shape (n_samples,)
        True labels, consisting of labels for the classes.
        In binary classification, the positive label must be
        greater than negative class

    pred_decision: cuDF DataFrame or cuPy array of shape (n_samples,) or \
            (n_samples, n_classes)
        Predicted decisions, as output by decision_function (floats)

    labels: cuDF Series or cuPy array, default=None
        In multiclass problems, this must include all class labels.

    sample_weight: cupy array of shape (n_samples,), default=None
        Sample weights to be used for computing the average

    Returns
    -------
    loss : float
           The average hinge loss.
    """

    yt_type = determine_array_type(y_true)
    pd_type = determine_array_type(pred_decision)
    labels_type = determine_array_type(labels)

    if yt_type not in ["cupy", "numba", "cudf"]:
        raise TypeError(
            "y_true needs to be either a cuDF Series or \
                        a cuPy/numba array."
        )

    if pd_type not in ["cupy", "numba", "cudf"]:
        raise TypeError(
            "pred_decision needs to be either a cuDF DataFrame or \
                        a cuPy/numba array."
        )

    if labels_type not in ["cupy", "numba", "cudf"]:
        raise TypeError(
            "labels needs to be either a cuDF Series or \
                        a cuPy/numba array."
        )

    if y_true.shape[0] != pred_decision.shape[0]:
        raise ValueError(
            "y_true and pred_decision must have the same"
            " number of rows(found {} and {})".format(
                y_true.shape[0], pred_decision.shape[0]
            )
        )

    if sample_weights and sample_weights.shape[0] != y_true.shape[0]:
        raise ValueError(
            "y_true and sample_weights must have the same "
            "number of rows (found {} and {})".format(
                y_true.shape[0], sample_weights.shape[0]
            )
        )

    if not isinstance(labels, cudf.Series):
        labels = cudf.Series(labels)

    if not isinstance(y_true, cudf.Series):
        y_true = cudf.Series(y_true)

    y_true_unique = cp.unique(labels if labels is not None else y_true)

    if y_true_unique.size > 2:
        if (
            labels is None
            and pred_decision.ndim > 1
            and (cp.size(y_true_unique) != pred_decision.shape[1])
        ):
            raise ValueError(
                "Please include all labels in y_true "
                "or pass labels as third argument"
            )
        if labels is None:
            labels = y_true_unique
        le = LabelEncoder(output_type="cudf")
        le.fit(labels)
        y_true = le.transform(y_true)
        if isinstance(pred_decision, cudf.DataFrame):
            pred_decision = pred_decision.values

        mask = cp.ones_like(pred_decision, dtype=bool)
        mask[cp.arange(y_true.shape[0]), y_true.values] = False
        margin = pred_decision[~mask]
        margin -= cp.max(
            pred_decision[mask].reshape(y_true.shape[0], -1), axis=1
        )
    else:
        # Handles binary class case
        # this code assumes that positive and negative labels
        # are encoded as +1 and -1 respectively
        if isinstance(pred_decision, cudf.DataFrame):
            pred_decision = pred_decision.values
        pred_decision = cp.ravel(pred_decision)

        lbin = LabelBinarizer(neg_label=-1, output_type="cupy")
        y_true = lbin.fit_transform(y_true)[:, 1]

        try:
            margin = y_true * pred_decision
        except TypeError:
            raise TypeError("pred_decision should be an array of floats.")

    losses = 1 - margin
    # The hinge_loss doesn't penalize good enough predictions.
    cp.clip(losses, 0, None, out=losses)
    return cp.average(losses, weights=sample_weights)
