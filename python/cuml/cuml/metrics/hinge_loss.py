#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import cudf
import cupy as cp
import numpy as np

from cuml.internals.validation import (
    check_array,
    check_consistent_length,
    check_sample_weight,
)
from cuml.preprocessing import LabelBinarizer, LabelEncoder


def hinge_loss(
    y_true, pred_decision, labels=None, sample_weights=None
) -> float:
    """
    Calculates non-regularized hinge loss. Adapted from scikit-learn hinge loss.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels, consisting of labels for the classes. In binary
        classification, the positive label must be greater than the negative
        label.

    pred_decision : array-like of shape (n_samples,) or (n_samples, n_classes)
        Predicted decisions, as output by ``decision_function`` (floats).

    labels : array-like, default=None
        In multiclass problems, this must include all class labels.

    sample_weights : array-like of shape (n_samples,), default=None
        Sample weights to be used for computing the average.

    Returns
    -------
    loss : float
        The average hinge loss.
    """
    y_true = check_array(
        y_true,
        ensure_2d=False,
        ensure_all_finite=False,
        input_name="y_true",
    )
    pred_decision = check_array(
        pred_decision,
        ensure_2d=False,
        dtype=(np.float32, np.float64),
        input_name="pred_decision",
    )
    if labels is not None:
        labels = check_array(
            labels,
            ensure_2d=False,
            ensure_all_finite=False,
            input_name="labels",
        )
    sample_weights = check_sample_weight(sample_weights, dtype=np.float64)
    check_consistent_length(y_true, pred_decision, sample_weights)

    y_true_unique = cp.unique(labels if labels is not None else y_true)

    if y_true_unique.size > 2:
        if (
            labels is None
            and pred_decision.ndim > 1
            and y_true_unique.size != pred_decision.shape[1]
        ):
            raise ValueError(
                "Please include all labels in y_true "
                "or pass labels as third argument"
            )
        if labels is None:
            labels = y_true_unique
        le = LabelEncoder(output_type="cudf")
        le.fit(cudf.Series(labels))
        y_true = le.transform(cudf.Series(y_true))

        n_samples = y_true.shape[0]
        mask = cp.ones_like(pred_decision, dtype=bool)
        mask[cp.arange(n_samples), y_true.values] = False
        margin = pred_decision[~mask]
        margin -= cp.max(pred_decision[mask].reshape(n_samples, -1), axis=1)
    else:
        # Handles binary class case
        # this code assumes that positive and negative labels
        # are encoded as +1 and -1 respectively
        if pred_decision.ndim > 1:
            pred_decision = cp.ravel(pred_decision)

        lbin = LabelBinarizer(neg_label=-1, output_type="cupy")
        y_true = lbin.fit_transform(cudf.Series(y_true))[:, 1]
        margin = y_true * pred_decision

    losses = 1 - margin
    # The hinge_loss doesn't penalize good enough predictions.
    cp.clip(losses, 0, None, out=losses)
    return float(cp.average(losses, weights=sample_weights))
