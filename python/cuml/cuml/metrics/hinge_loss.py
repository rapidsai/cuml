#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import numpy as np

from cuml.internals.validation import (
    check_array,
    check_consistent_length,
    check_sample_weight,
    check_y,
)


def hinge_loss(
    y_true,
    pred_decision,
    labels=None,
    sample_weight=None,
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

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights to be used for computing the average.

    Returns
    -------
    loss : float
        The average hinge loss.
    """
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
        classes = np.unique(cp.asnumpy(labels))
    else:
        classes = None

    if classes is None:
        y_true, classes = check_y(y_true, return_classes=True)
    elif classes.size > 2:
        # For multiclass hinge loss, supplied labels define the column order
        # for pred_decision and should be used to encode y_true.
        y_true, classes = check_y(y_true, return_classes=classes)
    else:
        # For sklearn-compatible binary hinge loss, supplied labels select the
        # binary branch, but the sign transform is fit from observed y_true.
        y_true, _ = check_y(y_true, return_classes=True)

    sample_weight = check_sample_weight(sample_weight, dtype=np.float64)
    check_consistent_length(y_true, pred_decision, sample_weight)

    if classes.size > 2:
        # Multiclass case
        if (
            labels is None
            and pred_decision.ndim > 1
            and classes.size != pred_decision.shape[1]
        ):
            raise ValueError(
                "Please include all labels in y_true "
                "or pass labels as third argument"
            )
        if pred_decision.ndim != 2:
            raise ValueError(
                "pred_decision must be 2D for multiclass hinge loss, "
                f"got a {pred_decision.ndim}D array instead."
            )

        # `y_true` is already encoded as column indices into `classes`.
        n_samples = y_true.shape[0]
        mask = cp.ones_like(pred_decision, dtype=bool)
        mask[cp.arange(n_samples), y_true] = False
        margin = pred_decision[~mask]
        margin -= cp.max(pred_decision[mask].reshape(n_samples, -1), axis=1)
    else:
        # Binary case. Codes are 0/1 with `classes` sorted, so code 1
        # corresponds to the larger class (positive label), matching the
        # convention used by sklearn's LabelBinarizer.
        if pred_decision.ndim > 1:
            pred_decision = cp.ravel(pred_decision)
        y_signed = cp.where(y_true == 1, 1, -1).astype(
            pred_decision.dtype, copy=False
        )
        margin = y_signed * pred_decision

    losses = 1 - margin
    # The hinge_loss doesn't penalize good enough predictions.
    cp.clip(losses, 0, None, out=losses)
    return float(cp.average(losses, weights=sample_weight))
