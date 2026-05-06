#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import warnings

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
    *,
    sample_weights="deprecated",
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

    sample_weights : array-like, default="deprecated"
        Deprecated alias for ``sample_weight``.

        .. deprecated:: 26.06
            ``sample_weights`` was renamed to ``sample_weight`` and will be
            removed in 26.08.

    Returns
    -------
    loss : float
        The average hinge loss.
    """
    # Handle the deprecated `sample_weights` alias.
    if not (
        isinstance(sample_weights, str) and sample_weights == "deprecated"
    ):
        warnings.warn(
            "`sample_weights` was renamed to `sample_weight` in 26.06 and "
            "will be removed in 26.08.",
            FutureWarning,
            stacklevel=2,
        )
        if sample_weight is None:
            sample_weight = sample_weights

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
        # `check_y(return_classes=...)` expects a sorted, deduplicated numpy
        # array specifying the classes to use for label encoding. It will
        # raise a descriptive error if `y_true` contains labels not in this
        # array.
        return_classes = np.unique(cp.asnumpy(labels))
    else:
        # Derive classes from `y_true` itself.
        return_classes = True

    # Label-encode `y_true` to integer codes (column indices into `classes`).
    # `classes` comes back as a sorted numpy array.
    y_true, classes = check_y(
        y_true,
        return_classes=return_classes,
    )

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
