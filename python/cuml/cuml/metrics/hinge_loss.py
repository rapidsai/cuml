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
    sample_weight = check_sample_weight(sample_weight, dtype=np.float64)
    check_consistent_length(y_true, pred_decision, sample_weight)

    # The set of unique labels (sorted) determines whether we're in the binary
    # or multiclass regime, and provides the column ordering for
    # `pred_decision` in the multiclass case.
    classes = cp.sort(cp.unique(y_true if labels is None else labels))

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

        # Encode `y_true` as column indices into `classes`. `searchsorted`
        # produces the right index when the value exists, and an
        # out-of-range / wrong index otherwise -- which we catch below.
        y_true_idx = cp.searchsorted(classes, y_true)
        if not bool(
            (y_true_idx < classes.size).all()
            and (
                classes[cp.minimum(y_true_idx, classes.size - 1)] == y_true
            ).all()
        ):
            raise ValueError(
                "y_true contains labels that are not present in `labels`."
            )

        n_samples = y_true.shape[0]
        mask = cp.ones_like(pred_decision, dtype=bool)
        mask[cp.arange(n_samples), y_true_idx] = False
        margin = pred_decision[~mask]
        margin -= cp.max(pred_decision[mask].reshape(n_samples, -1), axis=1)
    else:
        # Binary case. Map the smaller class to -1 and the larger class to +1,
        # matching the convention used by sklearn's LabelBinarizer.
        if pred_decision.ndim > 1:
            pred_decision = cp.ravel(pred_decision)
        pos_label = classes[-1]
        y_signed = cp.where(y_true == pos_label, 1, -1).astype(
            pred_decision.dtype, copy=False
        )
        margin = y_signed * pred_decision

    losses = 1 - margin
    # The hinge_loss doesn't penalize good enough predictions.
    cp.clip(losses, 0, None, out=losses)
    return float(cp.average(losses, weights=sample_weight))
