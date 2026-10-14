# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Mapping

import cupy as cp
import numpy as np


def process_class_weight(
    classes,
    y_ind,
    class_weight,
    sample_weight=None,
    dtype=None,
    balanced_with_sample_weight=True,
):
    """Processes the `class_weight` argument to classifiers.

    Parameters
    ----------
    classes : array-like
        An array of classes for this classifier.
    y_ind : cp.ndarray
        An integral array of the transformed labels, where values (in [0,
        n_classes - 1]) are indices into `classes` mapping `y_ind` back to the
        original `y`.
    class_weight : dict, 'balanced', or None
        If `"balanced"`, classes are weighted by the inverse of their
        (weighted) counts. If a dict, keys are classes and values are
        corresponding weights. If `None`, the class weights will be uniform.
    sample_weight : cp.ndarray, optional
        An optional array of weights assigned to individual samples.
    dtype : dtype-like, optional
        The dtype to use for the output weights. Defaults to the dtype of
        `sample_weight` (if provided), or float32 otherwise.
    balanced_with_sample_weight : bool, optional
        Whether to incorporate `sample_weight` when handling
        `class_weight='balanced'`. Statistically it makes sense to do this, but
        some sklearn and cuml estimators (e.g. `SVC`) weren't doing this and we
        may need to maintain this bug for a bit.

    Returns
    -------
    class_weight: np.ndarray, shape (n_classes,)
        Array of the applied weights, with `class_weight[i]` being the weight
        for the i-th class.
    sample_weight: cp.ndarray or None
        The resulting sample weights, or None if uniformly weighted.
    """
    if not (
        class_weight is None
        or isinstance(class_weight, Mapping)
        or (isinstance(class_weight, str) and class_weight == "balanced")
    ):
        raise ValueError(
            "class_weight must be a dict, 'balanced', or None; "
            f"got {class_weight!r}"
        )

    n_classes = len(classes)
    if dtype is None:
        dtype = getattr(sample_weight, "dtype", np.float32)
    else:
        dtype = np.dtype(dtype)

    if sample_weight is not None:
        sample_weight = sample_weight.astype(dtype, copy=False)

    if class_weight is None:
        # Uniform class weights
        weights = np.ones(n_classes, dtype=np.float64)
    elif class_weight == "balanced":
        counts = cp.bincount(
            y_ind,
            weights=(sample_weight if balanced_with_sample_weight else None),
        ).get()
        weights = (counts.sum() / (n_classes * counts)).astype(
            dtype, copy=False
        )
    else:
        weights = np.ones(n_classes, dtype=np.float64)
        unweighted = []
        for i, c in enumerate(cp.asnumpy(classes)):
            if c in class_weight:
                weights[i] = class_weight[c]
            else:
                unweighted.append(c)

        if unweighted and (n_classes - len(unweighted)) != len(class_weight):
            raise ValueError(
                f"The classes, {np.array(unweighted).tolist()}, are not in class_weight"
            )

    if (weights != 1).any():
        if sample_weight is None:
            sample_weight = cp.asarray(weights, dtype=dtype).take(y_ind)
        else:
            sample_weight = sample_weight.copy()
            for ind, weight in enumerate(weights):
                sample_weight[y_ind == ind] *= weight

    return weights, sample_weight
