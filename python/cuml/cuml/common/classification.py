# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import cupy as cp
import numpy as np

from cuml.internals.array import CumlArray
from cuml.internals.input_utils import input_to_cupy_array
from cuml.internals.memory_utils import cuda_ptr


def process_class_weight(
    classes,
    y_ind,
    class_weight,
    sample_weight=None,
    float64=False,
    balanced_with_sample_weight=True,
):
    """Processes the `class_weight` argument to classifiers.

    Parameters
    ----------
    classes : array-like
        An array of classes for this classifier.
    y_ind : cp.ndarray
        An integral array of the transformed labels, where values (in [0,
        n_classes - 1]) Are indices into `classes` mapping `y_ind` back to the
        original `y`.
    class_weight : dict, 'balanced', or None
        If `"balanced"`, classes are weighted by the inverse of their
        (weighted) counts. If a dict, keys are classes and values are
        corresponding weights. If `None`, the class weights will be uniform.
    sample_weight : array-like, optional
        An optional array of weights assigned to individual samples. May
        be unvalidated user-provided data.
    float64 : bool, optional
        Whether to use float64 for the weights, default False.
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
    sample_weight: CumlArray or None
        The resulting sample weights, or None if uniformly weighted.
    """
    dtype = np.float64 if float64 else np.float32
    n_samples = len(y_ind)
    n_classes = len(classes)

    sample_weight_cp = (
        None
        if sample_weight is None
        else input_to_cupy_array(
            sample_weight,
            check_cols=1,
            check_rows=n_samples,
            check_dtype=dtype,
            convert_to_dtype=dtype,
        ).array
    )

    if class_weight is None:
        # Uniform class weights
        weights = np.ones(n_classes, dtype=np.float64)
    elif class_weight == "balanced":
        counts = cp.bincount(
            y_ind,
            weights=(
                sample_weight_cp if balanced_with_sample_weight else None
            ),
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
        if sample_weight_cp is None:
            sample_weight_cp = cp.asarray(weights, dtype=dtype).take(y_ind)
        else:
            if cuda_ptr(sample_weight) == cuda_ptr(sample_weight_cp):
                # Need to make a copy
                sample_weight_cp = sample_weight_cp.copy()
            for ind, weight in enumerate(weights):
                sample_weight_cp[y_ind == ind] *= weight

    if sample_weight_cp is not None:
        sample_weight = CumlArray(data=sample_weight_cp)

    return weights, sample_weight
