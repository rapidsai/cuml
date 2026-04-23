# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import cudf
import cupy as cp
import numpy as np

from cuml.internals.array import CumlArray
from cuml.internals.output_utils import cudf_to_pandas


def decode_labels(y_encoded, classes, output_type="cupy"):
    """Convert encoded labels back into their original classes.

    Parameters
    ----------
    y_encoded : cp.ndarray
        The labels, encoded as integers in [0, n_classes - 1].
    classes : np.ndarray or list[np.ndarray]
        The array of classes, or a list of arrays if multi-target.
    output_type : str, optional
        The type to output. May be any of the output types cuml supports.

    Returns
    -------
    labels
        The decoded labels, as output type ``output_type``.
    """
    if isinstance(classes, list):
        # Multi-target output
        dtype = (
            classes[0].dtype
            if len(set(c.dtype for c in classes)) == 1
            else None
        )
        if dtype is not None and dtype.kind in "iufb":
            # All dtypes are identical and numeric, we can use cupy here
            if all((c == np.arange(len(c))).all() for c in classes):
                # Fast path for common case of monotonically increasing numeric classes
                labels = y_encoded.astype(dtype, copy=False)
            else:
                # Need to transform y_encoded back to classes
                labels = cp.empty(shape=y_encoded.shape, dtype=dtype)
                for i, c in enumerate(classes):
                    labels[:, i] = cp.asarray(c).take(y_encoded[:, i])

            out = CumlArray(labels)
        else:
            # At least one class is non-numeric, we need to use cudf
            out = cudf.DataFrame(
                {
                    i: cudf.Series(c)
                    .take(y_encoded[:, i])
                    .reset_index(drop=True)
                    for i, c in enumerate(classes)
                }
            )
    else:
        # Single-target output
        dtype = classes.dtype
        if classes.dtype.kind in "iufb":
            # Numeric dtype, we can use cupy here
            if (classes == np.arange(len(classes))).all():
                # Fast path for common case of monotonically increasing numeric classes
                labels = y_encoded.astype(classes.dtype, copy=False)
            else:
                # Need to transform y_encoded back to classes
                labels = cp.asarray(classes).take(y_encoded)

            out = CumlArray(labels)
        else:
            # Non-numeric classes. We use cudf since it supports all types, and will
            # error appropriately later on when converting to outputs like `cupy`
            # that don't support strings.
            out = cudf.Series(classes).take(y_encoded).reset_index(drop=True)

    # Coerce result to requested output_type
    if isinstance(out, CumlArray):
        # Common numeric case, can just rely on CumlArray here
        return out.to_output(output_type)
    elif (
        output_type in ("cudf", "df_obj")
        or (output_type == "dataframe" and isinstance(out, cudf.DataFrame))
        or (output_type == "series" and isinstance(out, cudf.Series))
    ):
        return out
    elif output_type == "pandas":
        return cudf_to_pandas(out)
    elif output_type in ("numpy", "array"):
        return out.to_numpy(dtype=dtype)
    else:
        raise TypeError(
            f"{output_type=!r} doesn't support outputs of dtype "
            f"{dtype or 'object'} and shape {y_encoded.shape}"
        )


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
