#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp

from cuml.internals.input_utils import input_to_cupy_array


def make_monotonic(labels, classes=None, copy=False):
    """
    Takes a set of labels that might not be drawn from the
    set [0, n-1] and renumbers them to be drawn from that
    interval.

    Labels not present in classes are mapped to len(classes).

    Parameters
    ----------
    labels : array-like of shape (n_samples,)
        Labels to convert to monotonic indices.
    classes : array-like of shape (n_classes,), optional
        The unique set of classes. If None, classes are derived
        from the unique values in labels.
    copy : bool, default=False
        If True, a copy of labels is returned. If False, the
        operation is performed in-place on device arrays.

    Returns
    -------
    mapped_labels : cupy.ndarray of shape (n_samples,)
        Labels mapped to monotonic indices [0, n_classes-1].
        Labels not in classes are mapped to n_classes.
    classes : cupy.ndarray of shape (n_classes,)
        Sorted unique class labels.
    """
    labels = input_to_cupy_array(labels, deepcopy=copy).array

    if labels.ndim != 1:
        raise ValueError("Labels array must be 1D")

    if classes is None:
        # Derive classes from labels and get inverse mapping directly
        classes, mapped_labels = cp.unique(labels, return_inverse=True)
        if not copy:
            labels[:] = mapped_labels
            mapped_labels = labels
    else:
        # Convert and sort provided classes
        classes = input_to_cupy_array(classes).array
        classes = cp.sort(classes)

        # Map each label to its index in sorted classes using binary search
        indices = cp.searchsorted(classes, labels)

        # Validate: check if the found position actually matches the label
        # Out-of-bounds indices are clamped to avoid index errors
        indices_safe = cp.minimum(indices, len(classes) - 1)
        valid = (indices < len(classes)) & (classes[indices_safe] == labels)

        # Map valid labels to their indices, invalid labels to len(classes)
        mapped_labels = cp.where(valid, indices, len(classes))

        if not copy:
            labels[:] = mapped_labels
            mapped_labels = labels

    return mapped_labels, classes
