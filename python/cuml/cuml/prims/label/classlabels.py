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
        Labels mapped to indices [0, n_classes-1].
        Labels not in classes are mapped to n_classes.
    classes : cupy.ndarray of shape (n_classes,)
        Unique class labels. These will be sorted if classes is None.
        If classes is provided, the original order is preserved.
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
        # Convert provided classes
        classes = input_to_cupy_array(classes).array

        # Sort provided classes for binary search, but keep track of
        # original indices to maintain provided order.
        sort_indices = cp.argsort(classes)
        sorted_classes = classes[sort_indices]

        # Map each label to its index in sorted classes using binary search
        indices = cp.searchsorted(sorted_classes, labels)

        # Validate: check if the found position actually matches the label
        # Out-of-bounds indices are clamped to avoid index errors
        indices_safe = cp.minimum(indices, len(classes) - 1)
        valid = (indices < len(classes)) & (
            sorted_classes[indices_safe] == labels
        )

        # Map valid labels to their indices in original classes
        mapped_labels = cp.where(
            valid, sort_indices[indices_safe], len(classes)
        )

        if not copy:
            labels[:] = mapped_labels
            mapped_labels = labels

    return mapped_labels, classes
