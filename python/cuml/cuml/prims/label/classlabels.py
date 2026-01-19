#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp

from cuml.internals.input_utils import input_to_cupy_array


def make_monotonic(labels, classes=None, copy=False):
    """
    Takes a set of labels that might not be drawn from the
    set [0, n-1] and renumbers them to be drawn that
    interval.

    Replaces labels not present in classes by len(classes)+1.

    Parameters
    ----------

    labels : array-like of size (n,) labels to convert
    classes : array-like of size (n_classes,) the unique
              set of classes in the set of labels
    copy : boolean if true, a copy will be returned and the
           operation will not be done in place.

    Returns
    -------

    mapped_labels : array-like of size (n,)
    classes : array-like of size (n_classes,)
    """
    labels = input_to_cupy_array(labels, deepcopy=copy).array

    if labels.ndim != 1:
        raise ValueError("Labels array must be 1D")

    if classes is None:
        classes = cp.unique(labels)
    else:
        classes = input_to_cupy_array(classes).array

    # Sort classes to ensure consistent ordering
    classes = cp.sort(classes)

    # Create mapping from original labels to monotonic indices [0, n_classes-1]
    # Use searchsorted to find the position of each label in the sorted classes
    mapped_labels = cp.searchsorted(classes, labels)

    # Check which labels are actually present in classes
    # Labels that match their corresponding class get their index,
    # others get len(classes)
    valid_mask = (mapped_labels < len(classes)) & (
        classes[mapped_labels] == labels
    )

    # Set invalid labels (not in classes) to len(classes)
    if copy:
        mapped_labels = cp.where(valid_mask, mapped_labels, len(classes))
    else:
        # Modify in-place
        labels[:] = cp.where(valid_mask, mapped_labels, len(classes))
        mapped_labels = labels

    return mapped_labels, classes
