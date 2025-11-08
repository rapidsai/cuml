#
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#


import cupy as cp


def sorted_unique_labels(*ys):
    """Extract an ordered array of unique labels from one or more arrays of
    labels."""
    ys = (cp.unique(y) for y in ys)
    labels = cp.unique(cp.concatenate(ys))
    return labels
