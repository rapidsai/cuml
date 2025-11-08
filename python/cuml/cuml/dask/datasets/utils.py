#
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import dask.array as da
import dask.delayed


def _get_X(t):
    return t[0]


def _get_labels(t):
    return t[1]


def _dask_array_from_delayed(part, dtype, nrows, ncols=None):
    # NOTE: ncols = None is for when we want to create a
    # flat array of ndim == 1. When ncols = 1, we go ahead
    # and make an array of shape (nrows, 1)

    shape = (nrows, ncols) if ncols else (nrows,)
    return da.from_delayed(
        dask.delayed(part), shape=shape, meta=cp.zeros((1)), dtype=dtype
    )


def _create_delayed(parts, dtype, rows_per_part, ncols=None):
    """
    This function takes a list of GPU futures and returns
    a list of delayed dask arrays, with each part having
    a corresponding dask.array in the list
    """

    return [
        _dask_array_from_delayed(part, dtype, rows_per_part[idx], ncols)
        for idx, part in enumerate(parts)
    ]
