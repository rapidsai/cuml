#
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import dask.array as da
import dask.delayed
from cuml.internals.safe_imports import gpu_only_import

cp = gpu_only_import("cupy")


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
