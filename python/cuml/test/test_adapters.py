#
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

import cupy as cp
import pytest

from cupyx.scipy.sparse import coo_matrix

from cuml.thirdparty_adapters.adapters import check_array


def test_check_array():
    # accept_sparse
    arr = coo_matrix((3, 4), dtype=cp.float64)
    check_array(arr, accept_sparse=True)
    with pytest.raises(ValueError):
        check_array(arr, accept_sparse=False)

    # dtype
    arr = cp.array([[1, 2]], dtype=cp.int64)
    check_array(arr, dtype=cp.int64, copy=False)

    arr = cp.array([[1, 2]], dtype=cp.float32)
    new_arr = check_array(arr, dtype=cp.int64)
    assert new_arr.dtype == cp.int64

    # order
    arr = cp.array([[1, 2]], dtype=cp.int64, order='F')
    new_arr = check_array(arr, order='F')
    assert new_arr.flags.f_contiguous
    new_arr = check_array(arr, order='C')
    assert new_arr.flags.c_contiguous

    # force_all_finite
    arr = cp.array([[1, cp.inf]])
    check_array(arr, force_all_finite=False)
    with pytest.raises(ValueError):
        check_array(arr, force_all_finite=True)

    # ensure_2d
    arr = cp.array([1, 2], dtype=cp.float32)
    check_array(arr, ensure_2d=False)
    with pytest.raises(ValueError):
        check_array(arr, ensure_2d=True)

    # ensure_min_samples
    arr = cp.array([[1, 2]], dtype=cp.float32)
    check_array(arr, ensure_min_samples=1)
    with pytest.raises(ValueError):
        check_array(arr, ensure_min_samples=2)

    # ensure_min_features
    arr = cp.array([[]], dtype=cp.float32)
    check_array(arr, ensure_min_features=0)
    with pytest.raises(ValueError):
        check_array(arr, ensure_min_features=1)
