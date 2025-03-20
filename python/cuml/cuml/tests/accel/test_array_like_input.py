#
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

import pytest
import numpy as np
import cupy as cp
from cuml.internals.array import CumlArray
from cuml.internals.api_decorators import support_array_like
from cuml.internals.global_settings import GlobalSettings


@pytest.mark.skipif(
    not GlobalSettings().accelerator_active,
    reason="These tests require an active accelerator",
)
def test_support_array_like():
    """Test the support_array_like function for converting list/tuple inputs to numpy arrays."""
    # Test with list input
    test_list = [1, 2, 3]
    result = support_array_like(test_list)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array(test_list))

    # Test with tuple input
    test_tuple = (1, 2, 3)
    result = support_array_like(test_tuple)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array(test_tuple))

    # Test with numpy array input (should be returned as is)
    test_array = np.array([1, 2, 3])
    result = support_array_like(test_array)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, test_array)

    # Test with cupy array input (should be returned as is)
    test_cupy = cp.array([1, 2, 3])
    result = support_array_like(test_cupy)
    assert isinstance(result, cp.ndarray)
    cp.testing.assert_array_equal(result, test_cupy)


@pytest.mark.skipif(
    not GlobalSettings().accelerator_active,
    reason="These tests require an active accelerator",
)
@pytest.mark.parametrize(
    "test_input,expected_dtype",
    [
        ([1, 2, 3], np.int64),
        ((1, 2, 3), np.int64),
        ([[1, 2], [3, 4]], np.int64),
        (((1, 2), (3, 4)), np.int64),
        ([(1, 2), [3, 4]], np.int64),
        ([1.0, 2.0, 3.0], np.float64),
        ([], np.float64),
        ((), np.float64),
        ([1], np.int64),
        ((1,), np.int64),
    ],
)
def test_cumlarray_list_tuple_input(test_input, expected_dtype):
    """Test CumlArray initialization with list/tuple inputs when accelerator is active."""
    cuml_array = CumlArray(test_input)
    assert isinstance(cuml_array._owner, np.ndarray)
    np.testing.assert_array_equal(
        cuml_array.to_output("numpy"), np.array(test_input)
    )
    if len(test_input) > 0:  # Only check dtype for non-empty inputs
        assert cuml_array.dtype == expected_dtype
