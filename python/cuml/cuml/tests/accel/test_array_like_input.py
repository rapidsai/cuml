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

import numpy as np
import pytest
from cuml.internals.array import CumlArray
from cuml.internals.global_settings import GlobalSettings


@pytest.mark.parametrize(
    "input_data,expected",
    [
        ([1, 2, 3], np.array([1, 2, 3])),
        ((1, 2, 3), np.array([1, 2, 3])),
        ([[1, 2], [3, 4]], np.array([[1, 2], [3, 4]])),
        (((1, 2), (3, 4)), np.array([[1, 2], [3, 4]])),
    ],
)
def test_cumlarray_list_tuple_input(input_data, expected):
    """Test CumlArray construction with list/tuple inputs."""
    if not GlobalSettings().accelerator_active:
        pytest.skip("Skipping test because accelerator is not active")

    arr = CumlArray(input_data)
    assert isinstance(arr, CumlArray)
    np.testing.assert_array_equal(arr.to_output("numpy"), expected)


@pytest.mark.parametrize(
    "input_data,expected",
    [
        ([], np.array([])),
        ((), np.array([])),
        ([[]], np.array([[]])),
        (((),), np.array([[]])),
    ],
)
def test_cumlarray_construction_with_empty_lists(input_data, expected):
    """Test CumlArray construction with empty lists/tuples."""
    if not GlobalSettings().accelerator_active:
        pytest.skip("Skipping test because accelerator is not active")

    arr = CumlArray(input_data)
    assert isinstance(arr, CumlArray)
    np.testing.assert_array_equal(arr.to_output("numpy"), expected)


@pytest.mark.parametrize(
    "input_data,expected",
    [
        ([(1, 2), [3, 4]], np.array([[1, 2], [3, 4]])),
        ([1, 2.0, 3], np.array([1.0, 2.0, 3.0])),
    ],
)
def test_cumlarray_construction_with_mixed_types(input_data, expected):
    """Test CumlArray construction with mixed type inputs."""
    if not GlobalSettings().accelerator_active:
        pytest.skip("Skipping test because accelerator is not active")

    arr = CumlArray(input_data)
    assert isinstance(arr, CumlArray)
    np.testing.assert_array_equal(arr.to_output("numpy"), expected)


@pytest.mark.parametrize(
    "input_data,expected",
    [
        ([1, 2, 3], np.array([1, 2, 3])),
        ((1, 2, 3), np.array([1, 2, 3])),
        ([[1, 2], [3, 4]], np.array([[1, 2], [3, 4]])),
        (((1, 2), (3, 4)), np.array([[1, 2], [3, 4]])),
        (np.array([1, 2, 3]), np.array([1, 2, 3])),
        ([], np.array([])),
        ([(1, 2), [3, 4]], np.array([[1, 2], [3, 4]])),
    ],
)
def test_cumlarray_from_input(input_data, expected):
    """Test CumlArray.from_input() with array-like inputs."""
    if not GlobalSettings().accelerator_active:
        pytest.skip("Skipping test because accelerator is not active")

    arr = CumlArray.from_input(input_data)
    assert isinstance(arr, CumlArray)
    np.testing.assert_array_equal(arr.to_output("numpy"), expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_cumlarray_from_input_with_dtype(dtype):
    """Test CumlArray.from_input() with different dtypes."""
    if not GlobalSettings().accelerator_active:
        pytest.skip("Skipping test because accelerator is not active")

    arr = CumlArray.from_input([1, 2, 3], convert_to_dtype=dtype)
    assert isinstance(arr, CumlArray)
    np.testing.assert_array_equal(
        arr.to_output("numpy"), np.array([1, 2, 3], dtype=dtype)
    )
