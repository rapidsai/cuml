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

from cuml.internals.api_decorators import _get_value
from cuml.internals.input_utils import is_array_like


def test_is_array_like_with_lists():
    """Test is_array_like function with list/tuple inputs."""
    # Test lists and tuples are accepted when accept_lists=True
    assert is_array_like([1, 2, 3], accept_lists=True)
    assert is_array_like((1, 2, 3), accept_lists=True)

    # Test lists and tuples are rejected when accept_lists=False
    assert not is_array_like([1, 2, 3], accept_lists=False)
    assert not is_array_like((1, 2, 3), accept_lists=False)

    # Test numpy arrays are always accepted
    assert is_array_like(np.array([1, 2, 3]), accept_lists=True)
    assert is_array_like(np.array([1, 2, 3]), accept_lists=False)


def test_get_value_with_lists():
    """Test _get_value function with list/tuple inputs."""
    # Test list input is converted to numpy array
    value = _get_value(
        [], {"test": [1, 2, 3]}, "test", 0, None, accept_lists=True
    )
    assert isinstance(value, np.ndarray)
    np.testing.assert_array_equal(value, np.array([1, 2, 3]))

    # Test tuple input is converted to numpy array
    value = _get_value(
        [], {"test": (1, 2, 3)}, "test", 0, None, accept_lists=True
    )
    assert isinstance(value, np.ndarray)
    np.testing.assert_array_equal(value, np.array([1, 2, 3]))

    # Test non-list/tuple inputs are not converted
    value = _get_value(
        [], {"test": "string"}, "test", 0, None, accept_lists=True
    )
    assert isinstance(value, str)
    assert value == "string"

    # Test list input is not converted
    value = _get_value(
        [], {"test": [1, 2, 3]}, "test", 0, None, accept_lists=False
    )
    assert isinstance(value, list)
    assert value == [1, 2, 3]

    # Test tuple input is not converted
    value = _get_value(
        [], {"test": (1, 2, 3)}, "test", 0, None, accept_lists=False
    )
    assert isinstance(value, tuple)
    assert value == (1, 2, 3)

    # Test non-list/tuple inputs are not converted
    value = _get_value(
        [], {"test": "string"}, "test", 0, None, accept_lists=False
    )
    assert isinstance(value, str)
    assert value == "string"
