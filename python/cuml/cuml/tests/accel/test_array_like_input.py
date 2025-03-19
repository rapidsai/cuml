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
from cuml.internals.array import CumlArray


def test_cumlarray_list_tuple_input(test_input, expected_dtype):
    """Test CumlArray initialization with list/tuple inputs when accelerator is active."""
    cuml_array = CumlArray(test_input)
    assert isinstance(cuml_array._owner, np.ndarray)
    np.testing.assert_array_equal(
        cuml_array.to_output("numpy"), np.array(test_input)
    )
    if len(test_input) > 0:  # Only check dtype for non-empty inputs
        assert cuml_array.dtype == expected_dtype
