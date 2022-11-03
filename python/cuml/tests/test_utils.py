# Copyright (c) 2022, NVIDIA CORPORATION.
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
from cuml.testing.utils import array_equal
from hypothesis import given, note
from hypothesis import strategies as st
from hypothesis import target
from hypothesis.extra.numpy import (array_shapes, arrays, floating_dtypes,
                                    integer_dtypes)


@given(
    arrays(
        dtype=st.one_of(floating_dtypes(), integer_dtypes()),
        shape=array_shapes(),
    ),
    st.floats(1e-4, 1.0))
def test_array_equal_same_array(array, tol):
    equal = array_equal(array, array, tol)
    note(equal)
    difference = equal.compute_difference()
    if np.isfinite(difference):
        target(float(np.abs(difference)))
    assert equal


@given(
    array_shapes().flatmap(
        lambda shape:
            st.tuples(
                arrays(
                    dtype=st.one_of(floating_dtypes(), integer_dtypes()),
                    shape=shape,
                ),
                arrays(
                    dtype=st.one_of(floating_dtypes(), integer_dtypes()),
                    shape=shape,
                ),
            )
    ),
    st.floats(1e-4, 1.0)
)
def test_array_equal_two_arrays(arrays, tol):
    array_a, array_b = arrays
    equal = array_equal(array_a, array_b, tol)
    note(equal)
    difference = equal.compute_difference()
    if np.isfinite(difference):
        target(float(np.abs(difference)))
    assert equal or np.abs(difference) != 0
