# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
from hypothesis.extra.numpy import (
    array_shapes,
    arrays,
    floating_dtypes,
    integer_dtypes,
)
from hypothesis import example, target
from hypothesis import strategies as st
from hypothesis import given, note
from cuml.testing.utils import array_equal, assert_array_equal
import pytest
from cuml.internals.safe_imports import cpu_only_import

np = cpu_only_import("numpy")


@example(array=np.array([1, 2, 3]), tol=1e-4)
@given(
    arrays(
        dtype=st.one_of(floating_dtypes(), integer_dtypes()),
        shape=array_shapes(),
    ),
    st.floats(1e-4, 1.0),
)
@pytest.mark.filterwarnings("ignore:invalid value encountered in subtract")
def test_array_equal_same_array(array, tol):
    equal = array_equal(array, array, tol)
    note(equal)
    difference = equal.compute_difference()
    if np.isfinite(difference):
        target(float(np.abs(difference)))
    assert equal
    assert equal == True  # noqa: E712
    assert bool(equal) is True
    assert_array_equal(array, array, tol)


@example(
    arrays=(np.array([1, 2, 3]), np.array([1, 2, 3])),
    unit_tol=1e-4,
    with_sign=False,
)
@given(
    arrays=array_shapes().flatmap(
        lambda shape: st.tuples(
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
    unit_tol=st.floats(1e-4, 1.0),
    with_sign=st.booleans(),
)
@pytest.mark.filterwarnings("ignore:invalid value encountered in subtract")
def test_array_equal_two_arrays(arrays, unit_tol, with_sign):
    array_a, array_b = arrays
    equal = array_equal(array_a, array_b, unit_tol, with_sign=with_sign)
    equal_flipped = array_equal(
        array_b, array_a, unit_tol, with_sign=with_sign
    )
    note(equal)
    difference = equal.compute_difference()
    a, b = (
        (array_a, array_b) if with_sign else (np.abs(array_a), np.abs(array_b))
    )
    expect_equal = np.sum(np.abs(a - b) > unit_tol) / array_a.size < 1e-4
    if expect_equal:
        assert_array_equal(array_a, array_b, unit_tol, with_sign=with_sign)
        assert equal
        assert bool(equal) is True
        assert equal == True  # noqa: E712
        assert True == equal  # noqa: E712
        assert equal != False  # noqa: E712
        assert False != equal  # noqa: E712
        assert equal_flipped
        assert bool(equal_flipped) is True
        assert equal_flipped == True  # noqa: E712
        assert True == equal_flipped  # noqa: E712
        assert equal_flipped != False  # noqa: E712
        assert False != equal_flipped  # noqa: E712
    else:
        with pytest.raises(AssertionError):
            assert_array_equal(array_a, array_b, unit_tol, with_sign=with_sign)
        assert not equal
        assert bool(equal) is not True
        assert equal != True  # noqa: E712
        assert True != equal  # noqa: E712
        assert equal == False  # noqa: E712
        assert False == equal  # noqa: E712
        assert difference != 0
