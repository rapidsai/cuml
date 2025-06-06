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

import pickle

import cupy as cp
import numpy as np
import pytest

import cuml
import cuml.internals
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.internals.array import CumlArray
from cuml.internals.input_utils import (
    determine_array_dtype,
    determine_array_type,
    input_to_cuml_array,
)

test_input_types = ["numpy", "numba", "cupy", "cudf"]

test_output_types_str = ["numpy", "numba", "cupy", "cudf"]

test_dtypes_short = [
    np.uint8,
    np.float16,
    np.int32,
    np.float64,
]

unsupported_cudf_dtypes = [
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
]

test_shapes = [10, (10, 1), (10, 5), (1, 10)]


class DummyTestEstimator(cuml.Base):

    input_any_ = CumlArrayDescriptor()

    def _set_input(self, X):
        self.input_any_ = X

    @cuml.internals.api_base_return_any()
    def store_input(self, X):
        self.input_any_ = X

    @cuml.internals.api_return_any()
    def get_input(self):
        return self.input_any_

    # === Standard Functions ===
    def fit(self, X, convert_dtype=True) -> "DummyTestEstimator":
        self._set_base_attributes(output_type=X, n_features=X)
        return self

    def predict(self, X, convert_dtype=True) -> CumlArray:

        return X

    def transform(self, X, convert_dtype=False) -> CumlArray:

        pass

    def fit_transform(self, X, y=None) -> CumlArray:

        return self.fit(X).transform(X)


def assert_array_identical(a, b):

    cupy_a = input_to_cuml_array(a, order="K").array
    cupy_b = input_to_cuml_array(b, order="K").array

    if len(a) == 0 and len(b) == 0:
        return True

    assert cupy_a.shape == cupy_b.shape
    assert cupy_a.dtype == cupy_b.dtype
    assert cupy_a.order == cupy_b.order
    assert cp.all(cp.asarray(cupy_a) == cp.asarray(cupy_b)).item()


def create_input(input_type, input_dtype, input_shape, input_order):
    rand_ary = cp.ones(input_shape, dtype=input_dtype, order=input_order)

    cuml_ary = CumlArray(rand_ary)

    return cuml_ary.to_output(input_type)


def create_output(X_in, output_type):

    cuml_ary_tuple = input_to_cuml_array(X_in, order="K")

    return cuml_ary_tuple.array.to_output(output_type)


@pytest.mark.parametrize("input_type", test_input_types)
def test_pickle(input_type):

    if input_type == "numba":
        pytest.skip("numba arrays cant be picked at this time")

    est = DummyTestEstimator()

    X_in = create_input(input_type, np.float32, (10, 5), "C")

    est.store_input(X_in)

    # Loop and verify we have filled the cache
    for out_type in test_output_types_str:
        with cuml.using_output_type(out_type):
            assert_array_identical(
                est.input_any_, create_output(X_in, out_type)
            )

    est_pickled_bytes = pickle.dumps(est)
    est_unpickled: DummyTestEstimator = pickle.loads(est_pickled_bytes)

    # Assert that we only resture the input
    assert est_unpickled.__dict__["input_any_"].input_type == input_type
    assert len(est_unpickled.__dict__["input_any_"].values) == 1

    assert_array_identical(est.get_input(), est_unpickled.get_input())
    assert_array_identical(est.input_any_, est_unpickled.input_any_)

    # Loop one more time with the picked one to make sure it works right
    for out_type in test_output_types_str:
        with cuml.using_output_type(out_type):
            assert_array_identical(
                est.input_any_, create_output(X_in, out_type)
            )

        est_unpickled.output_type = out_type

        assert_array_identical(
            est_unpickled.input_any_, create_output(X_in, out_type)
        )


@pytest.mark.parametrize("input_type", test_input_types)
@pytest.mark.parametrize("input_dtype", [np.float32, np.int16])
@pytest.mark.parametrize("input_shape", [10, (10, 5)])
@pytest.mark.parametrize("output_type", test_output_types_str)
def test_dec_input_output(input_type, input_dtype, input_shape, output_type):

    if input_type == "cudf" or output_type == "cudf":
        if input_dtype in unsupported_cudf_dtypes:
            pytest.skip("Unsupported cudf combination")

    X_in = create_input(input_type, input_dtype, input_shape, "C")
    X_out = create_output(X_in, output_type)

    # Test with output_type="input"
    est = DummyTestEstimator(output_type="input")

    est.store_input(X_in)

    # Test is was stored internally correctly
    assert X_in is est.get_input()

    assert est.__dict__["input_any_"].input_type == input_type

    # Check the current type matches input type
    assert determine_array_type(est.input_any_) == input_type

    assert_array_identical(est.input_any_, X_in)

    # Switch output type and check type and equality
    with cuml.using_output_type(output_type):

        assert determine_array_type(est.input_any_) == output_type

        assert_array_identical(est.input_any_, X_out)

    # Now Test with output_type=output_type
    est = DummyTestEstimator(output_type=output_type)

    est.store_input(X_in)

    # Check the current type matches output type
    assert determine_array_type(est.input_any_) == output_type

    assert_array_identical(est.input_any_, X_out)

    with cuml.using_output_type("input"):

        assert determine_array_type(est.input_any_) == input_type

        assert_array_identical(est.input_any_, X_in)


@pytest.mark.parametrize("input_type", test_input_types)
@pytest.mark.parametrize("input_dtype", [np.float32, np.int16])
@pytest.mark.parametrize("input_shape", test_shapes)
def test_auto_fit(input_type, input_dtype, input_shape):
    """
    Test autowrapping on fit that will set output_type, and n_features
    """
    X_in = create_input(input_type, input_dtype, input_shape, "C")

    # Test with output_type="input"
    est = DummyTestEstimator()

    est.fit(X_in)

    def calc_n_features(shape):
        if isinstance(shape, tuple) and len(shape) >= 1:

            # When cudf and shape[1] is used, a series is created which will
            # remove the last shape
            if input_type == "cudf" and shape[1] == 1:
                return 1

            return shape[1]

        return 1

    assert est._input_type == input_type
    assert est.target_dtype is None
    assert est.n_features_in_ == calc_n_features(input_shape)


@pytest.mark.parametrize("input_type", test_input_types)
@pytest.mark.parametrize("base_output_type", test_input_types)
@pytest.mark.parametrize(
    "global_output_type", test_output_types_str + ["input", None]
)
def test_auto_predict(input_type, base_output_type, global_output_type):
    """
    Test autowrapping on predict that will set target_type
    """
    X_in = create_input(input_type, np.float32, (10, 10), "F")

    # Test with output_type="input"
    est = DummyTestEstimator()

    # With cuml.global_settings.output_type == None, this should return the
    # input type
    X_out = est.predict(X_in)

    assert determine_array_type(X_out) == input_type

    assert_array_identical(X_in, X_out)

    # Test with output_type=base_output_type
    est = DummyTestEstimator(output_type=base_output_type)

    # With cuml.global_settings.output_type == None, this should return the
    # base_output_type
    X_out = est.predict(X_in)

    assert determine_array_type(X_out) == base_output_type

    assert_array_identical(X_in, X_out)

    # Test with global_output_type, should return global_output_type
    with cuml.using_output_type(global_output_type):
        X_out = est.predict(X_in)

        target_output_type = global_output_type

        if target_output_type is None or target_output_type == "input":
            target_output_type = base_output_type

        if target_output_type == "input":
            target_output_type = input_type

        assert determine_array_type(X_out) == target_output_type

        assert_array_identical(X_in, X_out)


@pytest.mark.parametrize("input_arg", ["X", "y", "bad", ...])
@pytest.mark.parametrize("target_arg", ["X", "y", "bad", ...])
@pytest.mark.parametrize("get_output_type", [True, False])
@pytest.mark.parametrize("get_output_dtype", [True, False])
def test_return_array(
    input_arg: str,
    target_arg: str,
    get_output_type: bool,
    get_output_dtype: bool,
):
    """
    Test autowrapping on predict that will set target_type
    """

    input_type_X = "numpy"
    input_dtype_X = np.float64

    input_type_Y = "cupy"
    input_dtype_Y = np.int32

    inner_type = "numba"
    inner_dtype = np.float16

    X_in = create_input(input_type_X, input_dtype_X, (10, 10), "F")
    Y_in = create_input(input_type_Y, input_dtype_Y, (10, 10), "F")

    def test_func(X, y):

        if not get_output_type:
            cuml.internals.set_api_output_type(inner_type)

        if not get_output_dtype:
            cuml.internals.set_api_output_dtype(inner_dtype)

        return X

    expected_to_fail = (input_arg == "bad" and get_output_type) or (
        target_arg == "bad" and get_output_dtype
    )

    try:
        test_func = cuml.internals.api_return_array(
            input_arg=input_arg,
            target_arg=target_arg,
            get_output_type=get_output_type,
            get_output_dtype=get_output_dtype,
        )(test_func)
    except ValueError:
        assert expected_to_fail
        return
    else:
        assert not expected_to_fail

    X_out = test_func(X=X_in, y=Y_in)

    target_type = None
    target_dtype = None

    if not get_output_type:
        target_type = inner_type
    else:
        if input_arg == "y":
            target_type = input_type_Y
        else:
            target_type = input_type_X

    if not get_output_dtype:
        target_dtype = inner_dtype
    else:
        if target_arg == "X":
            target_dtype = input_dtype_X
        else:
            target_dtype = input_dtype_Y

    assert determine_array_type(X_out) == target_type

    assert determine_array_dtype(X_out) == target_dtype
