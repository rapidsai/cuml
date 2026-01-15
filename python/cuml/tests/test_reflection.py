# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import pickle

import cudf
import cudf.pandas
import cupy as cp
import cupyx.scipy.sparse
import numpy as np
import pandas as pd
import pytest
import scipy.sparse
from numba.cuda import as_cuda_array, is_cuda_array

import cuml
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.internals import reflect, run_in_internal_context
from cuml.internals.array import CumlArray
from cuml.internals.array_sparse import SparseCumlArray
from cuml.internals.base import Base
from cuml.internals.global_settings import GlobalSettings

OUTPUT_TYPES = ["numpy", "numba", "cupy", "cudf", "pandas"]


@pytest.fixture(autouse=True)
def reset_global_output_type():
    yield
    # Ensure we reset the type at the end of the test
    cuml.set_global_output_type(None)


def assert_output_type(arr, output_type):
    if output_type == "numba":
        assert is_cuda_array(arr)
    else:
        cls = {
            "numpy": np.ndarray,
            "cupy": cp.ndarray,
            "cudf": (cudf.Series, cudf.DataFrame),
            "pandas": (pd.Series, pd.DataFrame),
        }[output_type]
        assert isinstance(arr, cls)


def rand_array(output_type, *, shape=(8, 4), seed=42):
    X = cp.random.default_rng(seed).uniform(
        low=0.0, high=10.0, size=shape, dtype="float32"
    )
    if output_type == "numba":
        return as_cuda_array(X)
    elif output_type == "cupy":
        return X
    elif output_type == "numpy":
        return cp.asnumpy(X)
    elif output_type == "pandas":
        return pd.DataFrame(X.get())
    else:
        assert output_type == "cudf"
        return cudf.DataFrame(X)


class DummyEstimator(Base):
    X_ = CumlArrayDescriptor()

    @reflect(reset=True)
    def fit(self, X, y=None):
        self.X_ = CumlArray.from_input(X)
        return self

    @reflect
    def example(self, X):
        return cp.zeros(3)

    @reflect
    def example_no_args(self):
        return cp.zeros(3)

    @run_in_internal_context
    def check_descriptor(self):
        # When run in an internal context, a descriptor returns its original
        # internal value.
        assert isinstance(self.X_, CumlArray)

        with cuml.using_output_type("cupy"):
            # Can override with using_output_type
            assert_output_type(self.X_, "cupy")


@reflect
def reflects_input(X):
    return X


@reflect
def returns_array_no_args():
    return cp.ones(3)


@reflect(array=None)
def returns_array_one_arg(n):
    return cp.ones(n)


def test_deprecated_memory_utils():
    for name in ["set_global_output_type", "using_output_type"]:
        with pytest.warns(FutureWarning, match=name):
            func = getattr(cuml.internals.memory_utils, name)
        assert func is getattr(cuml, name)

    # Unknown attributes error
    with pytest.raises(AttributeError, match="not_a_real_attr"):
        cuml.internals.memory_utils.not_a_real_attr


def test_set_global_output_type():
    gs = GlobalSettings()
    assert gs.output_type is None

    cuml.set_global_output_type("cupy")
    assert gs.output_type == "cupy"

    cuml.set_global_output_type(None)
    assert gs.output_type is None

    with pytest.raises(ValueError, match="`output_type` must be one of"):
        cuml.set_global_output_type("bad")


def test_using_output_type():
    gs = GlobalSettings()
    assert gs.output_type is None

    cuml.set_global_output_type("cupy")
    assert gs.output_type == "cupy"

    with cuml.using_output_type("cudf"):
        assert gs.output_type == "cudf"
    assert gs.output_type == "cupy"

    with cuml.using_output_type(None):
        assert gs.output_type is None
    assert gs.output_type == "cupy"

    with pytest.raises(ValueError, match="`output_type` must be one of"):
        with cuml.using_output_type("bad"):
            pass


@pytest.mark.parametrize("input_type", OUTPUT_TYPES)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_default_output_type(input_type):
    X = rand_array(input_type)
    model = cuml.DBSCAN(eps=1.0, min_samples=1)
    labels = model.fit_predict(X)
    assert_output_type(labels, input_type)
    assert_output_type(model.components_, input_type)


@pytest.mark.parametrize("input_type", OUTPUT_TYPES)
@pytest.mark.parametrize("output_type", OUTPUT_TYPES)
def test_estimator_output_type(input_type, output_type):
    X = rand_array(input_type)
    model = cuml.DBSCAN(eps=1.0, min_samples=1, output_type=output_type)
    labels = model.fit_predict(X)
    assert_output_type(labels, output_type)
    assert_output_type(model.components_, output_type)


@pytest.mark.parametrize("input_type", OUTPUT_TYPES)
@pytest.mark.parametrize("output_type", OUTPUT_TYPES)
def test_global_output_type(input_type, output_type):
    cuml.set_global_output_type(output_type)

    X = rand_array(input_type)
    model = cuml.DBSCAN(eps=1.0, min_samples=1)
    labels = model.fit_predict(X)
    assert_output_type(labels, output_type)
    assert_output_type(model.components_, output_type)


def test_invalid_estimator_output_type():
    X = rand_array("numpy")
    model = cuml.DBSCAN(eps=1.0, min_samples=1, output_type="invalid")
    model.fit(X)
    assert model.output_type == "invalid"

    # Descriptor raises appropriately
    with pytest.raises(ValueError, match="`output_type='invalid'`"):
        model.components_

    # Method raises appropriately
    with pytest.raises(ValueError, match="`output_type='invalid'`"):
        model.fit_predict(X)


def test_global_overrides_estimator_output_type():
    cuml.set_global_output_type("numpy")
    X = rand_array("pandas")
    model = cuml.DBSCAN(eps=1.0, min_samples=1, output_type="cupy")
    labels = model.fit_predict(X)
    assert_output_type(labels, "numpy")
    assert_output_type(model.components_, "numpy")


def test_global_input_with_estimator_output_type():
    cuml.set_global_output_type("input")
    X = rand_array("pandas")
    model = cuml.DBSCAN(eps=1.0, min_samples=1, output_type="cupy")
    labels = model.fit_predict(X)
    # The difference here is probably a bug, but it's been the behavior for a
    # long time. Methods respect `estimator.output_type` if the global
    # `output_type` is 'input', while attributes respect the global
    # `output_type`.
    assert_output_type(labels, "cupy")
    assert_output_type(model.components_, "pandas")


@pytest.mark.parametrize(
    "construct",
    [
        pytest.param(lambda x, y: [x, 1, y], id="list"),
        pytest.param(lambda x, y: (x, 1, y), id="tuple"),
        pytest.param(lambda x, y: {"x": x, "y": y, "z": 1}, id="dict"),
        pytest.param(lambda x, y: {"a": [(x, 1), (y, 2)]}, id="nested"),
    ],
)
@pytest.mark.parametrize("output_type", ["input", "cupy", "cudf"])
def test_convert_nested_outputs(construct, output_type):
    cuml.set_global_output_type(output_type)
    x = rand_array("numpy")

    @reflect(array="x")
    def apply(func, x):
        return func(x, x + 1)

    res = apply(construct, x)
    sol = construct("x", "y")

    expected_type = "numpy" if output_type == "input" else output_type

    def check_nested_types(res, sol):
        """Check types match, using `x` and `y` as placeholders for arrays"""
        if sol in ("x", "y"):
            assert_output_type(res, expected_type)
        else:
            assert type(res) is type(sol)
            if isinstance(res, dict):
                assert set(res) == set(sol)
                for k in res:
                    check_nested_types(res[k], sol[k])
            elif isinstance(res, (tuple, list)):
                assert len(res) == len(sol)
                for r, s in zip(res, sol):
                    check_nested_types(r, s)

    check_nested_types(res, sol)


@pytest.mark.parametrize("sparse_type", ["cupy", "numpy", "cuml"])
@pytest.mark.parametrize("output_type", [None, *OUTPUT_TYPES])
def test_convert_sparse_outputs(sparse_type, output_type):
    @reflect
    def make_sparse():
        arr = cupyx.scipy.sparse.random(5, 5, random_state=42)
        if sparse_type == "cupy":
            return arr
        elif sparse_type == "numpy":
            return arr.get()
        else:
            return SparseCumlArray(arr)

    cuml.set_global_output_type(output_type)
    res = make_sparse()

    if output_type == "cuml":
        assert isinstance(res, SparseCumlArray)
    elif output_type in [None, "input", "cupy", "cudf", "numba"]:
        assert cupyx.scipy.sparse.issparse(res)
    else:
        assert scipy.sparse.issparse(res)


@pytest.mark.parametrize("output_type", [None, *OUTPUT_TYPES])
def test_functions(output_type):
    cuml.set_global_output_type(output_type)
    X = rand_array("numpy")

    # Reflected functions treat None/"input" the same
    assert_output_type(
        reflects_input(X),
        "numpy" if output_type in (None, "input") else output_type,
    )

    # With no array argument functions default to 'cupy' unless
    # a concrete type is configured
    expected = "cupy" if output_type in (None, "input") else output_type
    assert_output_type(returns_array_no_args(), expected)
    assert_output_type(returns_array_one_arg(3), expected)


@pytest.mark.parametrize("output_type", [None, "input", "numpy"])
def test_internal_calls(output_type):
    @reflect(array="X")
    def apply(func, X):
        result = func(X)
        # Internal calls return internal types by default
        assert isinstance(result, CumlArray)

        with cuml.using_output_type("cupy"):
            temp = func(X)

        # Internal calls can configure output type to get
        # something specific when needed
        assert isinstance(temp, cp.ndarray)

        return result

    cuml.set_global_output_type(output_type)
    X = rand_array("pandas")
    res = apply(reflects_input, X)
    expected = "pandas" if output_type in (None, "input") else output_type
    assert_output_type(res, expected)


def test_run_in_internal_context():
    @run_in_internal_context
    def always_returns_numpy(func, X):
        result = func(X)
        # Internal calls return internal types by default
        assert isinstance(result, CumlArray)

        return result.to_output("numpy")

    cuml.set_global_output_type("cudf")
    X = rand_array("pandas")
    res = always_returns_numpy(reflects_input, X)
    assert_output_type(res, "numpy")


def test_reset_true():
    X = rand_array("numpy", shape=(10, 5))
    model = DummyEstimator().fit(X)
    assert model.n_features_in_ == 5
    assert model._input_type == "numpy"


def test_estimator_method_with_array_input():
    X = rand_array("numpy", shape=(10, 5))
    X2 = rand_array("cudf", shape=(10, 5))
    model = DummyEstimator().fit(X)

    # Reflects method input by default
    assert_output_type(model.example(X2), "cudf")

    # Estimator output_type can override
    model.output_type = "pandas"
    assert_output_type(model.example(X2), "pandas")

    # Global output type overrides
    with cuml.using_output_type("cupy"):
        assert_output_type(model.example(X2), "cupy")


def test_estimator_method_with_no_array_input():
    X = rand_array("numpy", shape=(10, 5))
    model = DummyEstimator().fit(X)

    # Reflects fit input by default
    assert_output_type(model.example_no_args(), "numpy")

    # Estimator output_type can override
    model.output_type = "cupy"
    assert_output_type(model.example_no_args(), "cupy")

    # Global output type overrides
    with cuml.using_output_type("pandas"):
        assert_output_type(model.example_no_args(), "pandas")


def test_cuml_array_descriptor_type_in_internal_context():
    X = rand_array("numpy")
    model = DummyEstimator().fit(X)
    assert_output_type(model.X_, "numpy")
    model.check_descriptor()


def test_array_descriptor_cache_behavior():
    X = rand_array("cupy")
    model = DummyEstimator().fit(X)
    assert_output_type(model.X_, "cupy")
    # Instance is cached
    assert model.X_ is model.X_
    assert len(model.__dict__["X_"].values) == 2  # cuml + cupy

    with cuml.using_output_type("pandas"):
        assert_output_type(model.X_, "pandas")
        # Instance is cached, but original cache isn't wiped
        assert model.X_ is model.X_
        assert len(model.__dict__["X_"].values) == 3  # cuml + cupy + pandas

    msg = pickle.dumps(model)
    model2 = pickle.loads(msg)
    # Only one array is serialized when pickled, and the cache is reset
    assert b"pandas" not in msg
    assert_output_type(model2.X_, "cupy")
    assert len(model2.__dict__["X_"].values) == 2  # cuml + cupy
