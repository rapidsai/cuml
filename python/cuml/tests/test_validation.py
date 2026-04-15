# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import re
import warnings
from contextlib import contextmanager

import cudf
import cupy as cp
import cupyx.scipy.sparse as cp_sp
import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import sklearn
from hypothesis import assume, example, given
from sklearn.exceptions import DataConversionWarning

from cuml.internals.validation import (
    _get_feature_names,
    _get_n_features,
    check_all_finite,
    check_array,
    check_consistent_length,
    check_features,
    check_non_negative,
    check_random_seed,
    check_y,
)


def gen_dense_array(
    *,
    kind="cupy",
    dtype="float32",
    order="C",
    ndim=2,
    n_samples=5,
    n_features=4,
):
    """Generate a suitable dense array input."""
    shape = (n_samples, n_features) if ndim == 2 else (n_samples,)
    dtype = np.dtype(dtype)

    if kind in ("cupy", "numpy", "pandas") and order is None:
        # generate a non-contiguous strided input if order=None
        shape = (n_samples * 2, n_features) if ndim == 2 else (n_samples * 2,)

    rng = np.random.default_rng(42)
    if dtype.kind == "f":
        data = rng.uniform(0, 100, size=shape).astype(dtype)
    else:
        data = rng.integers(0, 100, size=shape).astype(dtype)

    if kind == "cupy":
        out = cp.asarray(data, dtype=dtype, order=order)
        return out[::2] if order is None else out
    elif kind == "numpy":
        out = np.asarray(data, dtype=dtype, order=order)
        return out[::2] if order is None else out
    elif kind == "list":
        return data.tolist()
    elif kind == "pandas":
        out = pd.Series(data) if ndim == 1 else pd.DataFrame(data)
        return out[::2] if order is None else out
    elif kind == "cudf":
        return cudf.Series(data) if ndim == 1 else cudf.DataFrame(data)


@st.composite
def dense_arrays(
    draw,
    kind=None,
    dtype=None,
    order=None,
    ndim=2,
    n_samples=5,
    n_features=4,
):
    """A strategy for generating dense array inputs.

    Parameters
    ----------
    kind : {'cupy', 'numpy', 'list', 'pandas', 'cudf'} or list
        The input kind(s) to select from.
    dtype : dtype-like or list[dtype]
        The dtype(s) to select from.
    order : {'C', 'F', None} or list
        The contiguity order requirement(s) to select from.
    ndim : {1, 2} or list
        The number of dimensions to select from.
    n_samples : int
        The number of samples to generate
    n_features : int
        The number of features to generate (if ndim=2).
    """

    def select(value, choices, cast=None):
        if value is None:
            value = choices
        if isinstance(value, (list, tuple)):
            value = draw(st.sampled_from(value))
        if cast is not None:
            value = cast(value)
        assert value in choices
        return value

    kind = select(kind, ("cupy", "numpy", "list", "pandas", "cudf"))
    dtype = select(
        dtype,
        ("i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f2", "f4", "f8"),
        cast=np.dtype,
    )
    assume(not (kind == "cudf" and dtype == "float16"))

    ndim = select(ndim, (1, 2))

    if kind in ("cupy", "numpy", "pandas"):
        order = select(order, ("C", "F", None) if ndim == 2 else ("C", None))
    else:
        # Other containers don't have flexible contiguity
        order = None

    return gen_dense_array(
        kind=kind,
        dtype=dtype,
        order=order,
        ndim=ndim,
        n_samples=n_samples,
        n_features=n_features,
    )


@st.composite
def sparse_arrays(
    draw,
    kind=None,
    dtype=None,
    format=None,
    n_samples=5,
    n_features=4,
    density=0.5,
):
    """A strategy for generating sparse array inputs.

    Parameters
    ----------
    kind : {'cupy', 'scipy', 'scipy-array'} or list
        The input kind(s) to select from.
    dtype : dtype-like or list[dtype]
        The dtype(s) to select from.
    format : sparse format or list
        The format(s) to select from.
    n_samples : int
        The number of samples to generate
    n_features : int
        The number of features to generate
    density : float
        The density of the sparse matrix to generate
    """

    def select(value, choices, cast=None):
        if value is None:
            value = choices
        if isinstance(value, (list, tuple)):
            value = draw(st.sampled_from(value))
        if cast is not None:
            value = cast(value)
        assume(value in choices)
        return value

    kind = select(kind, ("cupy", "scipy", "scipy-array"))
    mem_type = "device" if kind == "cupy" else "host"

    DTYPES = ("f4", "f8")
    FORMATS = ("csr", "csc", "coo", "dia")
    if mem_type == "host":
        DTYPES += ("i4", "i8")
        FORMATS += ("bsr", "dok", "lil")

    dtype = select(dtype, DTYPES, cast=np.dtype)
    format = select(format, FORMATS)

    rng = np.random.default_rng(42)
    array = sp.random(
        n_samples,
        n_features,
        density=density,
        format=format,
        dtype=dtype,
        data_rvs=(
            (lambda n: rng.uniform(0, 100, size=n))
            if dtype.kind == "f"
            else (lambda n: rng.integers(0, 100, size=n))
        ),
        random_state=42,
    )
    if kind == "cupy":
        array = getattr(cp_sp, f"{format}_matrix")(array)
    elif kind == "scipy-array":
        array = getattr(sp, f"{format}_array")(array)

    return array


def as_cupy(array, dtype=None, order=None):
    """Coerce an array to cupy"""
    if isinstance(array, (cudf.Series, cudf.DataFrame)):
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        array = array.to_cupy()
    return cp.asarray(array, dtype=dtype, order=order)


def assert_contiguity(array, order):
    """Assert an array has the proper contiguity"""
    if order == "A":
        assert array.flags["C_CONTIGUOUS"] or array.flags["F_CONTIGUOUS"]
    elif order == "C":
        assert array.flags["C_CONTIGUOUS"]
    elif order == "F":
        assert array.flags["F_CONTIGUOUS"]


@contextmanager
def assert_no_warnings():
    """Small helper for asserting no warnings raised"""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


@pytest.mark.parametrize(
    "seed",
    [
        pytest.param(None, id="none"),
        pytest.param(42, id="int"),
        pytest.param(np.random.RandomState(42), id="numpy"),
        pytest.param(cp.random.RandomState(42), id="cupy"),
    ],
)
def test_check_random_seed(seed):
    res = check_random_seed(seed)
    assert isinstance(res, int)
    assert 0 <= res <= (2**32 - 1)  # in range for uint32
    if isinstance(seed, int):
        assert check_random_seed(seed) == res


def test_check_random_seed_errors():
    for bad in [-1, 2**32]:
        with pytest.raises(
            ValueError, match=r"Expected `0 <= random_state <= 2\*\*32 - 1`"
        ):
            check_random_seed(bad)
    for ok in [0, 2**32 - 1]:
        check_random_seed(ok)
    with pytest.raises(TypeError, match="`random_state` must be"):
        check_random_seed("incorrect type")


def test_get_n_features():
    class ArrayInterfaceOnly:
        def __init__(self, array):
            self.array = array

        @property
        def __array_interface__(self):
            return self.array.__array_interface__

        @property
        def __cuda_array_interface__(self):
            return self.array.__cuda_array_interface__

    x = np.ones((3, 2))
    cx = cp.ones((3, 2))

    assert _get_n_features(x) == 2
    assert _get_n_features(ArrayInterfaceOnly(x)) == 2
    assert _get_n_features(cx) == 2
    assert _get_n_features(ArrayInterfaceOnly(cx)) == 2

    assert _get_n_features([[1, 2, 3], [3, 4, 5]]) == 3
    assert _get_n_features([np.array([1, 2])]) == 2
    assert _get_n_features([]) == 0

    with pytest.raises(ValueError, match="Expected 2D array, got 3D array"):
        _get_n_features(np.ones((3, 2, 1)))


@pytest.mark.parametrize(
    "X",
    [
        [1, 2, 3],
        ["a", "b", "c"],
        [b"a", b"b", b"c"],
        [{"a": 1, "b": 2}, {"c": 3}],
        np.array([1, 2, 3]),
        cp.array([1, 2, 3]),
        pd.Series([1, 2, 3]),
        cudf.Series([1, 2, 3]),
    ],
)
def test_get_n_features_1D(X):
    if isinstance(X, (pd.Series, cudf.Series)):
        match = "Expected a 2-dimensional container"
    else:
        match = "Expected 2D array"

    with pytest.raises(ValueError, match=match):
        _get_n_features(X)


def test_get_feature_names():
    def assert_names(df, sol):
        names = _get_feature_names(df)
        if sol is None:
            assert names is None
        else:
            assert isinstance(names, np.ndarray)
            assert names.dtype == "object"
            assert (names == sol).all()

    class DataFrameInterfaceOnly:
        def __init__(self, df):
            self.df = df

        def __dataframe__(self):
            return self.df.__dataframe__()

    # Non-dataframe-like objects return None
    assert_names(None, None)
    assert_names([1, 2], None)
    assert_names(np.array([[1, 2], [3, 4]]), None)

    # 0-column frames return None
    empty = pd.DataFrame({})
    assert_names(empty, None)
    assert_names(cudf.DataFrame(empty), None)
    assert_names(DataFrameInterfaceOnly(empty), None)

    # str-names return object arrays
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert_names(df, ["a", "b"])
    assert_names(cudf.DataFrame(df), ["a", "b"])
    assert_names(DataFrameInterfaceOnly(df), ["a", "b"])

    # Fully non-str names are fully ignored
    df_int = pd.DataFrame({0: [1, 2], 1: [3, 4]})
    assert_names(df_int, None)

    # Mixed str & non-str names warn
    df_mixed = pd.DataFrame({"a": [1, 2], 1: [3, 4]})
    with pytest.raises(
        TypeError, match="all input features have string names"
    ):
        _get_feature_names(df_mixed)


class MyModel:
    def fit(self, X, y=None):
        check_features(self, X, reset=True)
        return self

    def predict(self, X):
        check_features(self, X)


def test_check_features_reset_true():
    X = np.ones((3, 2))
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    model = MyModel().fit(X)
    assert model.n_features_in_ == 2
    assert not hasattr(model, "feature_names_in_")

    model = MyModel().fit(df)
    assert model.n_features_in_ == 2
    assert (model.feature_names_in_ == ["a", "b"]).all()

    # refitting without feature names clears
    model.fit(X)
    assert model.n_features_in_ == 2
    assert not hasattr(model, "feature_names_in_")


def test_check_n_features_in():
    X = np.ones((3, 2))
    model = MyModel().fit(X)
    # no error
    model.predict(X)

    with pytest.raises(
        ValueError,
        match="X has 3 features, but MyModel is expecting 2 features as input",
    ):
        model.predict(np.ones((3, 3)))


def test_fit_and_predict_with_and_without_feature_names_warnings():
    X_unnamed = np.ones((3, 2))
    X_named = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    est_unnamed = MyModel().fit(X_unnamed)
    est_named = MyModel().fit(X_named)

    with assert_no_warnings():
        est_unnamed.predict(X_unnamed)

    with assert_no_warnings():
        est_named.predict(X_named)

    with pytest.warns(UserWarning, match="X has feature names"):
        est_unnamed.predict(X_named)

    with pytest.warns(
        UserWarning, match="X does not have valid feature names"
    ):
        est_named.predict(X_unnamed)


def test_feature_names_mismatch_errors():
    X = pd.DataFrame({"a": [1], "b": [2], "c": [3]})

    # Correct names are fine
    model = MyModel().fit(X)
    with assert_no_warnings():
        model.predict(X)

    # Missing column
    bad = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError, match="The feature names") as rec:
        model.predict(bad)
    assert "Feature names seen at fit time" in str(rec.value)

    # Extra column
    bad = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
    with pytest.raises(ValueError, match="The feature names") as rec:
        model.predict(bad)
    assert "Feature names unseen at fit time" in str(rec.value)

    # Reordered columns
    bad = pd.DataFrame({"a": [1], "c": [3], "b": [2]})
    with pytest.raises(ValueError, match="The feature names") as rec:
        model.predict(bad)
    assert "Feature names must be in the same order" in str(rec.value)


def test_check_consistent_length():
    y3 = np.empty(3)
    y4 = np.empty(4)
    x34 = np.empty((3, 4))

    check_consistent_length()
    check_consistent_length(None)
    check_consistent_length(x34)
    check_consistent_length(x34, y3)
    check_consistent_length(x34, None, y3)

    # Supports non-normalized inputs too
    check_consistent_length(x34.tolist(), y3, None)
    check_consistent_length(pd.DataFrame(x34), cudf.Series(y3), None)

    with pytest.raises(
        ValueError,
        match=r"Found input variables with inconsistent number of samples: \[3, 4\]",
    ):
        check_consistent_length(x34, y4)

    with pytest.raises(
        TypeError, match="Expected array-like, got scalar instead"
    ):
        check_consistent_length(x34, None, np.array(1.5))

    with pytest.raises(
        TypeError, match=re.escape(f"Expected array-like, got {int} instead")
    ):
        check_consistent_length(x34, None, 1)


@pytest.mark.parametrize("device", [True, False])
@pytest.mark.parametrize("sparse_format", [None, "csr", "coo"])
def test_check_all_finite(device, sparse_format):
    def array(values, dtype=None):
        x = cp.array(values, dtype=dtype)
        if sparse_format is not None:
            x = getattr(cp_sp, f"{sparse_format}_matrix")(x)
        if not device:
            x = x.get()
        return x

    non_floating = array([True, False, True], dtype="bool")
    f32_empty = array([], dtype="float32")
    f32_good = array([1.5, -1.5, 2.5], dtype="float32")
    f32_nan = array([1.5, float("nan"), 2.5], dtype="float32")
    f32_inf = array([1.5, float("inf"), 2.5], dtype="float32")
    f64_both = array([[1.5, float("inf"), float("nan")]], dtype="float64")

    check_all_finite(non_floating)
    check_all_finite(f32_empty)
    check_all_finite(f32_good, allow_nan=False)
    check_all_finite(f32_good, allow_nan=True)
    check_all_finite(f32_nan, allow_nan=True)

    with pytest.raises(
        ValueError, match="Input X contains NaN or infinite values"
    ):
        check_all_finite(f32_nan, allow_nan=False, input_name="X")

    with pytest.raises(
        ValueError, match="Input array contains infinite values"
    ):
        check_all_finite(f32_inf, allow_nan=True)

    with pytest.raises(
        ValueError, match="Input array contains NaN or infinite values"
    ):
        check_all_finite(f64_both)


def test_check_all_finite_host_fallback():
    x_good = np.array([1e307] * 100, dtype="float64")
    x_nan = np.array([1e307] * 99 + [float("nan")], dtype="float64")
    x_inf = np.array([1e307] * 99 + [float("inf")], dtype="float64")

    check_all_finite(x_good)
    check_all_finite(x_good, allow_nan=True)
    check_all_finite(x_nan, allow_nan=True)

    with pytest.raises(
        ValueError, match="Input array contains NaN or infinite values"
    ):
        check_all_finite(x_nan)

    with pytest.raises(
        ValueError, match="Input array contains infinite values"
    ):
        check_all_finite(x_inf, allow_nan=True)


def test_check_all_finite_assume_finite():
    bad = cp.array([1.5, float("nan"), 2.5], dtype="float32")
    # No errors for bad inputs if `assume_finite=True` configured
    with sklearn.config_context(assume_finite=True):
        check_all_finite(bad)


@pytest.mark.parametrize("device", [True, False])
@pytest.mark.parametrize("sparse_format", [None, "csr", "coo"])
def test_check_non_negative(device, sparse_format):
    def array(values, dtype=None):
        x = cp.array(values, dtype=dtype)
        if sparse_format is not None:
            x = getattr(cp_sp, f"{sparse_format}_matrix")(x)
        if not device:
            x = x.get()
        return x

    f32_empty = array([], dtype="float32")
    f32_good = array([0, 1, 2], dtype="float32")
    f64_good_nan = array([0, float("nan"), 1], dtype="float64")
    f32_bad = array([-1, 1, 2], dtype="float32")
    f64_bad_nan = array([-1, float("nan"), 1], dtype="float64")

    check_non_negative(f32_empty)
    check_non_negative(f32_good)
    check_non_negative(f64_good_nan)

    with pytest.raises(ValueError, match="Negative values in data"):
        check_non_negative(f32_bad)

    with pytest.raises(
        ValueError, match="Negative values in data passed to X"
    ):
        check_non_negative(f64_bad_nan, input_name="X")


def test_check_array_bad_args():
    with pytest.raises(ValueError, match="Unsupported mem_type='bad'"):
        check_array([1, 2, 3], mem_type="bad")

    with pytest.raises(ValueError, match="Unsupported order='bad'"):
        check_array([1, 2, 3], order="bad")

    # Exception raised by `np.dtype`, we don't really care what it is
    with pytest.raises(Exception, match="'bad'"):
        check_array([1, 2, 3], dtype="bad")
    with pytest.raises(Exception, match="'bad'"):
        check_array([1, 2, 3], dtype=("int32", "bad"))


@pytest.mark.parametrize(
    "func",
    [
        pytest.param(lambda x: x, id="list"),
        pytest.param(np.array, id="numpy"),
        pytest.param(cp.array, id="cupy"),
        pytest.param(
            lambda x: cp_sp.csr_matrix(cp.asarray(x)), id="cupyx.sparse"
        ),
        pytest.param(
            lambda x: sp.csr_matrix(np.asarray(x)), id="scipy.sparse"
        ),
    ],
)
def test_check_array_complex_errors(func):
    array = func([[complex(1), complex(2, 3)]])
    with pytest.raises(ValueError, match="Complex data not supported"):
        check_array(array)


@example(array=np.asarray([1, 2, 3]), mem_type=None)
@example(array=np.asarray([[1, 2], [3, 4]]), mem_type="device")
@example(array=cp.asarray([[1, 2], [3, 4]]), mem_type="host")
@given(
    array=dense_arrays(dtype="float32", order="C"),
    mem_type=st.sampled_from(["device", "host", None]),
)
def test_check_array_mem_type(array, mem_type):
    out = check_array(array, mem_type=mem_type, ensure_2d=False)
    if mem_type is None:
        cls = (
            cp.ndarray
            if isinstance(array, (cp.ndarray, cudf.DataFrame, cudf.Series))
            else np.ndarray
        )
    else:
        cls = cp.ndarray if mem_type == "device" else np.ndarray

    assert isinstance(out, cls)
    cp.testing.assert_allclose(as_cupy(array), as_cupy(out))


@example(
    array=cp.asarray([[1, 2], [3, 4]], order="C"),
    order="F",
    mem_type="device",
)
@example(
    array=cp.asarray([[1, 2], [3, 4]], order="F"),
    order="C",
    mem_type="device",
)
@example(
    array=np.array([[1, 2], [3, 4], [5, 6]])[::2],
    order="A",
    mem_type="host",
)
@given(
    array=dense_arrays(dtype="float32"),
    order=st.sampled_from(["C", "F", "A"]),
    mem_type=st.sampled_from(["device", "host", None]),
)
def test_check_array_order(array, order, mem_type):
    out = check_array(array, ensure_2d=False, order=order, mem_type=mem_type)
    cp.testing.assert_allclose(cp.asarray(out), as_cupy(array))
    assert_contiguity(out, order)


@example(array=[[1, 2]], mem_type="device")
@example(array=cp.array([[1, 2]]), mem_type="host")
@example(array=np.array([[1, 2]]), mem_type="device")
@example(array=cudf.DataFrame([[1, 2]]), mem_type="device")
@example(array=pd.DataFrame([[1, 2]]), mem_type="device")
@given(
    array=dense_arrays(ndim=2, dtype="int32"),
    mem_type=st.sampled_from(["device", "host", None]),
)
def test_check_array_dtype(array, mem_type):
    """Generate arrays with definitive different dtype, and ensure
    they're all cast appropriately. Checks that `dtype` is passed properly
    in all code paths."""

    # By default the input dtype is kept the same
    out = check_array(array, dtype=None, mem_type=mem_type)
    if hasattr(array, "dtype"):
        assert out.dtype == array.dtype

        out = check_array(array, dtype=array.dtype, mem_type=mem_type)
        assert out.dtype == array.dtype

        # If a sequence, no coercion done if dtype already valid
        out = check_array(
            array, dtype=("float32", array.dtype), mem_type=mem_type
        )
        assert out.dtype == array.dtype

    # Coercion to a specific dtype
    out = check_array(array, dtype="float32", mem_type=mem_type)
    assert out.dtype == "float32"
    cp.testing.assert_allclose(cp.asarray(out), as_cupy(array, "float32"))

    out = check_array(array, dtype="float64", mem_type=mem_type)
    assert out.dtype == "float64"
    cp.testing.assert_allclose(cp.asarray(out), as_cupy(array, "float64"))

    # If a sequence, the first dtype is used when coercion needed
    out = check_array(array, dtype=("float32", "float64"), mem_type=mem_type)
    assert out.dtype == "float32"


def test_check_array_convert_dtype():
    array = cp.array([[1, 2, 3]], dtype="float32")

    with pytest.raises(
        ValueError,
        match=r"Expected array with dtype in \['int32'\] but got 'float32'",
    ):
        check_array(array, dtype="int32", convert_dtype=False)

    array = pd.DataFrame({"x": [1, 2, 3], "y": [1.5, 2.5, 3.5]})
    with pytest.raises(ValueError, match=r"\['int32'\] but got 'float64'"):
        check_array(array, dtype="int32", convert_dtype=False)

    array = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "a"]})
    with pytest.raises(ValueError, match=r"\['int32'\] but got 'object'"):
        check_array(array, dtype="int32", convert_dtype=False)


@pytest.mark.parametrize("kind", ["cudf", "pandas"])
@pytest.mark.parametrize("mem_type", ["device", "host", None])
def test_check_array_dataframe_mixed_dtypes(kind, mem_type):
    xdf = cudf if kind == "cudf" else pd

    df = xdf.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [2.5, 3.5, 4.5, 5.5, 6.5],
            "z": ["1", "2", "3.5", "4", "5"],
        }
    )
    # Non-numeric columns -> object dtype by default
    if mem_type == "device" or mem_type is None and kind == "cudf":
        # cupy doesn't support object dtypes
        with pytest.raises((ValueError, TypeError), match="object"):
            check_array(df, mem_type=mem_type)
    else:
        # dtype=None does no conversion by default
        out = check_array(df, mem_type=mem_type)
        assert out.dtype == "object"

    # Can coerce all columns to specified dtype
    out = check_array(df, mem_type=mem_type, dtype=("float32", "float64"))
    assert out.dtype == "float32"
    np.testing.assert_allclose(cp.asnumpy(out), df.to_numpy(dtype="float32"))

    # Subset of numeric columns -> numeric by default
    df2 = df[["x", "y"]]
    out = check_array(df2, mem_type=mem_type)
    assert out.dtype == df2.y.dtype
    np.testing.assert_allclose(cp.asnumpy(out), df2.to_numpy())


@pytest.mark.parametrize("kind", ["cudf", "pandas"])
@pytest.mark.parametrize("mem_type", ["device", "host", None])
def test_check_array_dataframe_non_numpy_dtype(kind, mem_type):
    xdf = cudf if kind == "cudf" else pd

    df = xdf.DataFrame({"x": ["1", "2", "1", "3"]}).astype("category")
    # Can coerce all columns to specified dtype
    out = check_array(df, mem_type=mem_type, dtype=("float32", "float64"))
    assert out.dtype == "float32"
    np.testing.assert_allclose(cp.asnumpy(out), df.to_numpy(dtype="float32"))


@pytest.mark.parametrize("kind", ["cudf", "pandas"])
@pytest.mark.parametrize("ndim", [1, 2])
@pytest.mark.parametrize("mem_type", ["device", "host", None])
def test_check_array_return_index(kind, ndim, mem_type):
    xdf = cudf if kind == "cudf" else pd
    if ndim == 2:
        array = xdf.DataFrame({"x": [1, 2, 3]}, index=[1, 3, 5])
    else:
        array = xdf.Series([1, 2, 3], index=[1, 3, 5])

    out, index = check_array(
        array, ensure_2d=False, return_index=True, mem_type=mem_type
    )
    if mem_type is None:
        out_xdf = xdf
    else:
        out_xdf = cudf if mem_type == "device" else pd
    assert isinstance(index, out_xdf.Index)
    assert (cp.asnumpy(index) == np.array([1, 3, 5])).all()


@pytest.mark.parametrize("kind", ["numpy", "cudf", "pandas"])
@pytest.mark.parametrize("mem_type", ["device", "host", None])
def test_check_array_object_dtype(kind, mem_type):
    array = np.array(["1.5", "2.5", "3.5"], dtype="object")
    if kind == "cudf":
        array = cudf.Series(array)
    elif kind == "pandas":
        array = pd.Series(array)

    if mem_type == "device" or mem_type is None and kind == "cudf":
        # cupy doesn't support object dtypes
        with pytest.raises((ValueError, TypeError), match="object"):
            check_array(array, mem_type=mem_type, ensure_2d=False)
    else:
        out = check_array(array, mem_type=mem_type, ensure_2d=False)
        assert out.dtype == "object"

    # Can coerce to numeric if specified
    out = check_array(
        array, mem_type=mem_type, dtype=("float32", "float64"), ensure_2d=False
    )
    assert out.dtype == "float32"
    cp.testing.assert_allclose(cp.asarray(out), as_cupy(array, "float32"))


@pytest.mark.parametrize(
    "array",
    [
        pytest.param(cp_sp.csr_matrix(cp.array([[1.0, 0]])), id="cupy matrix"),
        pytest.param(sp.csr_matrix(np.array([[1.0, 0]])), id="scipy matrix"),
        pytest.param(sp.csr_array(np.array([[1.0, 0]])), id="scipy array"),
    ],
)
def test_check_array_sparse_not_supported(array):
    with pytest.raises(
        TypeError, match="Sparse data was passed, but dense data is required"
    ):
        check_array(array)

    with pytest.raises(
        TypeError,
        match="Sparse data was passed for X, but dense data is required",
    ):
        check_array(array, input_name="X")


@example(array=cp_sp.csr_matrix(cp.array([[1.0, 0], [0, 0]])), mem_type="host")
@example(array=sp.csr_matrix(np.array([[1.0, 0], [0, 0]])), mem_type="device")
@given(
    array=sparse_arrays(),
    mem_type=st.sampled_from(["device", "host", None]),
)
def test_check_array_sparse_input(array, mem_type):
    if mem_type is None:
        ns = cp_sp if cp_sp.issparse(array) else sp
    else:
        ns = cp_sp if mem_type == "device" else sp

    # dtype=None case
    if (
        mem_type == "device"
        and array.dtype.kind != "f"
        and array.format != "dia"
    ):
        # cupy only supports floating dtypes for these inputs. We let cupy
        # itself raise an exception, we don't really care what it is.
        with pytest.raises(ValueError, match="float32"):
            check_array(array, accept_sparse=True, mem_type=mem_type)
    else:
        out = check_array(array, accept_sparse=True, mem_type=mem_type)
        assert ns.issparse(out)
        assert out.dtype == array.dtype

    # Coerce to specified dtypes
    out = check_array(
        array, accept_sparse=True, dtype="float32", mem_type=mem_type
    )
    assert ns.issparse(out)
    assert out.dtype == "float32"


@example(
    array=cp_sp.csr_matrix(cp.array([[1.0, 0], [0, 0]])),
    mem_type="host",
    format="coo",
)
@example(
    array=sp.csr_matrix(np.array([[1.0, 0], [0, 0]])),
    mem_type="device",
    format="coo",
)
@given(
    array=sparse_arrays(dtype="float32"),
    mem_type=st.sampled_from(["device", "host", None]),
    format=st.sampled_from(["csr", "csc", "coo"]),
)
def test_check_array_sparse_input_format(array, mem_type, format):
    if mem_type is None:
        ns = cp_sp if cp_sp.issparse(array) else sp
    else:
        ns = cp_sp if mem_type == "device" else sp

    out = check_array(array, accept_sparse=True, mem_type=mem_type)
    assert ns.issparse(out)
    assert out.dtype == array.dtype
    # Format unchanged unless not a format cupy supports
    if array.format in ["csr", "coo", "csc", "dia"]:
        assert out.format == array.format
    else:
        assert out.format == "csr"

    # Coerce to specified formats
    out = check_array(array, accept_sparse=format, mem_type=mem_type)
    assert ns.issparse(out)
    assert out.dtype == array.dtype
    assert out.format == format

    out = check_array(array, accept_sparse=[format, "csc"], mem_type=mem_type)
    assert ns.issparse(out)
    assert out.dtype == array.dtype
    assert out.format == (
        array.format if array.format in [format, "csc"] else format
    )


@pytest.mark.parametrize("format", ["csr", "csc", "coo", "bsr"])
def test_check_array_accept_large_sparse(format):
    array = sp.random(20, 10, density=0.5, format=format, random_state=42)
    if array.format == "coo":
        array.coords = tuple(v.astype("int64") for v in array.coords)
    else:
        for name in ["indices", "indptr"]:
            setattr(array, name, getattr(array, name).astype("int64"))

    with pytest.raises(ValueError, match="sparse matrices with int32 indices"):
        check_array(array, accept_sparse=True)

    check_array(
        array, accept_sparse=True, accept_large_sparse=True, mem_type="host"
    )


@example(array=np.ones((3, 2)))
@example(array=cp_sp.csr_matrix(cp.ones((3, 2))))
@given(
    array=st.one_of(
        dense_arrays(dtype="float32", ndim=2, n_samples=3, n_features=2),
        sparse_arrays(dtype="float32", n_samples=3, n_features=2),
    )
)
def test_check_array_ensure_min_samples_and_ensure_min_features(array):
    with pytest.raises(
        ValueError,
        match=(
            r"Found array with 3 sample\(s\) \(shape=\(3, 2\)\) while a "
            "minimum of 6 is required"
        ),
    ):
        check_array(array, ensure_min_samples=6, accept_sparse=True)

    with pytest.raises(
        ValueError,
        match=(
            r"Found array with 2 feature\(s\) \(shape=\(3, 2\)\) while a "
            "minimum of 5 is required"
        ),
    ):
        check_array(array, ensure_min_features=5, accept_sparse=True)


@pytest.mark.parametrize(
    "func",
    [
        pytest.param(lambda x: x, id="list"),
        pytest.param(np.asarray, id="numpy"),
        pytest.param(cp.asarray, id="cupy"),
        pytest.param(cudf.Series, id="cudf"),
        pytest.param(pd.Series, id="pandas"),
        pytest.param(sp.csr_array, id="scipy.sparse.csr_array"),
    ],
)
def test_check_array_ensure_2d(func):
    array = func([1.5, 2.5, 3.5])
    if isinstance(array, (pd.Series, cudf.Series)):
        err_msg = "Expected a 2-dimensional container"
    else:
        err_msg = "Expected 2D array, got 1D array instead"
    with pytest.raises(ValueError, match=err_msg):
        check_array(array, accept_sparse=True)


def test_check_array_ensure_all_finite():
    """Tests plumbing of check_array -> check_all_finite"""
    f32_nan = np.array([[1.5, float("nan"), 2.5]], dtype="float32")
    f64_both = np.array([[1.5, float("inf"), float("nan")]], dtype="float64")

    # No errors
    check_array(f32_nan, ensure_all_finite="allow-nan")
    check_array(f64_both, ensure_all_finite=False)

    with pytest.raises(
        ValueError, match="Input X contains NaN or infinite values"
    ):
        check_array(f32_nan, input_name="X")

    with pytest.raises(
        ValueError, match="Input array contains NaN or infinite values"
    ):
        check_array(f64_both)


def test_check_array_ensure_non_negative():
    """Tests plumbing of check_array -> check_non_negative"""
    array = np.array([[-1, 1, 2]], dtype="float32")

    # No error, check disabled by default
    check_array(array)

    with pytest.raises(
        ValueError, match="Negative values in data passed to X"
    ):
        check_array(array, input_name="X", ensure_non_negative=True)


@example(y=cp.asarray([[1, 2], [3, 4]], order="F"), order="C", mem_type="host")
@example(
    y=np.asarray([[1, 2], [3, 4]], order="C"), order="F", mem_type="device"
)
@given(
    y=dense_arrays(ndim=(1, 2), dtype="int64"),
    order=st.sampled_from(["C", "F", "A"]),
    mem_type=st.sampled_from(["device", "host", None]),
)
def test_check_y(y, mem_type, order):
    if mem_type == "device":
        exp_type = cp.ndarray
    elif mem_type == "host":
        exp_type = np.ndarray
    else:
        exp_type = (
            cp.ndarray
            if isinstance(y, (cp.ndarray, cudf.DataFrame, cudf.Series))
            else np.ndarray
        )

    out = check_y(y, mem_type=mem_type, order=order, accept_multi_output=True)
    if hasattr(y, "dtype"):
        assert out.dtype == y.dtype
    assert isinstance(out, exp_type)
    assert_contiguity(out, order)
    cp.testing.assert_allclose(as_cupy(out), as_cupy(y))

    # Check that if dtype specified it's properly converted
    out = check_y(
        y,
        mem_type=mem_type,
        order=order,
        accept_multi_output=True,
        dtype="float32",
    )
    assert out.dtype == "float32"
    assert isinstance(out, exp_type)
    assert_contiguity(out, order)
    cp.testing.assert_allclose(as_cupy(out), as_cupy(y, "float32"))

    # Sequence of dtypes also works
    out = check_y(
        y,
        accept_multi_output=True,
        dtype=("int32", "int64"),
    )
    assert out.dtype in ("int32", "int64")


@pytest.mark.parametrize("kind", ["array", "dataframe"])
def test_check_y_accept_multi_output(kind):
    if kind == "array":
        y_2d_1col = np.array([1, 2, 3])[:, None]
        y_2d = np.array([[1, 2, 3], [4, 5, 6]])
    else:
        y_2d_1col = pd.DataFrame({"x": [1, 2, 3]})
        y_2d = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    # 2d with 1 column warns, but still returns 2D
    with pytest.warns(DataConversionWarning, match="A column-vector y"):
        out = check_y(y_2d_1col)
    assert out.ndim == 2

    # 2d with multiple columns just errors
    with pytest.raises(ValueError, match="y should be a 1d"):
        check_y(y_2d)

    # With accept_multi_output=True, no cases error or warn
    out = check_y(y_2d_1col, accept_multi_output=True)
    assert out.ndim == 2
    out = check_y(y_2d, accept_multi_output=True)
    assert out.ndim == 2


@example(
    kind="cupy",
    label_dtype="int32",
    n_classes=2,
    mem_type="host",
    dtype=None,
    order=None,
)
@example(
    kind="pandas",
    label_dtype="O",
    n_classes=(2, 4),
    mem_type="device",
    dtype="int32",
    order="F",
)
@example(
    kind="list",
    label_dtype="O",
    n_classes=2,
    mem_type=None,
    dtype=None,
    order="A",
)
@given(
    kind=st.sampled_from(["cupy", "numpy", "pandas", "cudf", "list"]),
    label_dtype=st.sampled_from(["int32", "int64", "bool", "O", "U"]),
    n_classes=st.sampled_from([2, 4, (3,), (2, 4)]),
    mem_type=st.sampled_from(["device", "host", None]),
    dtype=st.sampled_from([None, "int32", "int64", "float32"]),
    order=st.sampled_from(["C", "F", "A", None]),
)
def test_check_y_return_classes(
    kind, label_dtype, n_classes, mem_type, dtype, order
):
    # Construct input data
    if label_dtype in ("O", "U"):
        if mem_type is None:
            assume(kind not in ("cudf", "cupy"))
        else:
            assume(not (kind == "cupy" or mem_type == "device"))
        labels = np.array(["a", "b", "c", "d"], dtype=label_dtype)
    elif label_dtype == "bool":
        labels = np.array([True, False])
    else:
        labels = np.array([5, 10, 15, 20], dtype=label_dtype)

    rng = np.random.default_rng(42)
    if isinstance(n_classes, int):
        assume(n_classes < len(labels))
        inds = rng.integers(n_classes, size=100)
    else:
        assume(all(n < len(labels) for n in n_classes))
        inds = np.stack([rng.integers(n, size=100) for n in n_classes]).T

    y = labels.take(inds)
    if kind == "cupy":
        y = cp.asarray(y)
    elif kind == "pandas":
        y = pd.DataFrame(y) if inds.ndim == 2 else pd.Series(y)
    elif kind == "cudf":
        y = cudf.DataFrame(y) if inds.ndim == 2 else cudf.Series(y)
    elif kind == "list":
        y = y.tolist()

    # Construct the expected outputs
    sol = inds.flatten() if inds.ndim == 2 and inds.shape[1] == 1 else inds
    if dtype is not None:
        sol = sol.astype(dtype)
    if mem_type == "device" or mem_type is None and kind in ("cupy", "cudf"):
        sol = cp.asarray(sol)

    y2 = check_array(y, mem_type="host", ensure_2d=False)
    if inds.ndim == 2 and inds.shape[1] != 1:
        sol_classes = [np.unique(y2[:, i]) for i in range(y2.shape[1])]
    else:
        sol_classes = np.unique(y2)

    # Helper function so we don't have to repeat args below
    def check(accept_multi_output=False):
        return check_y(
            y,
            order=order,
            dtype=dtype,
            mem_type=mem_type,
            accept_multi_output=accept_multi_output,
            return_classes=True,
        )

    # Check the calls warn/raise appropriately and return expected types
    if inds.ndim == 2:
        if inds.shape[1] == 1:
            # For classifiers this always warns, even if multi-output accepted
            with pytest.warns(
                DataConversionWarning, match="A column-vector y"
            ):
                out, classes = check()
            assert out.ndim == 1
            assert isinstance(classes, np.ndarray)
            with pytest.warns(
                DataConversionWarning, match="A column-vector y"
            ):
                out, classes = check(True)
        else:
            with pytest.raises(ValueError, match="y should be a 1d"):
                check()
            out, classes = check(True)
    else:
        out, classes = check()

    # Assert encoded y is correct
    assert_contiguity(out, order)
    if dtype is None:
        assert out.dtype.kind in "iu"  # default to some integral type
    else:
        assert out.dtype == dtype
    assert out.shape == sol.shape
    assert isinstance(out, type(sol))
    np.testing.assert_allclose(cp.asnumpy(out), cp.asnumpy(sol))

    # Assert classes are correct
    if isinstance(sol_classes, list):
        assert isinstance(classes, list)
        assert len(classes) == len(sol_classes)
        for c, s in zip(classes, sol_classes):
            assert c.dtype == s.dtype
            assert (c == s).all()
    else:
        assert classes.dtype == sol_classes.dtype
        assert (classes == sol_classes).all()


def test_check_y_classifier_on_floating_input():
    # integral floating values are accepted
    good = np.array([1.0, 2.0, 1.0])
    out, classes = check_y(good, return_classes=True)
    assert classes.dtype == good.dtype
    np.testing.assert_array_equal(classes, np.unique(good))
    np.testing.assert_array_equal(cp.asnumpy(out), np.array([0, 1, 0]))

    # Non integral values error
    bad = [
        np.array([1.5, 2.5, 3.5]),
        cp.array([1.0, float("nan"), 3.0]),
        np.array([1.0, float("inf"), 3.0]),
    ]
    for array in bad:
        with pytest.raises(ValueError, match="Unknown label type: continuous"):
            check_y(array, return_classes=True)


def test_check_y_none():
    with pytest.raises(ValueError, match="This estimator requires y"):
        check_y(None)
