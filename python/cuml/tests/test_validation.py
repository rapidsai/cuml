# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import warnings
from contextlib import contextmanager

import cudf
import cupy as cp
import numpy as np
import pandas as pd
import pytest

from cuml.internals.validation import (
    _get_feature_names,
    _get_n_features,
    check_features,
    check_random_seed,
)


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
    with pytest.warns(FutureWarning, match="non-2-dimensional"):
        n_features = _get_n_features(X)
    assert n_features == 1


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
    with pytest.warns(FutureWarning, match="feature_names_in_") as rec:
        _get_feature_names(df_mixed)
    assert "all input features have string names" in str(rec[0].message)


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


def test_feature_names_mismatch_warnings():
    X = pd.DataFrame({"a": [1], "b": [2], "c": [3]})

    # Correct names are fine
    model = MyModel().fit(X)
    with assert_no_warnings():
        model.predict(X)

    # Missing column
    bad = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.warns(FutureWarning, match="feature_names_in_") as rec:
        # TODO: in 26.06 the FutureWarning will become an error,
        # and this raises check can go away.
        with pytest.raises(ValueError, match="X has 2 features"):
            model.predict(bad)
    assert "Feature names seen at fit time" in str(rec[0].message)

    # Extra column
    bad = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
    with pytest.warns(FutureWarning, match="feature_names_in_") as rec:
        # TODO: in 26.06 the FutureWarning will become an error,
        # and this raises check can go away.
        with pytest.raises(ValueError, match="X has 4 features"):
            model.predict(bad)
    assert "Feature names unseen at fit time" in str(rec[0].message)

    # Reordered columns
    bad = pd.DataFrame({"a": [1], "c": [3], "b": [2]})
    with pytest.warns(FutureWarning, match="feature_names_in_") as rec:
        model.predict(bad)
    assert "Feature names must be in the same order" in str(rec[0].message)
