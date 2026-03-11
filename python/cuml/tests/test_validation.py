# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import cupy as cp
import numpy as np
import pytest

from cuml.internals.validation import (
    _get_n_features,
    check_features,
    check_random_seed,
)


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
    assert _get_n_features([1, 2, 3]) == 1
    assert _get_n_features(["a", "b", "c"]) == 1
    assert _get_n_features([b"a", b"b", b"c"]) == 1
    assert _get_n_features([{"a": 1, "b": 2}, {"c": 3}]) == 1


def test_check_features_n_features_in():
    class MyModel:
        def fit(self, X, y=None):
            check_features(self, X, reset=True)
            return self

        def predict(self, X):
            check_features(self, X)

    X = np.ones((3, 2))
    model = MyModel().fit(X)
    assert model.n_features_in_ == 2
    model.predict(X)

    with pytest.raises(
        ValueError,
        match="X has 3 features, but MyModel is expecting 2 features as input",
    ):
        model.predict(np.ones((3, 3)))
