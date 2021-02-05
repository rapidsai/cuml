# Copyright (c) 2019, NVIDIA CORPORATION.
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

import cudf
import cupy as cp
import numpy as np
import pytest

from cuml.model_selection import train_test_split
from cuml.datasets import make_classification
from numba import cuda

test_array_input_types = [
    'numba', 'cupy'
]

test_seeds = [
    'int', 'cupy', 'numpy'
]


@pytest.mark.parametrize("train_size", [0.2, 0.6, 0.8])
@pytest.mark.parametrize("shuffle", [True, False])
def test_split_dataframe(train_size, shuffle):
    X = cudf.DataFrame({"x": range(100)})
    y = cudf.Series(([0] * (100 // 2)) + ([1] * (100 // 2)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, shuffle=shuffle
    )
    assert len(X_train) == len(y_train) == pytest.approx(train_size * len(X))
    assert (
        len(X_test) == len(y_test) == pytest.approx((1 - train_size) * len(X))
    )
    assert (all(X_train.index.to_pandas() == y_train.index.to_pandas()))
    assert (all(X_test.index.to_pandas() == y_test.index.to_pandas()))

    X_reconstructed = cudf.concat([X_train, X_test]).sort_values(
        by=["x"]
    )
    y_reconstructed = y_train.append(y_test).sort_values()

    assert all(X_reconstructed.reset_index(drop=True) == X)
    out = y_reconstructed.reset_index(drop=True).values_host == y.values_host
    assert all(out)


def test_split_column():
    data = cudf.DataFrame(
        {
            "x": range(100),
            "y": ([0] * (100 // 2)) + ([1] * (100 // 2)),
        }
    )
    train_size = 0.8

    X_train, X_test, y_train, y_test = train_test_split(
        data, "y", train_size=train_size
    )

    assert (
        len(X_train) == len(y_train) == pytest.approx(train_size * len(data))
    )
    assert (
        len(X_test)
        == len(y_test)
        == pytest.approx((1 - train_size) * len(data))
    )

    X_reconstructed = cudf.concat([X_train, X_test]).sort_values(
        by=["x"]
    )
    y_reconstructed = y_train.append(y_test).sort_values()

    assert all(
        data == X_reconstructed.assign(
            y=y_reconstructed).reset_index(drop=True)
    )


def test_split_size_mismatch():
    X = cudf.DataFrame({'x': range(3)})
    y = cudf.Series([0, 1])

    with pytest.raises(ValueError):
        train_test_split(X, y)


@pytest.mark.parametrize('train_size', [1.2, 100])
def test_split_invalid_proportion(train_size):
    X = cudf.DataFrame({'x': range(10)})
    y = cudf.Series([0] * 10)

    with pytest.raises(ValueError):
        train_test_split(X, y, train_size=train_size)


@pytest.mark.parametrize('seed_type', test_seeds)
def test_random_state(seed_type):
    for i in range(10):
        seed_n = np.random.randint(0, int(1e9))
        if seed_type == 'int':
            seed = seed_n
        if seed_type == 'cupy':
            seed = cp.random.RandomState(seed=seed_n)
        if seed_type == 'numpy':
            seed = np.random.RandomState(seed=seed_n)
        X = cudf.DataFrame({"x": range(100)})
        y = cudf.Series(([0] * (100 // 2)) + ([1] * (100 // 2)))

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=seed)

        if seed_type == 'cupy':
            seed = cp.random.RandomState(seed=seed_n)
        if seed_type == 'numpy':
            seed = np.random.RandomState(seed=seed_n)

        X_train2, X_test2, y_train2, y_test2 = \
            train_test_split(X, y, random_state=seed)

        assert X_train.equals(X_train2)
        assert X_test.equals(X_test2)
        assert y_train.equals(y_train2)
        assert y_test.equals(y_test2)


@pytest.mark.parametrize('type', test_array_input_types)
@pytest.mark.parametrize('test_size', [0.2, 0.4, None])
@pytest.mark.parametrize('train_size', [0.6, 0.8, None])
@pytest.mark.parametrize('shuffle', [True, False])
def test_array_split(type, test_size, train_size, shuffle):
    X = np.zeros((100, 10)) + np.arange(100).reshape(100, 1)
    y = np.arange(100).reshape(100, 1)

    if type == 'cupy':
        X = cp.asarray(X)
        y = cp.asarray(y)

    if type == 'numba':
        X = cuda.to_device(X)
        y = cuda.to_device(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=train_size,
                                                        test_size=test_size,
                                                        shuffle=shuffle,
                                                        random_state=0)

    if type == 'cupy':
        assert isinstance(X_train, cp.ndarray)
        assert isinstance(X_test, cp.ndarray)
        assert isinstance(y_train, cp.ndarray)
        assert isinstance(y_test, cp.ndarray)

    if type in ['numba', 'rmm']:
        assert cuda.devicearray.is_cuda_ndarray(X_train)
        assert cuda.devicearray.is_cuda_ndarray(X_test)
        assert cuda.devicearray.is_cuda_ndarray(y_train)
        assert cuda.devicearray.is_cuda_ndarray(y_test)

    if train_size is not None:
        assert X_train.shape[0] == X.shape[0] * train_size
        assert y_train.shape[0] == y.shape[0] * train_size

    if test_size is not None:
        assert X_test.shape[0] == X.shape[0] * test_size
        assert y_test.shape[0] == y.shape[0] * test_size

    if shuffle is None:
        assert X_train == X[0:train_size]
        assert y_train == y[0:train_size]
        assert X_test == X[-1 * test_size:]
        assert y_test == y[-1 * test_size:]

        X_rec = cp.sort(cp.concatenate(X_train, X_test))
        y_rec = cp.sort(cp.concatenate(y_train, y_test))

        assert X_rec == X
        assert y_rec == y


def test_default_values():
    X = np.zeros((100, 10)) + np.arange(100).reshape(100, 1)
    y = np.arange(100).reshape(100, 1)

    X = cp.asarray(X)
    y = cp.asarray(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    assert isinstance(X_train, cp.ndarray)
    assert isinstance(X_test, cp.ndarray)
    assert isinstance(y_train, cp.ndarray)
    assert isinstance(y_test, cp.ndarray)

    assert X_train.shape[0] == X.shape[0] * 0.75
    assert y_train.shape[0] == y.shape[0] * 0.75

    assert X_test.shape[0] == X.shape[0] * 0.25
    assert y_test.shape[0] == y.shape[0] * 0.25


@pytest.mark.parametrize('test_size', [0.2, 0.4, None])
@pytest.mark.parametrize('train_size', [0.6, 0.8, None])
@pytest.mark.parametrize('shuffle', [True, False])
def test_split_df_single_argument(test_size, train_size, shuffle):
    X = cudf.DataFrame({'x': range(50)})
    X_train, X_test = train_test_split(X,
                                       train_size=train_size,
                                       test_size=test_size,
                                       shuffle=shuffle,
                                       random_state=0)
    if train_size is not None:
        assert X_train.shape[0] == (int)(X.shape[0] * train_size)

    if test_size is not None:
        assert X_test.shape[0] == (int)(X.shape[0] * test_size)


@pytest.mark.parametrize('type', test_array_input_types)
@pytest.mark.parametrize('test_size', [0.2, 0.4, None])
@pytest.mark.parametrize('train_size', [0.6, 0.8, None])
@pytest.mark.parametrize('shuffle', [True, False])
def test_split_array_single_argument(type, test_size, train_size, shuffle):
    X = np.zeros((100, 10)) + np.arange(100).reshape(100, 1)
    if type == 'cupy':
        X = cp.asarray(X)

    if type == 'numba':
        X = cuda.to_device(X)
    X_train, X_test = train_test_split(X,
                                       train_size=train_size,
                                       test_size=test_size,
                                       shuffle=shuffle,
                                       random_state=0)

    if type == 'cupy':
        assert isinstance(X_train, cp.ndarray)
        assert isinstance(X_test, cp.ndarray)

    if type in ['numba', 'rmm']:
        assert cuda.devicearray.is_cuda_ndarray(X_train)
        assert cuda.devicearray.is_cuda_ndarray(X_test)

    if train_size is not None:
        assert X_train.shape[0] == (int)(X.shape[0] * train_size)

    if test_size is not None:
        assert X_test.shape[0] == (int)(X.shape[0] * test_size)

    if shuffle is None:
        assert X_train == X[0:train_size]
        assert X_test == X[-1 * test_size:]

        X_rec = cp.sort(cp.concatenate(X_train, X_test))

        assert X_rec == X


@pytest.mark.parametrize('type', test_array_input_types)
@pytest.mark.parametrize('test_size', [0.2, 0.4, None])
@pytest.mark.parametrize('train_size', [0.6, 0.8, None])
def test_stratified_split(type, test_size, train_size):
    # For more tolerance and reliable estimates
    X, y = make_classification(n_samples=10000)

    if type == 'cupy':
        X = cp.asarray(X)
        y = cp.asarray(y)

    if type == 'numba':
        X = cuda.to_device(X)
        y = cuda.to_device(y)

    def counts(y):
        _, y_indices = cp.unique(y, return_inverse=True)
        class_counts = cp.bincount(y_indices)
        total = cp.sum(class_counts)
        percent_counts = []
        for count in (class_counts):
            percent_counts.append(cp.around(float(count)/total.item(),
                                            decimals=2).item())
        return percent_counts

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=train_size,
                                                        test_size=test_size,
                                                        stratify=True)

    original_counts = counts(y)
    split_counts = counts(y_train)
    assert cp.isclose(original_counts, split_counts,
                      equal_nan=False, rtol=0.1).all()
    if type == 'cupy':
        assert isinstance(X_train, cp.ndarray)
        assert isinstance(X_test, cp.ndarray)

    if type in ['numba']:
        assert cuda.devicearray.is_cuda_ndarray(X_train)
        assert cuda.devicearray.is_cuda_ndarray(X_test)


@pytest.mark.parametrize('seed_type', test_seeds)
def test_stratified_random_seed(seed_type):
    for i in range(10):
        seed_n = np.random.randint(0, int(1e9))
        if seed_type == 'int':
            seed = seed_n
        if seed_type == 'cupy':
            seed = cp.random.RandomState(seed=seed_n)
        if seed_type == 'numpy':
            seed = np.random.RandomState(seed=seed_n)
        X = cudf.DataFrame({"x": range(100)})
        y = cudf.Series(([0] * (100 // 2)) + ([1] * (100 // 2)))
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=seed,
                                                        stratify=True)

    if seed_type == 'cupy':
        seed = cp.random.RandomState(seed=seed_n)
    if seed_type == 'numpy':
        seed = np.random.RandomState(seed=seed_n)

    X_train2, X_test2, y_train2, y_test2 = \
        train_test_split(X, y, random_state=seed, stratify=True)

    assert X_train.equals(X_train2)
    assert X_test.equals(X_test2)
    assert y_train.equals(y_train2)
    assert y_test.equals(y_test2)

    # Ensure that data is shuffled
    assert not (X.head().index.values == X_train.head().index.values).all()

    def monotonic_inc(x):
        dx = cp.diff(x.values, axis=0)
        return cp.all(dx == 1)

    assert not monotonic_inc(X_train)


@pytest.mark.parametrize('test_size', [0.2, 0.4, None])
@pytest.mark.parametrize('train_size', [0.6, 0.8, None])
def test_stratify_retain_index(test_size, train_size):
    X = cudf.DataFrame({"x": range(10)})
    y = cudf.Series(([0] * (10 // 2)) + ([1] * (10 // 2)))

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=train_size,
                                                        test_size=test_size,
                                                        shuffle=True,
                                                        stratify=True)
    assert (X_train["x"] == X_train.index).all()
    assert (X_test["x"] == X_test.index).all()

    if train_size is not None:
        assert X_train.shape[0] == (int)(X.shape[0] * train_size)

    elif test_size is not None:
        assert X_test.shape[0] == (int)(X.shape[0] * test_size)
