# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import floating_dtypes, integer_dtypes
from sklearn.datasets import make_classification, make_regression

from cuml.testing.strategies import (
    regression_datasets,
    split_datasets,
    standard_classification_datasets,
    standard_datasets,
    standard_regression_datasets,
)


@example(
    dataset=(
        np.ones((10, 5), dtype=np.float32),
        np.ones((10, 1), dtype=np.float32),
    )
)
@given(standard_datasets())
def test_standard_datasets_default(dataset):
    X, y = dataset

    assert X.ndim == 2
    assert X.shape[0] <= 200
    assert X.shape[1] <= 200
    assert (y.ndim == 0) or (y.ndim in (1, 2) and y.shape[0] <= 200)


@example(
    dataset=(
        np.ones((10, 30), dtype=np.float32),
        np.ones((10, 1), dtype=np.float32),
    )
)
@given(
    standard_datasets(
        dtypes=floating_dtypes(sizes=(32,)),
        n_samples=st.integers(10, 20),
        n_features=st.integers(30, 40),
    )
)
def test_standard_datasets(dataset):
    X, y = dataset

    assert X.ndim == 2
    assert 10 <= X.shape[0] <= 20
    assert 30 <= X.shape[1] <= 40
    assert 10 <= y.shape[0] <= 20
    assert y.shape[1] == 1


@example(
    split_dataset=(
        np.ones((10, 5), dtype=np.float64),
        np.ones((5, 5), dtype=np.float64),
        np.ones(10, dtype=np.float64),
        np.ones(5, dtype=np.float64),
    )
)
@given(split_datasets(standard_datasets()))
@settings(suppress_health_check=list(HealthCheck))
def test_split_datasets(split_dataset):
    X_train, X_test, y_train, y_test = split_dataset

    assert X_train.ndim == X_test.ndim == 2
    assert X_train.shape[1] == X_test.shape[1]
    assert 2 <= (len(X_train) + len(X_test)) <= 200

    assert y_train.ndim == y_test.ndim
    assert y_train.ndim in (0, 1, 2)
    assert (y_train.ndim == 0) or (2 <= (len(y_train) + len(y_test)) <= 200)


@example(
    dataset=(
        np.ones((10, 5), dtype=np.float32),
        np.ones((10, 1), dtype=np.float32),
    )
)
@given(standard_regression_datasets())
def test_standard_regression_datasets_default(dataset):
    X, y = dataset
    assert X.ndim == 2
    assert X.shape[0] <= 200
    assert X.shape[1] <= 200
    assert (y.ndim == 0) or (y.ndim in (1, 2) and y.shape[0] <= 200)
    assert X.dtype == y.dtype


@example(dataset=make_regression(n_samples=1, n_features=5, random_state=0))
@given(
    standard_regression_datasets(
        dtypes=floating_dtypes(sizes=64),
        n_samples=st.integers(min_value=2, max_value=200),
        n_features=st.integers(min_value=1, max_value=200),
        n_informative=st.just(10),
        random_state=0,
    )
)
def test_standard_regression_datasets(dataset):
    from sklearn.datasets import make_regression

    X, y = dataset
    assert X.ndim == 2
    assert X.shape[0] <= 200
    assert X.shape[1] <= 200
    assert (y.ndim == 1 and y.shape[0] <= 200) or y.ndim == 0
    assert X.dtype == y.dtype

    X_cmp, y_cmp = make_regression(
        n_samples=X.shape[0], n_features=X.shape[1], random_state=0
    )

    assert X.dtype.type == X_cmp.dtype.type
    assert X.ndim == X_cmp.ndim
    assert X.shape == X_cmp.shape
    assert y.dtype.type == y_cmp.dtype.type
    assert y.ndim == y_cmp.ndim
    assert y.shape == y_cmp.shape
    assert (X == X_cmp).all()
    assert (y == y_cmp).all()


@example(
    dataset=(
        np.ones((10, 5), dtype=np.float32),
        np.ones((10, 1), dtype=np.float32),
    )
)
@given(regression_datasets())
def test_regression_datasets(dataset):
    X, y = dataset

    assert X.ndim == 2
    assert X.shape[0] <= 200
    assert X.shape[1] <= 200
    assert (y.ndim == 0) or (y.ndim in (1, 2) and y.shape[0] <= 200)


@example(
    split_dataset=(
        np.ones((10, 5), dtype=np.float64),
        np.ones((5, 5), dtype=np.float64),
        np.ones(10, dtype=np.float64),
        np.ones(5, dtype=np.float64),
    )
)
@given(split_datasets(regression_datasets()))
@settings(
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large]
)
def test_split_regression_datasets(split_dataset):
    X_train, X_test, y_train, y_test = split_dataset

    assert X_train.ndim == X_test.ndim == 2
    assert y_train.ndim == y_test.ndim
    assert y_train.ndim in (0, 1, 2)
    assert 2 <= (len(X_train) + len(X_test)) <= 200


@example(
    dataset=(np.ones((10, 5), dtype=np.float32), np.ones(10, dtype=np.int32))
)
@given(standard_classification_datasets())
def test_standard_classification_datasets_default(dataset):
    X, y = dataset
    assert X.ndim == 2
    assert X.shape[0] <= 200
    assert X.shape[1] <= 200
    assert (y.ndim == 0) or (y.ndim in (1, 2) and y.shape[0] <= 200)
    assert np.issubdtype(X.dtype, np.floating)
    assert np.issubdtype(y.dtype, np.integer)


@example(
    dataset=make_classification(n_samples=1, n_features=5, random_state=0)
)
@given(
    standard_classification_datasets(
        dtypes=floating_dtypes(sizes=64),
        n_samples=st.integers(min_value=2, max_value=200),
        n_features=st.integers(min_value=4, max_value=200),
        n_informative=st.just(2),
        n_redundant=st.just(2),
        random_state=0,
        labels_dtypes=integer_dtypes(sizes=64),
    )
)
def test_standard_classification_datasets(dataset):
    from sklearn.datasets import make_classification

    X, y = dataset
    assert X.ndim == 2
    assert X.shape[0] <= 200
    assert X.shape[1] <= 200
    assert (y.ndim == 1 and y.shape[0] <= 200) or y.ndim == 0
    assert np.issubdtype(X.dtype, np.floating)
    assert np.issubdtype(y.dtype, np.integer)

    X_cmp, y_cmp = make_classification(
        n_samples=X.shape[0],
        n_features=X.shape[1],
        random_state=0,
    )

    assert X.dtype.type == X_cmp.dtype.type
    assert X.ndim == X_cmp.ndim
    assert X.shape == X_cmp.shape
    assert y.dtype.type == y_cmp.dtype.type
    assert y.ndim == y_cmp.ndim
    assert y.shape == y_cmp.shape
    assert (X == X_cmp).all()
    assert (y == y_cmp).all()
