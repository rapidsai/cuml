# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import hypothesis.strategies as st
import numpy as np
import pytest
import sklearn.linear_model
from hypothesis import assume, example, given, note, target

import cuml
from cuml.testing.datasets import (
    is_cuml_compatible_dataset,
    is_sklearn_compatible_dataset,
    make_regression,
    make_regression_dataset,
    small_regression_dataset,
    split_datasets,
    standard_regression_datasets,
)
from cuml.testing.strategies import dataset_dtypes
from cuml.testing.utils import array_equal


@given(
    split_datasets(
        standard_regression_datasets(),
    ),
)
@example(small_regression_dataset(np.float32))
@example(small_regression_dataset(np.float64))
def test_ridge_regression_model_default(dataset):
    assume(is_sklearn_compatible_dataset(*dataset))
    assume(is_cuml_compatible_dataset(*dataset))
    X_train, X_test, y_train, _ = dataset

    curidge = cuml.Ridge()

    # fit and predict cuml ridge regression model
    curidge.fit(X_train, y_train)
    curidge_predict = curidge.predict(X_test)

    # sklearn ridge regression model initialization, fit and predict
    skridge = sklearn.linear_model.Ridge()
    skridge.fit(X_train, y_train)
    skridge_predict = skridge.predict(X_test)

    equal = array_equal(
        skridge_predict,
        curidge_predict,
        unit_tol=1e-1,
        total_tol=1e-3,
        with_sign=True,
    )
    note(equal)
    target(float(np.abs(equal.compute_difference())))
    assert equal


@given(
    datatype=dataset_dtypes(),
    algorithm=st.sampled_from(["eig", "svd"]),
    nrows=st.integers(min_value=500, max_value=5000),
    column_info=st.sampled_from([[20, 10], [100, 50]]),
)
@example(datatype=np.float32, algorithm="eig", nrows=500, column_info=[20, 10])
@example(
    datatype=np.float64, algorithm="svd", nrows=5000, column_info=[100, 50]
)
def test_ridge_regression_model(datatype, algorithm, nrows, column_info):
    if algorithm == "svd" and nrows > 46340:
        pytest.skip(
            "svd solver is not supported for the data that has more"
            "than 46340 rows or columns if you are using CUDA version"
            "10.x"
        )

    ncols, n_info = column_info
    X_train, X_test, y_train, y_test = make_regression_dataset(
        datatype, nrows, ncols, n_info
    )

    # Initialization of cuML's ridge regression model
    curidge = cuml.Ridge(fit_intercept=False, solver=algorithm)

    # fit and predict cuml ridge regression model
    curidge.fit(X_train, y_train)
    curidge_predict = curidge.predict(X_test)

    if nrows < 500000:
        # sklearn ridge regression model initialization, fit and predict
        skridge = sklearn.linear_model.Ridge(fit_intercept=False)
        skridge.fit(X_train, y_train)

        skridge_predict = skridge.predict(X_test)

        assert array_equal(
            skridge_predict, curidge_predict, 1e-1, with_sign=True
        )


def test_ridge_and_least_squares_equal_when_alpha_is_0():
    X, y = make_regression(n_samples=5, n_features=4, random_state=0)

    ridge = cuml.Ridge(alpha=0.0, fit_intercept=False)
    ols = cuml.LinearRegression(fit_intercept=False)

    ridge.fit(X, y)
    ols.fit(X, y)
    assert array_equal(ridge.coef_, ols.coef_)


@given(
    datatype=dataset_dtypes(),
    algorithm=st.sampled_from(["eig", "svd"]),
    fit_intercept=st.booleans(),
    distribution=st.sampled_from(["lognormal", "exponential", "uniform"]),
)
@example(
    datatype=np.float32,
    algorithm="eig",
    fit_intercept=True,
    distribution="uniform",
)
@example(
    datatype=np.float64,
    algorithm="svd",
    fit_intercept=False,
    distribution="lognormal",
)
def test_weighted_ridge(datatype, algorithm, fit_intercept, distribution):
    nrows, ncols, n_info = 1000, 20, 10
    max_weight = 10
    noise = 20
    X_train, X_test, y_train, y_test = make_regression_dataset(
        datatype, nrows, ncols, n_info, noise=noise
    )

    # set weight per sample to be from 1 to max_weight
    if distribution == "uniform":
        wt = np.random.randint(1, high=max_weight, size=len(X_train))
    elif distribution == "exponential":
        wt = np.random.exponential(size=len(X_train))
    else:
        wt = np.random.lognormal(size=len(X_train))

    # Initialization of cuML's linear regression model
    curidge = cuml.Ridge(fit_intercept=fit_intercept, solver=algorithm)

    # fit and predict cuml linear regression model
    curidge.fit(X_train, y_train, sample_weight=wt)
    curidge_predict = curidge.predict(X_test)

    # sklearn linear regression model initialization, fit and predict
    skridge = sklearn.linear_model.Ridge(fit_intercept=fit_intercept)
    skridge.fit(X_train, y_train, sample_weight=wt)

    skridge_predict = skridge.predict(X_test)

    assert array_equal(skridge_predict, curidge_predict, 1e-1, with_sign=True)


@given(
    dataset=split_datasets(
        standard_regression_datasets(),
    ),
    test_dtype=dataset_dtypes(),
)
@example(dataset=small_regression_dataset(np.float32), test_dtype=np.float32)
@example(dataset=small_regression_dataset(np.float32), test_dtype=np.float64)
@example(dataset=small_regression_dataset(np.float64), test_dtype=np.float32)
@example(dataset=small_regression_dataset(np.float64), test_dtype=np.float64)
def test_ridge_predict_convert_dtype(dataset, test_dtype):
    assume(is_cuml_compatible_dataset(*dataset))
    X_train, X_test, y_train, _ = dataset

    clf = cuml.Ridge()
    clf.fit(X_train, y_train)
    clf.predict(X_test.astype(test_dtype))
