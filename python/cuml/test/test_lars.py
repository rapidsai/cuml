# Copyright (c) 2020, NVIDIA CORPORATION.
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
import cupy as cp
import numpy as np
import pytest

from cuml.experimental.linear_model import Lars as cuLars
from cuml.test.utils import (
    array_equal,
    unit_param,
    quality_param,
    stress_param,
)

from sklearn.datasets import load_boston
from sklearn.linear_model import Lars as skLars

from . test_linear_model import make_regression_dataset


def normalize_data(X, y):
    y_mean = np.mean(y)
    y = y - y_mean
    x_mean = np.mean(X, axis=0)
    x_scale = np.sqrt(np.var(X, axis=0) * X.shape[0])
    x_scale[x_scale == 0] = 1
    X = (X - x_mean) / x_scale
    return X, y, x_mean, x_scale, y_mean


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "nrows", [unit_param(500), quality_param(5000), stress_param(90000)]
)
@pytest.mark.parametrize(
    "column_info",
    [
        unit_param([1, 1]),
        unit_param([20, 10]),
        quality_param([100, 50]),
        stress_param([1000, 500])
    ],
)
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("precompute", [True, False, 'precompute'])
def test_lars_model(datatype, nrows, column_info, precompute, normalize):
    ncols, n_info = column_info
    X_train, X_test, y_train, y_test = make_regression_dataset(
        datatype, nrows, ncols, n_info
    )

    if precompute == 'precompute' or not normalize:
        # Apply normalization manually, because the solver expects normalized
        # input data
        X_train, y_train, x_mean, x_scale, y_mean = \
            normalize_data(X_train, y_train)
        y_test = y_test - y_mean
        X_test = (X_test - x_mean) / x_scale

    if precompute == 'precompute':
        precompute = np.dot(X_train.T, X_train)

    params = {'precompute': precompute, 'normalize': normalize}

    # Initialization of cuML's LARS
    culars = cuLars(**params)

    # fit and predict cuml LARS
    culars.fit(X_train, y_train)

    cu_score_train = culars.score(X_train, y_train)
    cu_score_test = culars.score(X_test, y_test)

    if nrows < 500000:
        # sklearn model initialization, fit and predict
        sklars = skLars(**params)
        sklars.fit(X_train, y_train)
        # Set tolerance to include the 95% confidence interval around
        # scikit-learn accuracy.
        accuracy_target = sklars.score(X_test, y_test)
        tol = 1.96 * np.sqrt(accuracy_target * (1.0-accuracy_target)/100.0)
        if tol < 0.001:
            tol = 0.001  # We allow at least 0.1% tolerance
        print(cu_score_train, cu_score_test, accuracy_target, tol)
        assert cu_score_train >= sklars.score(X_train, y_train) - tol
        assert cu_score_test >= accuracy_target - tol
    else:
        assert cu_score_test > 0.95


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "nrows", [unit_param(500), quality_param(5000), stress_param(500000)]
)
@pytest.mark.parametrize(
    "column_info",
    [
        unit_param([20, 10]),
        quality_param([100, 50]),
        stress_param([1000, 500])
    ],
)
@pytest.mark.parametrize("precompute", [True, False])
def test_lars_collinear(datatype, nrows, column_info, precompute):
    ncols, n_info = column_info

    X_train, X_test, y_train, y_test = make_regression_dataset(
        datatype, nrows, ncols, n_info
    )
    n_duplicate = min(ncols, 100)
    X_train = np.concatenate((X_train, X_train[:, :n_duplicate]), axis=1)
    X_test = np.concatenate((X_test, X_test[:, :n_duplicate]), axis=1)

    params = {"precompute": precompute, "n_nonzero_coefs": ncols + n_duplicate}
    culars = cuLars(**params)
    culars.fit(X_train, y_train)

    assert culars.score(X_train, y_train) > 0.85
    assert culars.score(X_test, y_test) > 0.85


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("params", [{"precompute": True},
                                    {"precompute": False},
                                    {"n_nonzero_coefs": 5},
                                    {"n_nonzero_coefs": 2},
                                    {"n_nonzero_coefs": 2,
                                     "fit_intercept": False}])
def test_lars_attributes(datatype, params):
    X, y = load_boston(return_X_y=True)
    X = X.astype(datatype)
    y = y.astype(datatype)

    culars = cuLars(**params)
    culars.fit(X, y)

    sklars = skLars(**params)
    sklars.fit(X, y)

    assert culars.score(X, y) >= sklars.score(X, y) - 0.01

    limit_max_iter = "n_nonzero_coefs" in params
    if limit_max_iter:
        n_iter_tol = 0
    else:
        n_iter_tol = 2

    assert abs(culars.n_iter_ - sklars.n_iter_) <= n_iter_tol

    tol = 1e-4 if params.pop("fit_intercept", True) else 1e-1
    n = min(culars.n_iter_, sklars.n_iter_)
    assert array_equal(culars.alphas_[:n], sklars.alphas_[:n], unit_tol=tol,
                       total_tol=1e-4)
    assert array_equal(culars.active_[:n], sklars.active_[:n])

    if limit_max_iter:
        assert array_equal(culars.coef_, sklars.coef_)

        if hasattr(sklars, 'coef_path_'):
            assert array_equal(culars.coef_path_,
                               sklars.coef_path_[sklars.active_],
                               unit_tol=1e-3)

        intercept_diff = abs(culars.intercept_ - sklars.intercept_)
        if abs(sklars.intercept_) > 1e-6:
            intercept_diff /= sklars.intercept_
            assert intercept_diff <= 1e-3


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
def test_lars_copy_X(datatype):
    X, y = load_boston(return_X_y=True)
    X = cp.asarray(X, dtype=datatype, order='F')
    y = cp.asarray(y, dtype=datatype, order='F')

    X0 = cp.copy(X)
    culars1 = cuLars(precompute=False, copy_X=True)
    culars1.fit(X, y)
    # Test that array was not changed
    assert cp.all(X0 == X)

    # We make a copy of X during preprocessing, we should preprocess
    # in place if copy_X is false to save memory. Afterwards we can enable
    # the following test:
    # culars2 = cuLars(precompute=False, copy_X=False)
    # culars2.fit(X, y)
    # Test that array was changed i.e. no unnecessary copies were made
    # assert cp.any(X0 != X)
    #
    # assert abs(culars1.score(X, y) - culars2.score(X, y)) < 1e-9
