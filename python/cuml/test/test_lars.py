# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
from functools import lru_cache
import cupy as cp
import numpy as np
import pytest
from distutils.version import LooseVersion
import cudf
from cuml.experimental.linear_model import Lars as cuLars
from cuml.common import logger
from cuml.test.utils import (
    array_equal,
    small_regression_dataset,
    small_classification_dataset,
    unit_param,
    quality_param,
    stress_param,
)

#import sklearn
from sklearn.datasets import load_boston
from sklearn.linear_model import Lars as skLars
from sklearn.model_selection import train_test_split

from . test_linear_model import make_regression_dataset


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
@pytest.mark.parametrize("precompute", [True, False, 'precompute'])
def test_lars_model(datatype, nrows, column_info, precompute):

    ncols, n_info = column_info
    X_train, X_test, y_train, y_test = make_regression_dataset(
        datatype, nrows, ncols, n_info
    )
    if precompute=='precompute':
        precompute = np.dot(X_train.T, X_train)
    params = {'precompute': precompute}

    # Initialization of cuML's LARS
    culars = cuLars(**params)

    # fit and predict cuml LARS
    if not (datatype == np.float32 and nrows >=500000):
        culars.fit(X_train, y_train)

        cu_score_train = culars.score(X_train, y_train)
        cu_score_test = culars.score(X_test, y_test)

    if nrows < 500000:
        # sklearn model initialization, fit and predict
        sklars = skLars(**params)
        sklars.fit(X_train, y_train)

        assert cu_score_train >= sklars.score(X_train, y_train) - 0.05
        assert cu_score_test >= sklars.score(X_test, y_test) - 0.1
    else:
        if datatype == np.float32:
            # We ignore this test until the cuBLAS error is fixed with
            # n_rows > 65536
            pass
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
def test_lars_colinear(datatype, nrows, column_info, precompute):
    # TODO Fix handling of colinear input and run these tests
    pytest.skip("Handling collinear input needs improvement")
    ncols, n_info = column_info

    X_train, X_test, y_train, y_test = make_regression_dataset(
        datatype, nrows, ncols, n_info
    )
    n_duplicate = min(ncols, 100)
    X_train = np.concatenate((X_train, X_train[:, :n_duplicate]), axis=1)
    X_test = np.concatenate((X_test, X_test[:, :n_duplicate]), axis=1)

    params = {"precompute": precompute, "n_nonzero_coefs": ncols + n_duplicate,
              "eps": 1e-6}
    culars = cuLars(**params)
    culars.fit(X_train, y_train)

    assert culars.score(X_train, y_train) > 0.1
    assert culars.score(X_test, y_test) > 0.1


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("params", [{"precompute": True},
                                    {"precompute": False},
                                    {"n_nonzero_coefs": 5},
                                    {"n_nonzero_coefs": 2}])
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

    n = min(culars.n_iter_, sklars.n_iter_)
    assert array_equal(culars.alphas_[:n], sklars.alphas_[:n])
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

    # TODO We make a copy of X during preprocessing, we should preprocess
    # in place if copy_X is false
    # culars2 = cuLars(precompute=False, copy_X=False)
    # culars2.fit(X, y)
    # Test that array was changed i.e. no unnecessary copies were made
    # assert cp.any(X0 != X)
    #
    # assert abs(culars1.score(X, y) - culars2.score(X, y)) < 1e-9
