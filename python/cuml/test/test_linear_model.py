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

import numpy as np
import pandas as pd
import pytest
from distutils.version import LooseVersion

from cuml import LinearRegression as cuLinearRegression
from cuml import LogisticRegression as cuLog
from cuml import Ridge as cuRidge
from cuml.test.utils import array_equal, small_regression_dataset, \
    small_classification_dataset, unit_param, quality_param, stress_param

import sklearn
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import LinearRegression as skLinearRegression
from sklearn.linear_model import Ridge as skRidge
from sklearn.linear_model import LogisticRegression as skLog
from sklearn.model_selection import train_test_split


def make_regression_dataset(datatype, nrows, ncols, n_info):
    X, y = make_regression(n_samples=nrows, n_features=ncols,
                           n_informative=n_info, random_state=0)
    X = X.astype(datatype)
    y = y.astype(datatype)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    return X_train, X_test, y_train, y_test


def make_classification_dataset(datatype, nrows, ncols, n_info, num_classes):
    X, y = make_classification(n_samples=nrows, n_features=ncols,
                               n_informative=n_info, n_classes=num_classes,
                               random_state=0)
    X = X.astype(datatype)
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    return X_train, X_test, y_train, y_test


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('algorithm', ['eig', 'svd'])
@pytest.mark.parametrize('nrows', [unit_param(20), quality_param(5000),
                         stress_param(500000)])
@pytest.mark.parametrize('ncols', [unit_param(3), quality_param(100),
                         stress_param(1000)])
@pytest.mark.parametrize('n_info', [unit_param(2), quality_param(50),
                         stress_param(500)])
def test_linear_regression_model(datatype, algorithm, nrows, ncols, n_info):

    X_train, X_test, y_train, y_test = make_regression_dataset(datatype,
                                                               nrows,
                                                               ncols,
                                                               n_info)

    # Initialization of cuML's linear regression model
    cuols = cuLinearRegression(fit_intercept=True,
                               normalize=False,
                               algorithm=algorithm)

    # fit and predict cuml linear regression model
    cuols.fit(X_train, y_train)
    cuols_predict = cuols.predict(X_test).to_array()

    if nrows < 500000:
        # sklearn linear regression model initialization, fit and predict
        skols = skLinearRegression(fit_intercept=True,
                                   normalize=False)
        skols.fit(X_train, y_train)

        skols_predict = skols.predict(X_test)

        assert array_equal(skols_predict, cuols_predict,
                           1e-1, with_sign=True)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
def test_linear_regression_model_default(datatype):

    X_train, X_test, y_train, y_test = small_regression_dataset(datatype)

    # Initialization of cuML's linear regression model
    cuols = cuLinearRegression()

    # fit and predict cuml linear regression model
    cuols.fit(X_train, y_train)
    cuols_predict = cuols.predict(X_test).to_array()

    # sklearn linear regression model initialization and fit
    skols = skLinearRegression()
    skols.fit(X_train, y_train)

    skols_predict = skols.predict(X_test)

    assert array_equal(skols_predict, cuols_predict,
                       1e-1, with_sign=True)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
def test_ridge_regression_model_default(datatype):

    X_train, X_test, y_train, y_test = small_regression_dataset(datatype)

    curidge = cuRidge()

    # fit and predict cuml ridge regression model
    curidge.fit(X_train, y_train)
    curidge_predict = curidge.predict(X_test).to_array()

    # sklearn ridge regression model initialization, fit and predict
    skridge = skRidge()
    skridge.fit(X_train, y_train)
    skridge_predict = skridge.predict(X_test)

    assert array_equal(skridge_predict, curidge_predict,
                       1e-1, with_sign=True)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('algorithm', ['eig', 'svd'])
@pytest.mark.parametrize('nrows', [unit_param(20), quality_param(5000),
                         stress_param(500000)])
@pytest.mark.parametrize('ncols', [unit_param(3), quality_param(100),
                         stress_param(1000)])
@pytest.mark.parametrize('n_info', [unit_param(2), quality_param(50),
                         stress_param(500)])
def test_ridge_regression_model(datatype, algorithm, nrows, ncols, n_info):

    X_train, X_test, y_train, y_test = make_regression_dataset(datatype,
                                                               nrows,
                                                               ncols,
                                                               n_info)

    # Initialization of cuML's ridge regression model
    curidge = cuRidge(fit_intercept=False,
                      normalize=False,
                      solver=algorithm)

    # fit and predict cuml ridge regression model
    curidge.fit(X_train, y_train)
    curidge_predict = curidge.predict(X_test).to_array()

    if nrows < 500000:
        # sklearn ridge regression model initialization, fit and predict
        skridge = skRidge(fit_intercept=False,
                          normalize=False)
        skridge.fit(X_train, y_train)

        skridge_predict = skridge.predict(X_test)

        assert array_equal(skridge_predict, curidge_predict,
                           1e-1, with_sign=True)


@pytest.mark.parametrize('num_classes', [2, 10])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('penalty', ['none', 'l1', 'l2', 'elasticnet'])
@pytest.mark.parametrize('l1_ratio', [0.0, 0.3, 0.5, 0.7, 1.0])
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('nrows', [unit_param(200), quality_param(5000),
                         stress_param(500000)])
@pytest.mark.parametrize('ncols', [unit_param(10), quality_param(60),
                         stress_param(100)])
@pytest.mark.parametrize('n_info', [unit_param(7), quality_param(40),
                         stress_param(70)])
def test_logistic_regression(num_classes, dtype, penalty, l1_ratio,
                             fit_intercept, nrows, ncols, n_info):

    # Checking sklearn >= 0.21 for testing elasticnet
    sk_check = LooseVersion(str(sklearn.__version__)) >= LooseVersion("0.21.0")
    if not sk_check and penalty == 'elasticnet':
        pytest.skip("Need sklearn > 0.21 for testing logistic with"
                    "elastic net.")

    X_train, X_test, y_train, y_test = \
        make_classification_dataset(datatype=dtype, nrows=nrows,
                                    ncols=ncols, n_info=n_info,
                                    num_classes=num_classes)
    y_train = y_train.astype(dtype)
    y_test = y_test.astype(dtype)
    culog = cuLog(penalty=penalty, l1_ratio=l1_ratio, C=5.0,
                  fit_intercept=fit_intercept, tol=1e-8)
    culog.fit(X_train, y_train)

    # Only solver=saga supports elasticnet in scikit
    if penalty in ['elasticnet', 'l1']:
        if sk_check:
            sklog = skLog(penalty=penalty, l1_ratio=l1_ratio, solver='saga',
                          C=5.0, fit_intercept=fit_intercept)
        else:
            sklog = skLog(penalty=penalty, solver='saga',
                          C=5.0, fit_intercept=fit_intercept)
    elif penalty == 'l2':
        sklog = skLog(penalty=penalty, solver='lbfgs', C=5.0,
                      fit_intercept=fit_intercept)
    else:
        if sk_check:
            sklog = skLog(penalty=penalty, solver='lbfgs', C=5.0,
                          fit_intercept=fit_intercept)
        else:
            sklog = skLog(penalty='l2', solver='lbfgs', C=1e9,
                          fit_intercept=fit_intercept)

    sklog.fit(X_train, y_train)

    preds = culog.predict(X_test)
    skpreds = sklog.predict(X_test)

    # Setting tolerance to lowest possible per loss to detect regressions
    # as much as possible
    if penalty in ['elasticnet', 'l1', 'l2']:
        assert np.sum(preds.to_array() != skpreds)/20000 < 1e-1
    else:
        # This is the only case where cuml and sklearn actually do a similar
        # lbfgs, other cases cuml does owl or sklearn does saga
        assert np.sum(preds.to_array() != skpreds)/20000 < 1e-3


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_logistic_regression_model_default(dtype):

    X_train, X_test, y_train, y_test = small_classification_dataset(dtype)
    y_train = y_train.astype(dtype)
    y_test = y_test.astype(dtype)
    culog = cuLog()
    culog.fit(X_train, y_train)
    # Only solver=saga supports elasticnet in scikit
    sklog = skLog()

    sklog.fit(X_train, y_train)

    preds = culog.predict(X_test)
    skpreds = sklog.predict(X_test)

    assert np.sum(preds.to_array() != skpreds)/20000 < 1e-1
