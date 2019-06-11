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

import pytest
from cuml import LinearRegression as cuLinearRegression
from cuml import Ridge as cuRidge
from sklearn.linear_model import LinearRegression as skLinearRegression
from sklearn.linear_model import Ridge as skRidge
from cuml.test.utils import array_equal
import cudf
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression


def unit_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.unit)


def quality_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.quality)


def stress_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.stress)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('X_type', ['ndarray'])
@pytest.mark.parametrize('y_type', ['series', 'ndarray'])
@pytest.mark.parametrize('algorithm', ['eig', 'svd'])
@pytest.mark.parametrize('nrows', [unit_param(20), quality_param(5000),
                         stress_param(500000)])
@pytest.mark.parametrize('ncols', [unit_param(3), quality_param(100),
                         stress_param(1000)])
@pytest.mark.parametrize('n_info', [unit_param(2), quality_param(50),
                         stress_param(500)])
def test_linear_models(datatype, X_type, y_type,
                       algorithm, nrows, ncols, n_info):
    train_rows = np.int32(nrows*0.8)
    X, y = make_regression(n_samples=(nrows), n_features=ncols,
                           n_informative=n_info, random_state=0)
    X_test = np.asarray(X[train_rows:, 0:]).astype(datatype)
    X_train = np.asarray(X[0:train_rows, :]).astype(datatype)
    y_train = np.asarray(y[0:train_rows, ]).astype(datatype)

    if nrows != 500000:
        # sklearn linear and ridge regression model initialization and fit
        skols = skLinearRegression(fit_intercept=True,
                                   normalize=False)
        skols.fit(X_train, y_train)
        skridge = skRidge(fit_intercept=False,
                          normalize=False)
        skridge.fit(X_train, y_train)

    # Initialization of cuML's linear and ridge regression models
    cuols = cuLinearRegression(fit_intercept=True,
                               normalize=False,
                               algorithm=algorithm)

    curidge = cuRidge(fit_intercept=False,
                      normalize=False,
                      solver=algorithm)

    if X_type == 'dataframe':
        y_train = pd.DataFrame({'labels': y_train[0:, ]})
        X_train = pd.DataFrame(
            {'fea%d' % i: X_train[0:, i] for i in range(X_train.shape[1])})
        X_test = pd.DataFrame(
            {'fea%d' % i: X_test[0:, i] for i in range(X_test.shape[1])})
        X_cudf = cudf.DataFrame.from_pandas(X_train)
        X_cudf_test = cudf.DataFrame.from_pandas(X_test)
        y_cudf = y_train.values
        y_cudf = y_cudf[:, 0]
        y_cudf = cudf.Series(y_cudf)

        # fit and predict cuml linear regression model
        cuols.fit(X_cudf, y_cudf)
        cuols_predict = cuols.predict(X_cudf_test).to_array()

        # fit and predict cuml ridge regression model
        curidge.fit(X_cudf, y_cudf)
        curidge_predict = curidge.predict(X_cudf_test).to_array()

    elif X_type == 'ndarray':

        # fit and predict cuml linear regression model
        cuols.fit(X_train, y_train)
        cuols_predict = cuols.predict(X_test).to_array()

        # fit and predict cuml ridge regression model
        curidge.fit(X_train, y_train)
        curidge_predict = curidge.predict(X_test).to_array()

    if nrows != 500000:
        skols_predict = skols.predict(X_test)
        skridge_predict = skridge.predict(X_test)
        assert array_equal(skols_predict, cuols_predict,
                           1e-1, with_sign=True)
        assert array_equal(skridge_predict, curidge_predict,
                           1e-1, with_sign=True)
