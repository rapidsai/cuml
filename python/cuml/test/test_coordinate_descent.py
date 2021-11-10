# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
import numpy as np

from cuml import Lasso as cuLasso
from cuml.linear_model import ElasticNet as cuElasticNet
from cuml.metrics import r2_score
from cuml.test.utils import unit_param, quality_param, stress_param

from sklearn.linear_model import Lasso, ElasticNet
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('X_type', ['ndarray'])
@pytest.mark.parametrize('alpha', [0.1, 0.001])
@pytest.mark.parametrize('algorithm', ['cyclic', 'random'])
@pytest.mark.parametrize('nrows', [unit_param(500), quality_param(5000),
                         stress_param(500000)])
@pytest.mark.parametrize('column_info', [unit_param([20, 10]),
                         quality_param([100, 50]),
                         stress_param([1000, 500])])
@pytest.mark.filterwarnings("ignore:Objective did not converge::sklearn[.*]")
def test_lasso(datatype, X_type, alpha, algorithm,
               nrows, column_info):
    ncols, n_info = column_info
    X, y = make_regression(n_samples=nrows, n_features=ncols,
                           n_informative=n_info, random_state=0)
    X = X.astype(datatype)
    y = y.astype(datatype)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=0)
    cu_lasso = cuLasso(alpha=np.array([alpha]), fit_intercept=True,
                       normalize=False, max_iter=1000,
                       selection=algorithm, tol=1e-10)

    cu_lasso.fit(X_train, y_train)
    assert cu_lasso.coef_ is not None
    cu_predict = cu_lasso.predict(X_test)

    cu_r2 = r2_score(y_test, cu_predict)

    if nrows < 500000:
        sk_lasso = Lasso(alpha=np.array([alpha]), fit_intercept=True,
                         normalize=False, max_iter=1000,
                         selection=algorithm, tol=1e-10)
        sk_lasso.fit(X_train, y_train)
        sk_predict = sk_lasso.predict(X_test)
        sk_r2 = r2_score(y_test, sk_predict)
        assert cu_r2 >= sk_r2 - 0.07


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('column_info', [unit_param([20, 10]),
                         quality_param([100, 50]),
                         stress_param([1000, 500])])
@pytest.mark.parametrize('nrows', [unit_param(500), quality_param(5000),
                         stress_param(500000)])
def test_lasso_default(datatype, nrows, column_info):

    ncols, n_info = column_info
    X, y = make_regression(n_samples=nrows, n_features=ncols,
                           n_informative=n_info, random_state=0)
    X = X.astype(datatype)
    y = y.astype(datatype)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=0)

    cu_lasso = cuLasso()

    cu_lasso.fit(X_train, y_train)
    assert cu_lasso.coef_ is not None
    cu_predict = cu_lasso.predict(X_test)
    cu_r2 = r2_score(y_test, cu_predict)

    sk_lasso = Lasso()
    sk_lasso.fit(X_train, y_train)
    sk_predict = sk_lasso.predict(X_test)
    sk_r2 = r2_score(y_test, sk_predict)
    assert cu_r2 >= sk_r2 - 0.07


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('X_type', ['ndarray'])
@pytest.mark.parametrize('alpha', [0.2, 0.7])
@pytest.mark.parametrize('algorithm', ['cyclic', 'random'])
@pytest.mark.parametrize('nrows', [unit_param(500), quality_param(5000),
                         stress_param(500000)])
@pytest.mark.parametrize('column_info', [unit_param([20, 10]),
                         quality_param([100, 50]),
                         stress_param([1000, 500])])
@pytest.mark.filterwarnings("ignore:Objective did not converge::sklearn[.*]")
def test_elastic_net(datatype, X_type, alpha, algorithm,
                     nrows, column_info):
    ncols, n_info = column_info
    X, y = make_regression(n_samples=nrows, n_features=ncols,
                           n_informative=n_info, random_state=0)
    X = X.astype(datatype)
    y = y.astype(datatype)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=0)

    elastic_cu = cuElasticNet(alpha=np.array([alpha]), fit_intercept=True,
                              normalize=False, max_iter=1000,
                              selection=algorithm, tol=1e-10)

    elastic_cu.fit(X_train, y_train)
    cu_predict = elastic_cu.predict(X_test)

    cu_r2 = r2_score(y_test, cu_predict)

    if nrows < 500000:
        elastic_sk = ElasticNet(alpha=np.array([alpha]), fit_intercept=True,
                                normalize=False, max_iter=1000,
                                selection=algorithm, tol=1e-10)
        elastic_sk.fit(X_train, y_train)
        sk_predict = elastic_sk.predict(X_test)
        sk_r2 = r2_score(y_test, sk_predict)

        assert cu_r2 >= sk_r2 - 0.07


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('column_info', [unit_param([20, 10]),
                         quality_param([100, 50]),
                         stress_param([1000, 500])])
@pytest.mark.parametrize('nrows', [unit_param(500), quality_param(5000),
                         stress_param(500000)])
def test_elastic_net_default(datatype, nrows, column_info):

    ncols, n_info = column_info
    X, y = make_regression(n_samples=nrows, n_features=ncols,
                           n_informative=n_info, random_state=0)
    X = X.astype(datatype)
    y = y.astype(datatype)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=0)

    elastic_cu = cuElasticNet()
    elastic_cu.fit(X_train, y_train)
    cu_predict = elastic_cu.predict(X_test)
    cu_r2 = r2_score(y_test, cu_predict)

    elastic_sk = ElasticNet()
    elastic_sk.fit(X_train, y_train)
    sk_predict = elastic_sk.predict(X_test)
    sk_r2 = r2_score(y_test, sk_predict)
    assert cu_r2 >= sk_r2 - 0.07


@pytest.mark.parametrize('train_dtype', [np.float32, np.float64])
@pytest.mark.parametrize('test_dtype', [np.float64, np.float32])
def test_elastic_net_predict_convert_dtype(train_dtype, test_dtype):
    X, y = make_regression(n_samples=50, n_features=10,
                           n_informative=5, random_state=0)
    X = X.astype(train_dtype)
    y = y.astype(train_dtype)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=0)

    clf = cuElasticNet()
    clf.fit(X_train, y_train)
    clf.predict(X_test.astype(test_dtype))


@pytest.mark.parametrize('train_dtype', [np.float32, np.float64])
@pytest.mark.parametrize('test_dtype', [np.float64, np.float32])
def test_lasso_predict_convert_dtype(train_dtype, test_dtype):
    X, y = make_regression(n_samples=50, n_features=10,
                           n_informative=5, random_state=0)
    X = X.astype(train_dtype)
    y = y.astype(train_dtype)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=0)

    clf = cuLasso()
    clf.fit(X_train, y_train)
    clf.predict(X_test.astype(test_dtype))


@pytest.mark.parametrize('algo', [cuElasticNet, cuLasso])
def test_set_params(algo):
    x = np.linspace(0, 1, 50)
    y = 2 * x

    model = algo(alpha=0.01)
    model.fit(x, y)
    coef_before = model.coef_

    model = algo(selection="random", alpha=0.1)
    model.fit(x, y)
    coef_after = model.coef_

    model = algo(alpha=0.01)
    model.set_params(**{'selection': "random", 'alpha': 0.1})
    model.fit(x, y)
    coef_test = model.coef_

    assert coef_before != coef_after
    assert coef_after == coef_test
