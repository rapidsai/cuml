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
import numpy as np
import cuml
import cuml.svm
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


def unit_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.unit)


def quality_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.quality)


def stress_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.stress)


@pytest.mark.parametrize('params', [
    {'kernel': 'linear', 'C': 1},
    {'kernel': 'linear', 'C': 10},
    {'kernel': 'rbf', 'C': 1, 'gamma': 1},
    {'kernel': 'rbf', 'C': 1, 'gamma': 'auto'},
    {'kernel': 'rbf', 'C': 0.1, 'gamma': 'auto'},
    {'kernel': 'rbf', 'C': 10, 'gamma': 'auto'},
    {'kernel': 'rbf', 'C': 1, 'gamma': 'scale'},
    {'kernel': 'poly', 'C': 1, 'gamma': 1},
    {'kernel': 'poly', 'C': 1, 'gamma': 'auto'},
    {'kernel': 'poly', 'C': 1, 'gamma': 'scale'},
    {'kernel': 'poly', 'C': 1, 'gamma': 'auto', 'degree': 2},
    {'kernel': 'poly', 'C': 1, 'gamma': 'auto', 'coef0': 1.37},
    {'kernel': 'sigmoid', 'C': 1, 'gamma': 'auto'},
    {'kernel': 'sigmoid', 'C': 1, 'gamma': 'scale', 'coef0': 0.42}
])
# @pytest.mark.parametrize('name', [unit_param(None), quality_param('iris')])
def test_svm_fit_predict(params, name='iris'):
    if name=='iris':
        iris = load_iris()
        X = iris.data
        y = iris.target
        y = (y>0).astype(X.dtype)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        # we create 40 separable points
        # TODO: tune the sensitivity of the test for this
        print('Unit test')
        np.random.seed(0)
        X = np.r_[np.random.randn(20, 2) - [2, 2],
                  np.random.randn(20, 2) + [2, 2]]
        y = np.array([0] * 20 + [1] * 20)

    cuSVC = cuml.svm.SVC(**params)
    cuSVC.fit(X, y)
    cu_y_hat = cuSVC.predict(X).to_array()
    cu_n_wrong = np.sum(np.abs(y - cu_y_hat))

    sklSVC = svm.SVC(**params)
    sklSVC.fit(X, y)
    skl_y_hat = sklSVC.predict(X)
    skl_n_wrong = np.sum(np.abs(y - skl_y_hat))

    n_support1 = np.sum(sklSVC.n_support_)
    n_support2 = np.sum(cuSVC.n_support_)
    n_sv_eps = max(2, n_support1*0.01)
    assert abs(n_support1-n_support2) <= n_sv_eps

    assert abs(sklSVC.intercept_-cuSVC.intercept_) <= 10*sklSVC.tol

    if params['kernel']=='linear':
        assert np.all(np.abs(sklSVC.coef_-cuSVC.coef_) <= sklSVC.tol)

    assert cu_n_wrong == skl_n_wrong

# TODO test different input types
# @pytest.mark.parametrize('x_datatype', [np.float32, np.float64])
# @pytest.mark.parametrize('y_datatype', [np.float32, np.float64, np.int32])
