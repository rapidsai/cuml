# Copyright (c) 2018-2019, NVIDIA CORPORATION.
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
import pytest

from cuml.solvers import SGD as cumlSGD
from cuml.test.utils import unit_param, quality_param, \
    stress_param

from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from sklearn import datasets


@pytest.mark.parametrize('lrate', ['constant', 'invscaling', 'adaptive'])
@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('penalty', ['none', 'l1', 'l2', 'elasticnet'])
@pytest.mark.parametrize('loss', ['hinge', 'log', 'squared_loss'])
@pytest.mark.parametrize('name', [unit_param(None), quality_param('iris'),
                         stress_param('blobs')])
def test_svd(datatype, lrate, penalty, loss, name):

    if name == 'blobs':
        X, y = make_blobs(n_samples=500000,
                          n_features=1000, random_state=0)
        X = X.astype(datatype)
        y = y.astype(datatype)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            train_size=0.8)

    elif name == 'iris':
        iris = datasets.load_iris()
        X = (iris.data).astype(datatype)
        y = (iris.target).astype(datatype)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            train_size=0.8)

    else:
        X_train = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]],
                           dtype=datatype)
        y_train = np.array([1, 1, 2, 2], dtype=datatype)
        X_test = np.array([[3.0, 5.0], [2.0, 5.0]]).astype(datatype)

    cu_sgd = cumlSGD(learning_rate=lrate, eta0=0.005, epochs=2000,
                     fit_intercept=True, batch_size=4096,
                     tol=0.0, penalty=penalty, loss=loss)

    cu_sgd.fit(X_train, y_train)
    cu_pred = cu_sgd.predict(X_test).to_array()
    print("cuML predictions : ", cu_pred)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
def test_svd_default(datatype):

    X_train = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]],
                       dtype=datatype)
    y_train = np.array([1, 1, 2, 2], dtype=datatype)
    X_test = np.array([[3.0, 5.0], [2.0, 5.0]]).astype(datatype)

    cu_sgd = cumlSGD()

    cu_sgd.fit(X_train, y_train)
    cu_pred = cu_sgd.predict(X_test).to_array()
    print("cuML predictions : ", cu_pred)
