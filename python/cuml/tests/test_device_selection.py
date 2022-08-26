# Copyright (c) 2022, NVIDIA CORPORATION.
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
import cuml
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression as skLinearRegression
from cuml.linear_model import LinearRegression
from cuml.common.device_selection import using_device_type, using_memory_type


@pytest.mark.parametrize('device_type', ['cpu', 'gpu', None])
def test_device_type(device_type):
    initial_device_type = cuml.global_settings.device_type
    with using_device_type(device_type):
        assert cuml.global_settings.device_type == device_type
    assert cuml.global_settings.device_type == initial_device_type


def test_device_type_exception():
    with pytest.raises(ValueError):
        with using_device_type('wrong_option'):
            assert True


@pytest.mark.parametrize('memory_type', ['global', 'host', 'managed',
                                         'mirror', None])
def test_memory_type(memory_type):
    initial_memory_type = cuml.global_settings.memory_type
    with using_memory_type(memory_type):
        assert cuml.global_settings.memory_type == memory_type
    assert cuml.global_settings.memory_type == initial_memory_type


def test_memory_type_exception():
    with pytest.raises(ValueError):
        with using_memory_type('wrong_option'):
            assert True


X, y = make_regression(n_samples=2000, n_features=20, n_informative=15)
X_train, X_test = X[:1800], X[1800:]
y_train, _ = y[:1800], y[1800:]


@pytest.mark.parametrize('fit_intercept', [False, True])
@pytest.mark.parametrize('normalize', [False, True])
def test_train_cpu_infer_cpu(fit_intercept, normalize):
    model = LinearRegression(fit_intercept=fit_intercept,
                             normalize=normalize)
    with using_device_type('cpu'):
        model.fit(X_train, y_train)
        cu_pred = model.predict(X_test)

    sk_model = skLinearRegression(fit_intercept=fit_intercept,
                                  normalize=normalize)
    sk_model.fit(X_train, y_train)
    sk_pred = sk_model.predict(X_test)
    np.testing.assert_allclose(sk_pred, cu_pred)


@pytest.mark.parametrize('fit_intercept', [False, True])
@pytest.mark.parametrize('normalize', [False, True])
def test_train_gpu_infer_cpu(fit_intercept, normalize):
    model = LinearRegression(fit_intercept=fit_intercept,
                             normalize=normalize)
    with using_device_type('gpu'):
        model.fit(X_train, y_train)
    with using_device_type('cpu'):
        cu_pred = model.predict(X_test)

    sk_model = skLinearRegression(fit_intercept=fit_intercept,
                                  normalize=normalize)
    sk_model.fit(X_train, y_train)
    sk_pred = sk_model.predict(X_test)
    np.testing.assert_allclose(sk_pred, cu_pred)


@pytest.mark.parametrize('fit_intercept', [False, True])
@pytest.mark.parametrize('normalize', [False, True])
def test_train_cpu_infer_gpu(fit_intercept, normalize):
    model = LinearRegression(fit_intercept=fit_intercept,
                             normalize=normalize)
    with using_device_type('cpu'):
        model.fit(X_train, y_train)
    with using_device_type('gpu'):
        cu_pred = model.predict(X_test)

    sk_model = skLinearRegression(fit_intercept=fit_intercept,
                                  normalize=normalize)
    sk_model.fit(X_train, y_train)
    sk_pred = sk_model.predict(X_test)
    np.testing.assert_allclose(sk_pred, cu_pred)
