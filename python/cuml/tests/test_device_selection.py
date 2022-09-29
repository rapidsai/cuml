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


import itertools as it
import pytest
import cuml
import numpy as np
import pandas as pd
import cudf
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression as skLinearRegression
from cuml.linear_model import LinearRegression
from cuml.testing.test_preproc_utils import to_output_type
from cuml.common.device_selection import DeviceType, using_device_type
from cuml.common.device_selection import MemoryType, using_memory_type
import pickle


@pytest.mark.parametrize('input', [('cpu', DeviceType.host),
                                   ('gpu', DeviceType.device)])
def test_device_type(input):
    initial_device_type = cuml.global_settings.device_type
    with using_device_type(input[0]):
        assert cuml.global_settings.device_type == input[1]
    assert cuml.global_settings.device_type == initial_device_type


def test_device_type_exception():
    with pytest.raises(ValueError):
        with using_device_type('wrong_option'):
            assert True


@pytest.mark.parametrize('input', [('device', MemoryType.device),
                                   ('host', MemoryType.host),
                                   ('managed', MemoryType.managed),
                                   ('mirror', MemoryType.mirror)])
def test_memory_type(input):
    initial_memory_type = cuml.global_settings.memory_type
    with using_memory_type(input[0]):
        assert cuml.global_settings.memory_type == input[1]
    assert cuml.global_settings.memory_type == initial_memory_type


def test_memory_type_exception():
    with pytest.raises(ValueError):
        with using_memory_type('wrong_option'):
            assert True


X, y = make_regression(n_samples=2000, n_features=20, n_informative=15)
X_train, X_test = X[:1800], X[1800:]
y_train, _ = y[:1800], y[1800:]


def fixture_generation_helper(params):
    param_names = sorted(params)
    param_combis = list(it.product(*(params[param_name]
                                     for param_name in param_names)))
    ids = ['-'.join(map(str, param_combi)) for param_combi in param_combis]
    param_combis = [dict(zip(param_names, param_combi))
                    for param_combi in param_combis]
    return {
        'scope': 'session',
        'params': param_combis,
        'ids': ids
    }


@pytest.fixture(**fixture_generation_helper({
                    'input_type': ['numpy', 'dataframe', 'cupy',
                                   'cudf', 'numba'],
                    'fit_intercept': [False, True],
                    'normalize': [False, True]
                }))
def lr_data(request):
    sk_model = skLinearRegression(fit_intercept=request.param['fit_intercept'],
                                  normalize=request.param['normalize'])
    sk_model.fit(X_train, y_train)

    input_type = request.param['input_type']

    if input_type == 'dataframe':
        modified_y_train = pd.Series(y_train)
    elif input_type == 'cudf':
        modified_y_train = cudf.Series(y_train)
    else:
        modified_y_train = to_output_type(y_train, input_type)

    return {
        'fit_intercept': request.param['fit_intercept'],
        'normalize': request.param['normalize'],
        'X_train': to_output_type(X_train, input_type),
        'y_train': modified_y_train,
        'X_test': to_output_type(X_test, input_type),
        'ref_y_test': sk_model.predict(X_test)
    }


def test_train_cpu_infer_cpu(lr_data):
    model = LinearRegression(fit_intercept=lr_data['fit_intercept'],
                             normalize=lr_data['normalize'])
    with using_device_type('cpu'):
        model.fit(lr_data['X_train'], lr_data['y_train'])
        cu_pred = model.predict(lr_data['X_test'])

    cu_pred = to_output_type(cu_pred, 'numpy').flatten()
    np.testing.assert_allclose(lr_data['ref_y_test'], cu_pred)


def test_train_gpu_infer_cpu(lr_data):
    model = LinearRegression(fit_intercept=lr_data['fit_intercept'],
                             normalize=lr_data['normalize'])
    with using_device_type('gpu'):
        model.fit(lr_data['X_train'], lr_data['y_train'])
    with using_device_type('cpu'):
        cu_pred = model.predict(lr_data['X_test'])

    cu_pred = to_output_type(cu_pred, 'numpy').flatten()
    np.testing.assert_allclose(lr_data['ref_y_test'], cu_pred)


def test_train_cpu_infer_gpu(lr_data):
    model = LinearRegression(fit_intercept=lr_data['fit_intercept'],
                             normalize=lr_data['normalize'])
    with using_device_type('cpu'):
        model.fit(lr_data['X_train'], lr_data['y_train'])
    with using_device_type('gpu'):
        cu_pred = model.predict(lr_data['X_test'])

    cu_pred = to_output_type(cu_pred, 'numpy').flatten()
    np.testing.assert_allclose(lr_data['ref_y_test'], cu_pred)


def test_train_gpu_infer_gpu(lr_data):
    model = LinearRegression(fit_intercept=lr_data['fit_intercept'],
                             normalize=lr_data['normalize'])
    with using_device_type('gpu'):
        model.fit(lr_data['X_train'], lr_data['y_train'])
        cu_pred = model.predict(lr_data['X_test'])

    cu_pred = to_output_type(cu_pred, 'numpy').flatten()
    np.testing.assert_allclose(lr_data['ref_y_test'], cu_pred)
    sk_model = skLinearRegression(fit_intercept=lr_data['fit_intercept'],
                                  normalize=lr_data['normalize'])
    sk_model.fit(X_train, y_train)
    sk_pred = sk_model.predict(X_test)
    np.testing.assert_allclose(sk_pred, cu_pred)


@pytest.mark.parametrize('fit_intercept', [False, True])
@pytest.mark.parametrize('normalize', [False, True])
def test_pickle_interop(fit_intercept, normalize):
    pickle_filepath = '/tmp/model.pickle'

    model = LinearRegression(fit_intercept=fit_intercept,
                             normalize=normalize)
    with using_device_type('gpu'):
        model.fit(X_train, y_train)

    with open(pickle_filepath, 'wb') as pf:
        pickle.dump(model, pf)

    del model

    with open(pickle_filepath, 'rb') as pf:
        pickled_model = pickle.load(pf)

    with using_device_type('cpu'):
        cu_pred = pickled_model.predict(X_test)

        sk_model = skLinearRegression(fit_intercept=fit_intercept,
                                      normalize=normalize)
    sk_model.fit(X_train, y_train)
    sk_pred = sk_model.predict(X_test)
    np.testing.assert_allclose(sk_pred, cu_pred)
