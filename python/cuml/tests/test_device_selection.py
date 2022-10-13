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
import pickle

from pytest_cases import fixture_union, pytest_fixture_plus
from sklearn.datasets import make_regression, make_blobs
from sklearn.manifold import trustworthiness

from cuml.testing.test_preproc_utils import to_output_type
from cuml.common.device_selection import DeviceType, using_device_type
from cuml.common.memory_utils import MemoryType, using_memory_type

from sklearn.linear_model import LinearRegression as skLinearRegression
from cuml.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression as skLogisticRegression
from cuml.linear_model import LogisticRegression
from umap import UMAP as refUMAP
from cuml.manifold import UMAP


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


def make_reg_dataset():
    X, y = make_regression(n_samples=2000, n_features=20,
                           n_informative=18, random_state=0)
    X_train, X_test = X[:1800], X[1800:]
    y_train, _ = y[:1800], y[1800:]
    return X_train, y_train, X_test


def make_blob_dataset():
    X, y = make_blobs(n_samples=2000, n_features=20, centers=20)
    X_train, X_test = X[:1800], X[1800:]
    y_train, _ = y[:1800], y[1800:]
    return X_train, y_train, X_test


X_train_reg, y_train_reg, X_test_reg = make_reg_dataset()
X_train_blob, y_train_blob, X_test_blob = make_reg_dataset()


def check_trustworthiness(cuml_output, test_data):
    input = to_output_type(test_data['X_test'], 'numpy')
    embeddings = to_output_type(cuml_output, 'numpy')
    n_neighbors = test_data['kwargs']['n_neighbors']
    trust = trustworthiness(input, embeddings, n_neighbors)
    assert trust >= 0.5


def check_allclose(cuml_output, test_data):
    ref_output = to_output_type(test_data['ref_y_test'], 'numpy')
    cuml_output = to_output_type(cuml_output, 'numpy')
    np.testing.assert_allclose(ref_output, cuml_output)


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


@pytest_fixture_plus(**fixture_generation_helper({
                    'input_type': ['numpy', 'dataframe', 'cupy',
                                   'cudf', 'numba'],
                    'fit_intercept': [False, True],
                    'normalize': [False, True]
                }))
def linreg_test_data(request):
    kwargs = {
        'fit_intercept': request.param['fit_intercept'],
        'normalize': request.param['normalize'],
    }

    sk_model = skLinearRegression(**kwargs)
    sk_model.fit(X_train_reg, y_train_reg)

    input_type = request.param['input_type']

    if input_type == 'dataframe':
        modified_y_train = pd.Series(y_train_reg)
    elif input_type == 'cudf':
        modified_y_train = cudf.Series(y_train_reg)
    else:
        modified_y_train = to_output_type(y_train_reg, input_type)

    return {
        'cuEstimator': LinearRegression,
        'kwargs': kwargs,
        'infer_func': 'predict',
        'assert_func': check_allclose,
        'X_train': to_output_type(X_train_reg, input_type),
        'y_train': modified_y_train,
        'X_test': to_output_type(X_test_reg, input_type),
        'ref_y_test': sk_model.predict(X_test_reg)
    }


@pytest_fixture_plus(**fixture_generation_helper({
                    'input_type': ['numpy', 'dataframe', 'cupy',
                                   'cudf', 'numba'],
                    'penalty': ['none', 'l2'],
                    'fit_intercept': [False, True]
                }))
def logreg_test_data(request):
    kwargs = {
        'penalty': request.param['penalty'],
        'fit_intercept': request.param['fit_intercept']
    }

    above_average = y_train_reg > np.median(y_train_reg)
    modified_y_train = y_train_reg
    modified_y_train[above_average] = 1
    modified_y_train[~above_average] = 0

    sk_model = skLogisticRegression(**kwargs)
    sk_model.fit(X_train_reg, modified_y_train)

    input_type = request.param['input_type']

    if input_type == 'dataframe':
        modified_y_train = pd.Series(modified_y_train)
    elif input_type == 'cudf':
        modified_y_train = cudf.Series(modified_y_train)
    else:
        modified_y_train = to_output_type(modified_y_train, input_type)

    return {
        'cuEstimator': LogisticRegression,
        'kwargs': kwargs,
        'infer_func': 'predict',
        'assert_func': check_allclose,
        'X_train': to_output_type(X_train_reg, input_type),
        'y_train': modified_y_train,
        'X_test': to_output_type(X_test_reg, input_type),
        'ref_y_test': sk_model.predict(X_test_reg)
    }


@pytest_fixture_plus(**fixture_generation_helper({
                    'input_type': ['cupy'],
                    'n_components': [2, 16],
                    'init': ['spectral', 'random']
                }))
def umap_test_data(request):
    kwargs = {
        'n_neighbors': 12,
        'n_components': request.param['n_components'],
        'init': request.param['init'],
        'random_state': 42
    }

    ref_model = refUMAP(**kwargs)
    ref_model.fit(X_train_blob, y_train_blob)

    input_type = request.param['input_type']

    if input_type == 'dataframe':
        modified_y_train = pd.Series(y_train_blob)
    elif input_type == 'cudf':
        modified_y_train = cudf.Series(y_train_blob)
    else:
        modified_y_train = to_output_type(y_train_blob, input_type)

    return {
        'cuEstimator': UMAP,
        'kwargs': kwargs,
        'infer_func': 'transform',
        'assert_func': check_trustworthiness,
        'X_train': to_output_type(X_train_blob, input_type),
        'y_train': modified_y_train,
        'X_test': to_output_type(X_test_blob, input_type),
        'ref_y_test': ref_model.transform(X_test_blob)
    }


fixture_union('test_data', ['linreg_test_data',
                            'logreg_test_data',
                            'umap_test_data'])


def test_train_cpu_infer_cpu(test_data):
    cuEstimator = test_data['cuEstimator']
    model = cuEstimator(**test_data['kwargs'])
    with using_device_type('cpu'):
        model.fit(test_data['X_train'], test_data['y_train'])
        infer_func = getattr(model, test_data['infer_func'])
        cuml_output = infer_func(test_data['X_test'])

    assert_func = test_data['assert_func']
    assert_func(cuml_output, test_data)


def test_train_gpu_infer_cpu(test_data):
    cuEstimator = test_data['cuEstimator']
    model = cuEstimator(**test_data['kwargs'])
    with using_device_type('gpu'):
        model.fit(test_data['X_train'], test_data['y_train'])
    with using_device_type('cpu'):
        infer_func = getattr(model, test_data['infer_func'])
        cuml_output = infer_func(test_data['X_test'])

    assert_func = test_data['assert_func']
    assert_func(cuml_output, test_data)


def test_train_cpu_infer_gpu(test_data):
    cuEstimator = test_data['cuEstimator']
    model = cuEstimator(**test_data['kwargs'])
    with using_device_type('cpu'):
        model.fit(test_data['X_train'], test_data['y_train'])
    with using_device_type('gpu'):
        infer_func = getattr(model, test_data['infer_func'])
        cuml_output = infer_func(test_data['X_test'])

    assert_func = test_data['assert_func']
    assert_func(cuml_output, test_data)


def test_train_gpu_infer_gpu(test_data):
    cuEstimator = test_data['cuEstimator']
    model = cuEstimator(**test_data['kwargs'])
    with using_device_type('gpu'):
        model.fit(test_data['X_train'], test_data['y_train'])
        infer_func = getattr(model, test_data['infer_func'])
        cuml_output = infer_func(test_data['X_test'])

    assert_func = test_data['assert_func']
    assert_func(cuml_output, test_data)


def test_pickle_interop(test_data):
    pickle_filepath = '/tmp/model.pickle'

    cuEstimator = test_data['cuEstimator']
    model = cuEstimator(**test_data['kwargs'])
    with using_device_type('gpu'):
        model.fit(test_data['X_train'], test_data['y_train'])

    with open(pickle_filepath, 'wb') as pf:
        pickle.dump(model, pf)

    del model

    with open(pickle_filepath, 'rb') as pf:
        pickled_model = pickle.load(pf)

    with using_device_type('cpu'):
        infer_func = getattr(pickled_model, test_data['infer_func'])
        cuml_output = infer_func(test_data['X_test'])

    cuml_output = to_output_type(cuml_output, 'numpy').flatten()
    assert_func = test_data['assert_func']
    assert_func(cuml_output, test_data)


@pytest.mark.parametrize('estimator', [LinearRegression,
                                       LogisticRegression,
                                       UMAP])
def test_hyperparams_defaults(estimator):
    model = estimator()
    model.check_hyperparams_defaults()
