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
import os
import cuml
from cuml.test.utils import array_equal, np_to_cudf
import cudf
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
import pickle

regression_models = dict(
    LinearRegression=cuml.LinearRegression(),
    Lasso=cuml.Lasso(),
    Ridge=cuml.Ridge(),
    ElasticNet=cuml.ElasticNet()
)

solver_models = dict(
    CD=cuml.CD(),
    SGD=cuml.SGD(eta0=0.005)
)

cluster_models = dict(
    KMeans=cuml.KMeans()
)

decomposition_models = dict(
    PCA=cuml.PCA(),
    TruncatedSVD=cuml.TruncatedSVD(),
    UMAP=cuml.UMAP()
)

neighbor_models = dict(
    NearestNeighbors=cuml.NearestNeighbors()
)

def unit_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.unit)


def quality_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.quality)


def stress_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.stress)

def make_cudf_series(arr):
    df = pd.DataFrame(
                           {'fea0': arr[:, ]})
    df = df.values
    df = df[:, 0]
    return cudf.Series(df)

def pickle_save_load(model):
    os.mkdir('tmp')

    pickle_file = 'tmp/cu_model'
    try:
        with open(pickle_file, 'wb') as pf:
            pickle.dump(model, pf)
    except (TypeError, ValueError) as e:
        pf.close()
        os.remove(pickle_file)
        os.rmdir('tmp')
        pytest.fail(e)

    with open(pickle_file, 'rb') as pf:
        cu_after_pickle_model = pickle.load(pf)

    os.remove(pickle_file)
    os.rmdir('tmp')

    return cu_after_pickle_model

def make_dataset(datatype, input_type, nrows, ncols):
    train_rows = np.int32(nrows*0.8)
    X, y = make_regression(n_samples=(nrows), n_features=ncols,
                           random_state=0)
    X_test = np.asarray(X[train_rows:, :]).astype(datatype)
    X_train = np.asarray(X[:train_rows, :]).astype(datatype)
    y_train = np.asarray(y[:train_rows, ]).astype(datatype)

    if input_type == 'dataframe':
        X_train = np_to_cudf(X_train)
        y_train = make_cudf_series(y_train)
        X_test = np_to_cudf(X_test)

    return X_train, y_train, X_test

@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
@pytest.mark.parametrize('model', regression_models.values())
@pytest.mark.parametrize('nrows', [unit_param(20)])
@pytest.mark.parametrize('ncols', [unit_param(3)])
def test_regressor_pickle(datatype, input_type, model, nrows, ncols,):
    X_train, y_train, X_test = make_dataset(datatype, input_type, nrows, ncols)

    model.fit(X_train, y_train)
    cu_before_pickle_predict = model.predict(X_test).to_array()

    cu_after_pickle_model = pickle_save_load(model)

    cu_after_pickle_predict = cu_after_pickle_model.predict(X_test).to_array()

    assert array_equal(cu_before_pickle_predict, cu_after_pickle_predict, with_sign=True)

@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
@pytest.mark.parametrize('model', solver_models.values())
@pytest.mark.parametrize('nrows', [unit_param(20)])
@pytest.mark.parametrize('ncols', [unit_param(3)])
def test_solver_pickle(datatype, input_type, model, nrows, ncols):
    X_train, y_train, X_test = make_dataset(datatype, input_type, nrows, ncols)

    model.fit(X_train, y_train)
    cu_before_pickle_predict = model.predict(X_test).to_array()

    cu_after_pickle_model = pickle_save_load(model)

    cu_after_pickle_predict = cu_after_pickle_model.predict(X_test).to_array()

    assert array_equal(cu_before_pickle_predict, cu_after_pickle_predict, with_sign=True)

@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
@pytest.mark.parametrize('model', cluster_models.values())
@pytest.mark.parametrize('nrows', [unit_param(20)])
@pytest.mark.parametrize('ncols', [unit_param(3)])
def test_cluster_pickle(datatype, input_type, model, nrows, ncols):
    X_train, _, X_test = make_dataset(datatype, input_type, nrows, ncols)

    model.fit(X_train)
    cu_before_pickle_predict = model.predict(X_test).to_array()

    cu_after_pickle_model = pickle_save_load(model)

    cu_after_pickle_predict = cu_after_pickle_model.predict(X_test).to_array()

    assert array_equal(cu_before_pickle_predict, cu_after_pickle_predict, with_sign=True)

@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
@pytest.mark.parametrize('model', decomposition_models.values())
@pytest.mark.parametrize('nrows', [unit_param(20)])
@pytest.mark.parametrize('ncols', [unit_param(3)])
@pytest.mark.xfail
def test_decomposition_pickle(datatype, input_type, model, nrows, ncols):
    X_train, _, _ = make_dataset(datatype, input_type, nrows, ncols)

    cu_before_pickle_transform = model.fit_transform(X_train)

    cu_after_pickle_model = pickle_save_load(model)

    cu_after_pickle_transform = cu_after_pickle_model.transform(X_train)

    for col in cu_before_pickle_transform.columns:
        assert array_equal(cu_before_pickle_transform[col].to_array(), cu_after_pickle_transform[col].to_array(), with_sign=True)

@pytest.mark.parametrize('datatype', [np.float32])
@pytest.mark.parametrize('input_type', ['dataframe'])
@pytest.mark.parametrize('model', neighbor_models.values())
@pytest.mark.parametrize('nrows', [unit_param(20)])
@pytest.mark.parametrize('ncols', [unit_param(3)])
@pytest.mark.parametrize('k', [unit_param(3)])
@pytest.mark.xfail
def test_neighbors_pickle(datatype, input_type, model, nrows, ncols, k):
    X_train, _, X_test = make_dataset(datatype, input_type, nrows, ncols)

    model.fit(X_train)
    cu_before_pickle_predict = cu_before_pickle_model.kneighbors(X_test, k=k)

    cu_after_pickle_model = pickle_save_load(model)

    cu_after_pickle_predict = cu_after_pickle_model.kneighbors(X_train)

    for col in cu_before_pickle_transform.columns:
        assert array_equal(cu_before_pickle_predict[col].to_array(), cu_after_pickle_predict[col].to_array(), with_sign=True)