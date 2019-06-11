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
from sklearn.datasets import make_regression, make_blobs
import pickle
import dill

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

def pickle_save_load(serializer_type, model):
    os.mkdir('tmp')

    if serializer_type == 'pickle':
        serializer = pickle
    else:
        serializer = dill

    pickle_file = 'tmp/cu_model'
    try:
        with open(pickle_file, 'wb') as pf:
            serializer.dump(model, pf)
    except (TypeError, ValueError) as e:
        pf.close()
        os.remove(pickle_file)
        os.rmdir('tmp')
        pytest.fail(e)

    with open(pickle_file, 'rb') as pf:
        cu_after_pickle_model = serializer.load(pf)

    os.remove(pickle_file)
    os.rmdir('tmp')

    return cu_after_pickle_model

@pytest.mark.parametrize('datatype', [np.float32])
@pytest.mark.parametrize('input_type', ['dataframe'])
@pytest.mark.parametrize('model', ['LinearRegression', 'Lasso', 'Ridge', 'ElasticNet'])
@pytest.mark.parametrize('model_objects', [regression_models])
@pytest.mark.parametrize('nrows', [unit_param(20)])
@pytest.mark.parametrize('ncols', [unit_param(3)])
@pytest.mark.parametrize('n_info', [unit_param(2)])
@pytest.mark.parametrize('serializer_type', ['pickle', 'dill'])
def test_regressor_pickle(datatype, input_type, model, model_objects, nrows, ncols, n_info,
                          serializer_type):
    train_rows = np.int32(nrows*0.8)
    X, y = make_regression(n_samples=(nrows), n_features=ncols,
                           n_informative=n_info, random_state=0)
    X_test = np.asarray(X[train_rows:, :]).astype(datatype)
    X_train = np.asarray(X[:train_rows, :]).astype(datatype)
    y_train = np.asarray(y[:train_rows, ]).astype(datatype)

    cu_before_pickle_model = model_objects[model]

    if input_type == 'dataframe':
        X_cudf = np_to_cudf(X_train)
        y_cudf = make_cudf_series(y_train)
        X_cudf_test = np_to_cudf(X_test)

        cu_before_pickle_model.fit(X_cudf, y_cudf)
        cu_before_pickle_predict = cu_before_pickle_model.predict(X_cudf_test).to_array()

    else:
        cu_before_pickle_model.fit(X_train, y_train)
        cu_before_pickle_predict = cu_before_pickle_model.predict(X_test).to_array()

    cu_after_pickle_model = pickle_save_load(serializer_type, cu_before_pickle_model)

    if input_type == 'dataframe':
        cu_after_pickle_predict = cu_after_pickle_model.predict(X_cudf_test).to_array()
    else:
        cu_after_pickle_predict = cu_after_pickle_model.predict(X_test).to_array()

    assert array_equal(cu_before_pickle_predict, cu_after_pickle_predict, with_sign=True)

@pytest.mark.parametrize('datatype', [np.float32])
@pytest.mark.parametrize('input_type', ['dataframe'])
@pytest.mark.parametrize('model', ['CD', 'SGD'])
@pytest.mark.parametrize('model_objects', [solver_models])
@pytest.mark.parametrize('nrows', [unit_param(20)])
@pytest.mark.parametrize('ncols', [unit_param(3)])
@pytest.mark.parametrize('n_info', [unit_param(2)])
@pytest.mark.parametrize('serializer_type', ['pickle', 'dill'])
def test_solver_pickle(datatype, input_type, model, model_objects, nrows, ncols, n_info,
                          serializer_type):
    train_rows = np.int32(nrows*0.8)
    X, y = make_regression(n_samples=(nrows), n_features=ncols,
                           n_informative=n_info, random_state=0)
    X_test = np.asarray(X[train_rows:, :]).astype(datatype)
    X_train = np.asarray(X[:train_rows, :]).astype(datatype)
    y_train = np.asarray(y[:train_rows, ]).astype(datatype)

    cu_before_pickle_model = model_objects[model]

    if input_type == 'dataframe':
        X_cudf = np_to_cudf(X_train)
        y_cudf = make_cudf_series(y_train)
        X_cudf_test = np_to_cudf(X_test)

        cu_before_pickle_model.fit(X_cudf, y_cudf)
        cu_before_pickle_predict = cu_before_pickle_model.predict(X_cudf_test).to_array()

    else:
        cu_before_pickle_model.fit(X_train, y_train)
        cu_before_pickle_predict = cu_before_pickle_model.predict(X_test).to_array()

    cu_after_pickle_model = pickle_save_load(serializer_type, cu_before_pickle_model)

    if input_type == 'dataframe':
        cu_after_pickle_predict = cu_after_pickle_model.predict(X_cudf_test).to_array()
    else:
        cu_after_pickle_predict = cu_after_pickle_model.predict(X_test).to_array()

    assert array_equal(cu_before_pickle_predict, cu_after_pickle_predict, with_sign=True)

@pytest.mark.parametrize('datatype', [np.float32])
@pytest.mark.parametrize('input_type', ['dataframe'])
@pytest.mark.parametrize('model', ['KMeans'])
@pytest.mark.parametrize('model_objects', [cluster_models])
@pytest.mark.parametrize('nrows', [unit_param(20)])
@pytest.mark.parametrize('ncols', [unit_param(3)])
@pytest.mark.parametrize('serializer_type', ['pickle', 'dill'])
@pytest.mark.xfail
def test_cluster_pickle(datatype, input_type, model, model_objects, nrows, ncols,
                        serializer_type):
    train_rows = np.int32(nrows*0.8)
    X, _ = make_blobs(n_samples=(nrows), n_features=ncols, random_state=0)
    X_test = np.asarray(X[train_rows:, :]).astype(datatype)
    X_train = np.asarray(X[:train_rows, :]).astype(datatype)

    cu_before_pickle_model = model_objects[model]

    if input_type == 'dataframe':
        X_cudf = np_to_cudf(X_train)
        X_cudf_test = np_to_cudf(X_test)

        cu_before_pickle_model.fit(X_cudf)
        cu_before_pickle_predict = cu_before_pickle_model.predict(X_cudf_test).to_array()

    else:
        cu_before_pickle_model.fit(X_train)
        cu_before_pickle_predict = cu_before_pickle_model.predict(X_test).to_array()

    cu_after_pickle_model = pickle_save_load(serializer_type, cu_before_pickle_model)

    if input_type == 'dataframe':
        cu_after_pickle_predict = cu_after_pickle_model.predict(X_cudf_test).to_array()
    else:
        cu_after_pickle_predict = cu_after_pickle_model.predict(X_test).to_array()

    assert array_equal(cu_before_pickle_predict, cu_after_pickle_predict, with_sign=True)

@pytest.mark.parametrize('datatype', [np.float32])
@pytest.mark.parametrize('input_type', ['dataframe'])
@pytest.mark.parametrize('model', ['PCA', 'TruncatedSVD', 'UMAP'])
@pytest.mark.parametrize('model_objects', [decomposition_models])
@pytest.mark.parametrize('nrows', [unit_param(20)])
@pytest.mark.parametrize('ncols', [unit_param(3)])
@pytest.mark.parametrize('serializer_type', ['pickle', 'dill'])
@pytest.mark.xfail
def test_decomposition_pickle(datatype, input_type, model, model_objects, nrows, ncols,
                        serializer_type):
    train_rows = np.int32(nrows*0.8)
    X, _ = make_blobs(n_samples=(nrows), n_features=ncols, random_state=0)

    cu_before_pickle_model = model_objects[model]

    if input_type == 'dataframe':
        X_cudf = np_to_cudf(X)
        cu_before_pickle_transform = cu_before_pickle_model.fit_transform(X_cudf)

    else:
        cu_before_pickle_transform = cu_before_pickle_model.fit_transform(X)

    cu_after_pickle_model = pickle_save_load(serializer_type, cu_before_pickle_model)

    if input_type == 'dataframe':
        cu_after_pickle_transform = cu_after_pickle_model.transform(X_cudf)
    else:
        cu_after_pickle_transform = cu_after_pickle_model.transform(X)

    for col_name in cu_before_pickle_transform.columns:
        assert array_equal(cu_before_pickle_transform[col_name].to_array(), cu_after_pickle_transform[col_name].to_array(), with_sign=True)

@pytest.mark.parametrize('datatype', [np.float32])
@pytest.mark.parametrize('input_type', ['dataframe'])
@pytest.mark.parametrize('model', ['NearestNeighbors'])
@pytest.mark.parametrize('model_objects', [neighbor_models])
@pytest.mark.parametrize('nrows', [unit_param(20)])
@pytest.mark.parametrize('ncols', [unit_param(3)])
@pytest.mark.parametrize('serializer_type', ['pickle', 'dill'])
@pytest.mark.parametrize('k', [unit_param(3)])
@pytest.mark.xfail
def test_neighbors_pickle(datatype, input_type, model, model_objects, nrows, ncols,
                        serializer_type, k):
    train_rows = np.int32(nrows*0.8)
    X, _ = make_blobs(n_samples=(nrows), n_features=ncols, random_state=0)
    X_test = np.asarray(X[train_rows:, :]).astype(datatype)
    X_train = np.asarray(X[:train_rows, :]).astype(datatype)

    cu_before_pickle_model = model_objects[model]

    if input_type == 'dataframe':
        X_cudf = np_to_cudf(X_train)
        X_cudf_test = np_to_cudf(X_test)

        cu_before_pickle_model.fit(X_cudf)
        cu_before_pickle_predict = cu_before_pickle_model.kneighbors(X_cudf_test, k=k)

    else:
        cu_before_pickle_model.fit(X_train)
        cu_before_pickle_predict = cu_before_pickle_model.kneighbors(X_test, k=k)

    cu_after_pickle_model = pickle_save_load(serializer_type, cu_before_pickle_model)

    if input_type == 'dataframe':
        cu_after_pickle_predict = cu_after_pickle_model.kneighbors(X_cudf_test, k=k)
    else:
        cu_after_pickle_predict = cu_after_pickle_model.kneighbors(X_test, k=k)

    for col_name in cu_before_pickle_transform.columns:
        assert array_equal(cu_before_pickle_transform[col_name].to_array(), cu_after_pickle_transform[col_name].to_array(), with_sign=True)
