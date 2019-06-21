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
import cuml
from cuml.test.utils import array_equal
import numpy as np
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
    UMAP=cuml.UMAP(),
    GaussianRandomProjection=cuml.GaussianRandomProjection(),
    SparseRandomProjection=cuml.SparseRandomProjection()
)

neighbor_models = dict(
    NearestNeighbors=cuml.NearestNeighbors()
)

dbscan_model = dict(
    DBSCAN=cuml.DBSCAN()
)


def unit_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.unit)


def quality_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.quality)


def stress_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.stress)


def pickle_save_load(tmpdir, model):
    pickle_file = tmpdir.join('cu_model.pickle')

    try:
        with open(pickle_file, 'wb') as pf:
            pickle.dump(model, pf)
    except (TypeError, ValueError) as e:
        pf.close()
        pytest.fail(e)

    with open(pickle_file, 'rb') as pf:
        cu_after_pickle_model = pickle.load(pf)

    return cu_after_pickle_model


def make_dataset(datatype, nrows, ncols):
    train_rows = np.int32(nrows*0.8)
    X, y = make_regression(n_samples=nrows, n_features=ncols,
                           random_state=0)
    X_test = np.asarray(X[train_rows:, :]).astype(datatype)
    X_train = np.asarray(X[:train_rows, :]).astype(datatype)
    y_train = np.asarray(y[:train_rows, ]).astype(datatype)

    return X_train, y_train, X_test


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('model', regression_models.values())
@pytest.mark.parametrize('nrows', [unit_param(20)])
@pytest.mark.parametrize('ncols', [unit_param(3)])
def test_regressor_pickle(tmpdir, datatype, model, nrows, ncols):
    X_train, y_train, X_test = make_dataset(datatype, nrows, ncols)

    model.fit(X_train, y_train)
    cu_before_pickle_predict = model.predict(X_test).to_array()

    cu_after_pickle_model = pickle_save_load(tmpdir, model)

    cu_after_pickle_predict = cu_after_pickle_model.predict(X_test).to_array()

    assert array_equal(cu_before_pickle_predict, cu_after_pickle_predict)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('model', solver_models.values())
@pytest.mark.parametrize('nrows', [unit_param(20)])
@pytest.mark.parametrize('ncols', [unit_param(3)])
def test_solver_pickle(tmpdir, datatype, model, nrows, ncols):
    X_train, y_train, X_test = make_dataset(datatype, nrows, ncols)

    model.fit(X_train, y_train)
    cu_before_pickle_predict = model.predict(X_test).to_array()

    cu_after_pickle_model = pickle_save_load(tmpdir, model)

    cu_after_pickle_predict = cu_after_pickle_model.predict(X_test).to_array()

    assert array_equal(cu_before_pickle_predict, cu_after_pickle_predict)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('model', cluster_models.values())
@pytest.mark.parametrize('nrows', [unit_param(20)])
@pytest.mark.parametrize('ncols', [unit_param(3)])
def test_cluster_pickle(tmpdir, datatype, model, nrows, ncols):
    X_train, _, X_test = make_dataset(datatype, nrows, ncols)

    model.fit(X_train)
    cu_before_pickle_predict = model.predict(X_test).to_array()

    cu_after_pickle_model = pickle_save_load(tmpdir, model)

    cu_after_pickle_predict = cu_after_pickle_model.predict(X_test).to_array()

    assert array_equal(cu_before_pickle_predict, cu_after_pickle_predict)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('model', decomposition_models.values())
@pytest.mark.parametrize('nrows', [unit_param(20)])
@pytest.mark.parametrize('ncols', [unit_param(3)])
@pytest.mark.xfail
def test_decomposition_pickle(tmpdir, datatype, model, nrows,
                              ncols):
    X_train, _, _ = make_dataset(datatype, nrows, ncols)

    cu_before_pickle_transform = model.fit_transform(X_train)

    cu_after_pickle_model = pickle_save_load(tmpdir, model)

    cu_after_pickle_transform = cu_after_pickle_model.transform(X_train)

    assert array_equal(cu_before_pickle_transform, cu_after_pickle_transform)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('model', neighbor_models.values())
@pytest.mark.parametrize('nrows', [unit_param(20)])
@pytest.mark.parametrize('ncols', [unit_param(3)])
@pytest.mark.parametrize('k', [unit_param(3)])
@pytest.mark.xfail
def test_neighbors_pickle(tmpdir, datatype, model, nrows,
                          ncols, k):
    X_train, _, X_test = make_dataset(datatype, nrows, ncols)

    model.fit(X_train)
    D_before, I_before = model.kneighbors(X_test, k=k)

    cu_after_pickle_model = pickle_save_load(tmpdir, model)

    D_after, I_after = cu_after_pickle_model.kneighbors(X_test, k=k)

    assert array_equal(D_before, D_after)
    assert array_equal(I_before, I_after)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('model', dbscan_model.values())
@pytest.mark.parametrize('nrows', [unit_param(20)])
@pytest.mark.parametrize('ncols', [unit_param(3)])
def test_dbscan_pickle(tmpdir, datatype, model, nrows, ncols):
    X_train, _, _ = make_dataset(datatype, nrows, ncols)

    cu_before_pickle_predict = model.fit_predict(X_train).to_array()

    cu_after_pickle_model = pickle_save_load(tmpdir, model)

    cu_after_pickle_predict = cu_after_pickle_model.fit_predict(
                              X_train
                              ).to_array()

    assert array_equal(cu_before_pickle_predict, cu_after_pickle_predict)
