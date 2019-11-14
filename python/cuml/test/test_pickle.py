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

import cuml
import numpy as np
import pickle
import pytest

from cuml.test.utils import array_equal, unit_param, stress_param
from cuml.test.test_svm import compare_svm

from sklearn.datasets import load_iris
from sklearn.datasets import make_regression
from sklearn.manifold.t_sne import trustworthiness
from sklearn.model_selection import train_test_split


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
)


decomposition_models_xfail = dict(
    GaussianRandomProjection=cuml.GaussianRandomProjection(),
    SparseRandomProjection=cuml.SparseRandomProjection()
)

neighbor_models = dict(
    NearestNeighbors=cuml.NearestNeighbors()
)

dbscan_model = dict(
    DBSCAN=cuml.DBSCAN()
)

umap_model = dict(
    UMAP=cuml.UMAP()
)


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


def make_dataset(datatype, nrows, ncols, n_info):
    X, y = make_regression(n_samples=nrows, n_features=ncols,
                           n_informative=n_info, random_state=0)
    X = X.astype(datatype)
    y = y.astype(datatype)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    return X_train, y_train, X_test


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('model', regression_models.values())
@pytest.mark.parametrize('data_size', [unit_param([500, 20, 10]),
                         stress_param([500000, 1000, 500])])
def test_regressor_pickle(tmpdir, datatype, model, data_size):
    nrows, ncols, n_info = data_size
    X_train, y_train, X_test = make_dataset(datatype, nrows, ncols, n_info)

    model.fit(X_train, y_train)
    cu_before_pickle_predict = model.predict(X_test).to_array()

    cu_after_pickle_model = pickle_save_load(tmpdir, model)

    del model

    cu_after_pickle_predict = cu_after_pickle_model.predict(X_test).to_array()

    assert array_equal(cu_before_pickle_predict, cu_after_pickle_predict)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('model', solver_models.values())
@pytest.mark.parametrize('data_size', [unit_param([500, 20, 10]),
                         stress_param([500000, 1000, 500])])
def test_solver_pickle(tmpdir, datatype, model, data_size):
    nrows, ncols, n_info = data_size
    X_train, y_train, X_test = make_dataset(datatype, nrows, ncols, n_info)

    model.fit(X_train, y_train)
    cu_before_pickle_predict = model.predict(X_test).to_array()

    cu_after_pickle_model = pickle_save_load(tmpdir, model)

    del model

    cu_after_pickle_predict = cu_after_pickle_model.predict(X_test).to_array()

    assert array_equal(cu_before_pickle_predict, cu_after_pickle_predict)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('model', cluster_models.values())
@pytest.mark.parametrize('data_size', [unit_param([500, 20, 10]),
                         stress_param([500000, 1000, 500])])
def test_cluster_pickle(tmpdir, datatype, model, data_size):
    nrows, ncols, n_info = data_size
    X_train, _, X_test = make_dataset(datatype, nrows, ncols, n_info)

    model.fit(X_train)
    cu_before_pickle_predict = model.predict(X_test).to_array()

    cu_after_pickle_model = pickle_save_load(tmpdir, model)

    del model

    cu_after_pickle_predict = cu_after_pickle_model.predict(X_test).to_array()

    assert array_equal(cu_before_pickle_predict, cu_after_pickle_predict)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('model', decomposition_models_xfail.values())
@pytest.mark.parametrize('data_size', [unit_param([500, 20, 10]),
                         stress_param([500000, 1000, 500])])
@pytest.mark.xfail
def test_decomposition_pickle(tmpdir, datatype, model, data_size):
    nrows, ncols, n_info = data_size
    X_train, _, _ = make_dataset(datatype, nrows, ncols, n_info)

    cu_before_pickle_transform = model.fit_transform(X_train)

    cu_after_pickle_model = pickle_save_load(tmpdir, model)

    del model

    cu_after_pickle_transform = cu_after_pickle_model.transform(X_train)

    assert array_equal(cu_before_pickle_transform, cu_after_pickle_transform)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('model', umap_model.values())
def test_umap_pickle(tmpdir, datatype, model):

    iris = load_iris()
    iris_selection = np.random.RandomState(42).choice(
        [True, False], 150, replace=True, p=[0.75, 0.25])
    X_train = iris.data[iris_selection]

    cu_before_pickle_transform = model.fit_transform(X_train)

    cu_before_embed = model.arr_embed

    cu_trust_before = trustworthiness(X_train,
                                      cu_before_pickle_transform,
                                      10)

    cu_after_pickle_model = pickle_save_load(tmpdir, model)

    del model

    cu_after_pickle_transform = cu_after_pickle_model.transform(X_train)

    cu_after_embed = cu_after_pickle_model.arr_embed

    cu_trust_after = trustworthiness(X_train, cu_after_pickle_transform,
                                     10)

    assert array_equal(cu_before_embed[0][0], cu_after_embed[0][0])
    assert cu_trust_after >= cu_trust_before - 0.2


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('model', decomposition_models.values())
@pytest.mark.parametrize('data_size', [unit_param([500, 20, 10]),
                         stress_param([500000, 1000, 500])])
@pytest.mark.xfail
def test_decomposition_pickle_xfail(tmpdir, datatype, model, data_size):
    nrows, ncols, n_info = data_size
    X_train, _, _ = make_dataset(datatype, nrows, ncols, n_info)

    cu_before_pickle_transform = model.fit_transform(X_train)

    cu_after_pickle_model = pickle_save_load(tmpdir, model)

    del model

    cu_after_pickle_transform = cu_after_pickle_model.transform(X_train)

    assert array_equal(cu_before_pickle_transform, cu_after_pickle_transform)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('model', neighbor_models.values())
@pytest.mark.parametrize('data_info', [unit_param([500, 20, 10, 5]),
                         stress_param([500000, 1000, 500, 50])])
def test_neighbors_pickle(tmpdir, datatype, model, data_info):
    nrows, ncols, n_info, k = data_info
    X_train, _, X_test = make_dataset(datatype, nrows, ncols, n_info)

    model.fit(X_train)
    D_before, I_before = model.kneighbors(X_test, k=k)

    cu_after_pickle_model = pickle_save_load(tmpdir, model)

    del model

    D_after, I_after = cu_after_pickle_model.kneighbors(X_test, k=k)

    assert array_equal(D_before, D_after)
    assert array_equal(I_before, I_after)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('data_info', [unit_param([500, 20, 10, 5]),
                         stress_param([500000, 1000, 500, 50])])
def test_neighbors_pickle_nofit(tmpdir, datatype, data_info):

    """
    Note: This test digs down a bit far into the
    internals of the implementation, but it's
    important that regressions do not occur
    from changes to the class.
    """
    nrows, ncols, n_info, k = data_info
    model = cuml.neighbors.NearestNeighbors()

    unpickled = pickle_save_load(tmpdir, model)

    state = unpickled.__dict__

    print(str(state))

    assert state["n_indices"] == 0
    assert "X_m" not in state
    assert state["sizes"] is None
    assert state["input"] is None

    X_train, _, X_test = make_dataset(datatype, nrows, ncols, n_info)

    model.fit(X_train)

    unpickled = pickle_save_load(tmpdir, model)

    del model

    state = unpickled.__dict__

    assert state["n_indices"] == 1
    assert "X_m" in state
    assert state["sizes"] is not None
    assert state["input"] is not None


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.xfail(strict=True)
def test_neighbors_mg_fails(tmpdir, datatype):

    model = cuml.neighbors.NearestNeighbors()
    model.n_indices = 2

    pickle_save_load(tmpdir, model)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('model', dbscan_model.values())
@pytest.mark.parametrize('data_size', [unit_param([500, 20, 10]),
                         stress_param([500000, 1000, 500])])
def test_dbscan_pickle(tmpdir, datatype, model, data_size):
    nrows, ncols, n_info = data_size
    X_train, _, _ = make_dataset(datatype, nrows, ncols, n_info)

    cu_before_pickle_predict = model.fit_predict(X_train).to_array()

    cu_after_pickle_model = pickle_save_load(tmpdir, model)

    del model

    cu_after_pickle_predict = cu_after_pickle_model.fit_predict(
                              X_train
                              ).to_array()

    assert array_equal(cu_before_pickle_predict, cu_after_pickle_predict)


def test_tsne_pickle(tmpdir):
    iris = load_iris()
    iris_selection = np.random.RandomState(42).choice(
        [True, False], 150, replace=True, p=[0.75, 0.25])
    X = iris.data[iris_selection]

    model = cuml.manifold.TSNE(n_components=2, random_state=199)

    # Pickle the model
    model_pickle = pickle_save_load(tmpdir, model)

    del model

    model_params = model_pickle.__dict__
    if "handle" in model_params:
        del model_params["handle"]

    # Confirm params in model are identical
    new_keys = set(model_params.keys())
    for key, value in zip(model_params.keys(), model_params.values()):
        assert (model_params[key] == value)
        new_keys -= set([key])

    # Check all keys have been checked
    assert(len(new_keys) == 0)

    # Transform data
    model_pickle.fit(X)
    trust_before = trustworthiness(X, model_pickle.Y, 10)

    # Save model + embeddings
    model_after_pickling = pickle_save_load(tmpdir, model_pickle)

    del model_pickle

    trust_after = trustworthiness(X, model_after_pickling.Y.to_pandas(), 10)

    assert trust_before == trust_after


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
def test_svm_pickle(tmpdir, datatype):

    model = cuml.svm.SVC()
    iris = load_iris()
    iris_selection = np.random.RandomState(42).choice(
        [True, False], 150, replace=True, p=[0.75, 0.25])
    X_train = iris.data[iris_selection]
    y_train = iris.target[iris_selection]
    y_train = (y_train > 0).astype(datatype)

    model.fit(X_train, y_train)
    model_pickle = pickle_save_load(tmpdir, model)
    compare_svm(model, model_pickle, X_train, y_train, cmp_sv=0,
                dcoef_tol=0)
