# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
from sklearn.base import clone
from sklearn.datasets import load_iris, make_classification,\
    make_regression
from sklearn.manifold.t_sne import trustworthiness
from sklearn.model_selection import train_test_split


regression_models = {
    "LinearRegression": lambda fit_intercept=True: cuml.LinearRegression(
        fit_intercept=fit_intercept),
    "LogisticRegression": lambda fit_intercept=True: cuml.LogisticRegression(
        fit_intercept=fit_intercept),
    "Lasso": lambda fit_intercept=True: cuml.Lasso(
        fit_intercept=fit_intercept),
    "Ridge": lambda fit_intercept=True: cuml.Ridge(
        fit_intercept=fit_intercept),
    "ElasticNet": lambda fit_intercept=True: cuml.ElasticNet(
        fit_intercept=fit_intercept)
}

solver_models = {
    "CD": lambda: cuml.CD(),
    "SGD": lambda: cuml.SGD(eta0=0.005),
    "QN": lambda: cuml.QN(loss="softmax")
}

cluster_models = {
    "KMeans": lambda: cuml.KMeans()
}

decomposition_models = {
    "PCA": lambda: cuml.PCA(),
    "TruncatedSVD": lambda: cuml.TruncatedSVD(),
}

decomposition_models_xfail = {
    "GaussianRandomProjection": lambda: cuml.GaussianRandomProjection(),
    "SparseRandomProjection": lambda: cuml.SparseRandomProjection()
}

neighbor_models = {
    "NearestNeighbors": lambda: cuml.NearestNeighbors()
}

dbscan_model = {
    "DBSCAN": lambda: cuml.DBSCAN()
}

umap_model = {
    "UMAP": lambda: cuml.UMAP()
}

rf_models = {
    "rfc": lambda: cuml.RandomForestClassifier(),
    "rfr": lambda: cuml.RandomForestRegressor()
}

all_models = {**regression_models,
              **solver_models,
              **cluster_models,
              **decomposition_models,
              **decomposition_models_xfail,
              **neighbor_models,
              **dbscan_model,
              **umap_model,
              **rf_models}


def pickle_save_load(tmpdir, func_create_model, func_assert):
    model, X_test = func_create_model()
    pickle_file = tmpdir.join('cu_model.pickle')

    try:
        with open(pickle_file, 'wb') as pf:
            pickle.dump(model, pf)
    except (TypeError, ValueError) as e:
        pf.close()
        pytest.fail(e)

    del model

    with open(pickle_file, 'rb') as pf:
        cu_after_pickle_model = pickle.load(pf)

    func_assert(cu_after_pickle_model, X_test)


def make_classification_dataset(datatype, nrows, ncols, n_info):
    X, y = make_classification(n_samples=nrows, n_features=ncols,
                               n_informative=n_info,
                               random_state=0, n_classes=2)
    X = X.astype(datatype)
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    return X_train, y_train, X_test


def make_dataset(datatype, nrows, ncols, n_info):
    X, y = make_regression(n_samples=nrows, n_features=ncols,
                           n_informative=n_info, random_state=0)
    X = X.astype(datatype)
    y = y.astype(datatype)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    return X_train, y_train, X_test


@pytest.mark.parametrize('datatype', [np.float32])
@pytest.mark.parametrize('key', rf_models.keys())
@pytest.mark.parametrize('nrows', [unit_param(500)])
@pytest.mark.parametrize('ncols', [unit_param(16)])
@pytest.mark.parametrize('n_info', [unit_param(7)])
def test_rf_regression_pickle(tmpdir, datatype, nrows, ncols, n_info, key):
    result = {}

    def create_mod():
        if key == 'rfr':
            X_train, y_train, X_test = make_dataset(datatype,
                                                    nrows,
                                                    ncols,
                                                    n_info)
        else:
            X_train, y_train, X_test = make_classification_dataset(datatype,
                                                                   nrows,
                                                                   ncols,
                                                                   n_info)

        model = rf_models[key]()
        model.fit(X_train, y_train)
        result["rf_res"] = model.predict(X_test)
        return model, X_test

    def assert_model(pickled_model, X_test):
        assert array_equal(result["rf_res"], pickled_model.predict(X_test))
        # Confirm no crash from score
        pickled_model.score(X_test, np.zeros(X_test.shape[0]))

    pickle_save_load(tmpdir, create_mod, assert_model)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('keys', regression_models.keys())
@pytest.mark.parametrize('data_size', [unit_param([500, 20, 10]),
                         stress_param([500000, 1000, 500])])
@pytest.mark.parametrize('fit_intercept', [True, False])
def test_regressor_pickle(tmpdir, datatype, keys, data_size, fit_intercept):
    result = {}

    def create_mod():
        nrows, ncols, n_info = data_size
        X_train, y_train, X_test = make_dataset(datatype, nrows,
                                                ncols, n_info)
        model = regression_models[keys](fit_intercept=fit_intercept)
        model.fit(X_train, y_train)
        result["regressor"] = model.predict(X_test)
        return model, X_test

    def assert_model(pickled_model, X_test):
        assert array_equal(result["regressor"], pickled_model.predict(X_test))

    pickle_save_load(tmpdir, create_mod, assert_model)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('keys', solver_models.keys())
@pytest.mark.parametrize('data_size', [unit_param([500, 20, 10]),
                         stress_param([500000, 1000, 500])])
def test_solver_pickle(tmpdir, datatype, keys, data_size):
    result = {}

    def create_mod():
        nrows, ncols, n_info = data_size
        X_train, y_train, X_test = make_dataset(datatype, nrows,
                                                ncols, n_info)
        model = solver_models[keys]()
        model.fit(X_train, y_train)
        result["solver"] = model.predict(X_test)
        return model, X_test

    def assert_model(pickled_model, X_test):
        assert array_equal(result["solver"], pickled_model.predict(X_test))

    pickle_save_load(tmpdir, create_mod, assert_model)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('keys', cluster_models.keys())
@pytest.mark.parametrize('data_size', [unit_param([500, 20, 10]),
                         stress_param([500000, 1000, 500])])
def test_cluster_pickle(tmpdir, datatype, keys, data_size):
    result = {}

    def create_mod():
        nrows, ncols, n_info = data_size
        X_train, y_train, X_test = make_dataset(datatype, nrows,
                                                ncols, n_info)
        model = cluster_models[keys]()
        model.fit(X_train)
        result["cluster"] = model.predict(X_test)
        return model, X_test

    def assert_model(pickled_model, X_test):
        assert array_equal(result["cluster"], pickled_model.predict(X_test))

    pickle_save_load(tmpdir, create_mod, assert_model)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('keys', decomposition_models_xfail.values())
@pytest.mark.parametrize('data_size', [unit_param([500, 20, 10]),
                         stress_param([500000, 1000, 500])])
@pytest.mark.xfail
def test_decomposition_pickle(tmpdir, datatype, keys, data_size):
    result = {}

    def create_mod():
        nrows, ncols, n_info = data_size
        X_train, y_train, X_test = make_dataset(datatype, nrows,
                                                ncols, n_info)
        model = decomposition_models_xfail[keys]()
        result["decomposition"] = model.fit_transform(X_train)
        return model, X_train

    def assert_model(pickled_model, X_test):
        assert array_equal(result["decomposition"],
                           pickled_model.transform(X_test))

    pickle_save_load(tmpdir, create_mod, assert_model)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('keys', umap_model.keys())
def test_umap_pickle(tmpdir, datatype, keys):
    result = {}

    def create_mod():
        X_train = load_iris().data

        model = umap_model[keys]()
        cu_before_pickle_transform = model.fit_transform(X_train)

        result["umap_embedding"] = model.embedding_
        n_neighbors = model.n_neighbors

        result["umap"] = trustworthiness(X_train,
                                         cu_before_pickle_transform,
                                         n_neighbors)
        return model, X_train

    def assert_model(pickled_model, X_train):
        cu_after_embed = pickled_model.embedding_

        n_neighbors = pickled_model.n_neighbors
        assert array_equal(result["umap_embedding"][0][0],
                           cu_after_embed[0][0])

        cu_trust_after = trustworthiness(X_train,
                                         pickled_model.transform(X_train),
                                         n_neighbors)
        assert cu_trust_after >= result["umap"] - 0.2

    pickle_save_load(tmpdir, create_mod, assert_model)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('keys', decomposition_models.keys())
@pytest.mark.parametrize('data_size', [unit_param([500, 20, 10]),
                         stress_param([500000, 1000, 500])])
@pytest.mark.xfail
def test_decomposition_pickle_xfail(tmpdir, datatype, keys, data_size):
    result = {}

    def create_mod():
        nrows, ncols, n_info = data_size
        X_train, _, _ = make_dataset(datatype, nrows,
                                     ncols, n_info)
        model = decomposition_models[keys]()
        result["decomposition"] = model.fit_transform(X_train)
        return model, X_train

    def assert_model(pickled_model, X_test):
        assert array_equal(result["decomposition"],
                           pickled_model.transform(X_test))

    pickle_save_load(tmpdir, create_mod, assert_model)


@pytest.mark.parametrize('model_name',
                         all_models.keys())
def test_unfit_pickle(model_name):
    # Any model xfailed in this test cannot be used for hyperparameter sweeps
    # with dask or sklearn
    if model_name in decomposition_models_xfail.keys():
        pytest.xfail()

    # Pickling should work even if fit has not been called
    mod = all_models[model_name]()
    mod_pickled_bytes = pickle.dumps(mod)
    mod_unpickled = pickle.loads(mod_pickled_bytes)
    assert mod_unpickled is not None

@pytest.mark.parametrize('model_name',
                         all_models.keys())
def test_unfit_clone(model_name):
    # Cloning runs into many of the same problems as pickling
    mod = all_models[model_name]()
    clone(mod)
    # TODO: check parameters exactly?


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('keys', neighbor_models.keys())
@pytest.mark.parametrize('data_info', [unit_param([500, 20, 10, 5]),
                         stress_param([500000, 1000, 500, 50])])
def test_neighbors_pickle(tmpdir, datatype, keys, data_info):
    result = {}

    def create_mod():
        nrows, ncols, n_info, k = data_info
        X_train, _, X_test = make_dataset(datatype, nrows, ncols, n_info)

        model = neighbor_models[keys]()
        model.fit(X_train)
        result["neighbors_D"], result["neighbors_I"] = \
            model.kneighbors(X_test, n_neighbors=k)
        return model, X_test

    def assert_model(pickled_model, X_test):
        D_after, I_after = pickled_model.kneighbors(X_test,
                                                    n_neighbors=data_info[3])
        assert array_equal(result["neighbors_D"], D_after)
        assert array_equal(result["neighbors_I"], I_after)

    pickle_save_load(tmpdir, create_mod, assert_model)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('data_info', [unit_param([500, 20, 10, 5]),
                         stress_param([500000, 1000, 500, 50])])
def test_neighbors_pickle_nofit(tmpdir, datatype, data_info):
    result = {}
    """
    Note: This test digs down a bit far into the
    internals of the implementation, but it's
    important that regressions do not occur
    from changes to the class.
    """

    def create_mod():
        nrows, ncols, n_info, k = data_info
        X_train, _, X_test = make_dataset(datatype, nrows, ncols, n_info)
        model = cuml.neighbors.NearestNeighbors()
        result["model"] = model
        return model, [X_train, X_test]

    def assert_model(loaded_model, X):
        state = loaded_model.__dict__
        assert state["n_indices"] == 0
        assert "X_m" not in state
        loaded_model.fit(X[0])

        state = loaded_model.__dict__

        assert state["n_indices"] == 1
        assert "X_m" in state

    pickle_save_load(tmpdir, create_mod, assert_model)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('keys', dbscan_model.keys())
@pytest.mark.parametrize('data_size', [unit_param([500, 20, 10]),
                         stress_param([500000, 1000, 500])])
def test_dbscan_pickle(tmpdir, datatype, keys, data_size):
    result = {}

    def create_mod():
        nrows, ncols, n_info = data_size
        X_train, _, _ = make_dataset(datatype, nrows, ncols, n_info)
        model = dbscan_model[keys]()
        result["dbscan"] = model.fit_predict(X_train).to_array()
        return model, X_train

    def assert_model(pickled_model, X_train):
        pickle_after_predict = pickled_model.fit_predict(X_train).to_array()
        assert array_equal(result["dbscan"], pickle_after_predict)

    pickle_save_load(tmpdir, create_mod, assert_model)


def test_tsne_pickle(tmpdir):
    result = {}

    def create_mod():
        iris = load_iris()
        iris_selection = np.random.RandomState(42).choice(
            [True, False], 150, replace=True, p=[0.75, 0.25])
        X = iris.data[iris_selection]

        model = cuml.manifold.TSNE(n_components=2, random_state=199)
        result["model"] = model
        return model, X

    def assert_model(pickled_model, X):
        model_params = pickled_model.__dict__
        # Confirm params in model are identical
        new_keys = set(model_params.keys())
        for key, value in zip(model_params.keys(), model_params.values()):
            assert (model_params[key] == value)
            new_keys -= set([key])

        # Check all keys have been checked
        assert(len(new_keys) == 0)

        # Transform data
        result["fit_model"] = pickled_model.fit(X)
        result["data"] = X
        result["trust"] = trustworthiness(X, pickled_model.Y, 10)

    def create_mod_2():
        model = result["fit_model"]
        return model, result["data"]

    def assert_second_model(pickled_model, X):
        trust_after = trustworthiness(X, pickled_model.Y.to_pandas(), 10)
        assert result["trust"] == trust_after

    pickle_save_load(tmpdir, create_mod, assert_model)
    pickle_save_load(tmpdir, create_mod_2, assert_second_model)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
def test_svc_pickle(tmpdir, datatype):
    result = {}

    def create_mod():
        model = cuml.svm.SVC()
        iris = load_iris()
        iris_selection = np.random.RandomState(42).choice(
            [True, False], 150, replace=True, p=[0.75, 0.25])
        X_train = iris.data[iris_selection]
        y_train = iris.target[iris_selection]
        y_train = (y_train > 0).astype(datatype)
        data = [X_train, y_train]
        result["model"] = model.fit(X_train, y_train)
        return model, data

    def assert_model(pickled_model, data):
        compare_svm(result["model"], pickled_model, data[0], data[1], cmp_sv=0,
                    dcoef_tol=0)

    pickle_save_load(tmpdir, create_mod, assert_model)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('nrows', [unit_param(500)])
@pytest.mark.parametrize('ncols', [unit_param(16)])
@pytest.mark.parametrize('n_info', [unit_param(7)])
def test_svr_pickle(tmpdir, datatype, nrows, ncols, n_info):
    result = {}

    def create_mod():
        X_train, y_train, X_test = make_dataset(datatype, nrows,
                                                ncols, n_info)
        model = cuml.svm.SVR()
        model.fit(X_train, y_train)
        result["svr"] = model.predict(X_test)
        return model, X_test

    def assert_model(pickled_model, X_test):
        assert array_equal(result["svr"], pickled_model.predict(X_test))

    pickle_save_load(tmpdir, create_mod, assert_model)
