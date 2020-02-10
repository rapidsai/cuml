
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

import rmm


import cudf

from cuml.neighbors import KNeighborsClassifier as cuKNN
from sklearn.neighbors import KNeighborsClassifier as skKNN

from sklearn.datasets import make_blobs
import numpy as np
from cuml.test.utils import array_equal

import pandas as pd


@pytest.mark.parametrize("datatype", ["dataframe", "numpy"])
@pytest.mark.parametrize("nrows", [1000, 10000])
@pytest.mark.parametrize("ncols", [50, 100])
@pytest.mark.parametrize("n_neighbors", [2, 5, 10])
@pytest.mark.parametrize("n_clusters", [2, 5, 10])
def test_neighborhood_predictions(nrows, ncols, n_neighbors,
                                  n_clusters, datatype):

    X, y = make_blobs(n_samples=nrows,
                      centers=n_clusters,
                      n_features=ncols,
                      cluster_std=0.01,
                      random_state=0)

    X = X.astype(np.float32)

    if datatype == "dataframe":
        X = cudf.DataFrame.from_gpu_matrix(rmm.to_device(X))
        y = cudf.DataFrame.from_gpu_matrix(rmm.to_device(y.reshape(nrows, 1)))

    knn_cu = cuKNN(n_neighbors=n_neighbors)
    knn_cu.fit(X, y)

    predictions = knn_cu.predict(X)

    if datatype == "dataframe":
        assert isinstance(predictions, cudf.DataFrame)
    else:
        assert isinstance(predictions, np.ndarray)

    assert array_equal(predictions.astype(np.int32), y.astype(np.int32))


@pytest.mark.parametrize("datatype", ["dataframe", "numpy"])
@pytest.mark.parametrize("nrows", [1000, 10000])
@pytest.mark.parametrize("ncols", [50, 100])
@pytest.mark.parametrize("n_neighbors", [2, 5, 10])
@pytest.mark.parametrize("n_clusters", [2, 5, 10])
def test_score(nrows, ncols, n_neighbors, n_clusters, datatype):

    X, y = make_blobs(n_samples=nrows, centers=n_clusters,
                      n_features=ncols, random_state=0,
                      cluster_std=0.01)

    X = X.astype(np.float32)

    if datatype == "dataframe":
        X = cudf.DataFrame.from_gpu_matrix(rmm.to_device(X))
        y = cudf.DataFrame.from_gpu_matrix(rmm.to_device(y.reshape(nrows, 1)))

    knn_cu = cuKNN(n_neighbors=n_neighbors)
    knn_cu.fit(X, y)

    assert knn_cu.score(X, y) >= (1.0 - 0.004)


@pytest.mark.parametrize("datatype", ["dataframe", "numpy"])
@pytest.mark.parametrize("nrows", [1000, 10000])
@pytest.mark.parametrize("ncols", [50, 100])
@pytest.mark.parametrize("n_neighbors", [2, 5, 10])
@pytest.mark.parametrize("n_clusters", [2, 5, 10])
def test_predict_proba(nrows, ncols, n_neighbors, n_clusters, datatype):

    X, y = make_blobs(n_samples=nrows,
                      centers=n_clusters,
                      n_features=ncols,
                      cluster_std=0.01,
                      random_state=0)

    X = X.astype(np.float32)

    if datatype == "dataframe":
        X = cudf.DataFrame.from_gpu_matrix(rmm.to_device(X))
        y = cudf.DataFrame.from_gpu_matrix(rmm.to_device(y.reshape(nrows, 1)))

    knn_cu = cuKNN(n_neighbors=n_neighbors)
    knn_cu.fit(X, y)

    predictions = knn_cu.predict_proba(X)

    if datatype == "dataframe":
        assert isinstance(predictions, cudf.DataFrame)
        predictions = predictions.as_gpu_matrix().copy_to_host()
        y = y.as_gpu_matrix().copy_to_host().reshape(nrows)
    else:
        assert isinstance(predictions, np.ndarray)

    y_hat = np.argmax(predictions, axis=1)

    assert array_equal(y_hat.astype(np.int32), y.astype(np.int32))
    assert array_equal(predictions.sum(axis=1), np.ones(nrows))


@pytest.mark.parametrize("n_samples", [100])
@pytest.mark.parametrize("n_features", [40])
@pytest.mark.parametrize("n_neighbors", [4])
@pytest.mark.parametrize("n_query", [100])
def test_predict_non_gaussian(n_samples, n_features, n_neighbors, n_query):

    np.random.seed(123)

    X_host_train = pd.DataFrame(np.random.uniform(0, 1,
                                                  (n_samples, n_features)))
    y_host_train = pd.DataFrame(np.random.randint(0, 5, (n_samples, 1)))
    X_host_test = pd.DataFrame(np.random.uniform(0, 1,
                                                 (n_query, n_features)))

    X_device_train = cudf.DataFrame.from_pandas(X_host_train)
    y_device_train = cudf.DataFrame.from_pandas(y_host_train)

    X_device_test = cudf.DataFrame.from_pandas(X_host_test)

    knn_sk = skKNN(algorithm="brute", n_neighbors=n_neighbors, n_jobs=1)
    knn_sk.fit(X_host_train, y_host_train)

    sk_result = knn_sk.predict(X_host_test)

    knn_cuml = cuKNN(n_neighbors=n_neighbors)
    knn_cuml.fit(X_device_train, y_device_train)

    cuml_result = knn_cuml.predict(X_device_test)

    assert np.array_equal(
        np.asarray(cuml_result.as_gpu_matrix())[:, 0], sk_result)


def test_nonmonotonic_labels():

    X = np.array([[0, 0, 1], [1, 0, 1]]).astype(np.float32)

    y = np.array([15, 5]).astype(np.int32)

    knn_cu = cuKNN(n_neighbors=1)
    knn_cu.fit(X, y)

    p = knn_cu.predict(X)

    assert array_equal(p.astype(np.int32), y)


@pytest.mark.parametrize("datatype", ["dataframe", "numpy"])
def test_predict_multioutput(datatype):

    X = np.array([[0, 0, 1], [1, 0, 1]]).astype(np.float32)
    y = np.array([[15, 2], [5, 4]]).astype(np.int32)

    if datatype == "dataframe":
        X = cudf.DataFrame.from_gpu_matrix(rmm.to_device(X))
        y = cudf.DataFrame.from_gpu_matrix(rmm.to_device(y))

    knn_cu = cuKNN(n_neighbors=1)
    knn_cu.fit(X, y)

    p = knn_cu.predict(X)

    if datatype == "dataframe":
        assert isinstance(p, cudf.DataFrame)
    else:
        assert isinstance(p, np.ndarray)

    assert array_equal(p.astype(np.int32), y)


@pytest.mark.parametrize("datatype", ["dataframe", "numpy"])
def test_predict_proba_multioutput(datatype):

    X = np.array([[0, 0, 1], [1, 0, 1]]).astype(np.float32)
    y = np.array([[15, 2], [5, 4]]).astype(np.int32)

    if datatype == "dataframe":
        X = cudf.DataFrame.from_gpu_matrix(rmm.to_device(X))
        y = cudf.DataFrame.from_gpu_matrix(rmm.to_device(y))

    expected = (np.array([[0., 1.], [1., 0.]]).astype(np.float32),
                np.array([[1., 0.], [0., 1.]]).astype(np.float32))

    knn_cu = cuKNN(n_neighbors=1)
    knn_cu.fit(X, y)

    p = knn_cu.predict_proba(X)

    assert isinstance(p, tuple)

    for i in p:
        if datatype == "dataframe":
            assert isinstance(i, cudf.DataFrame)
        else:
            assert isinstance(i, np.ndarray)

    assert array_equal(p[0].astype(np.float32), expected[0])
    assert array_equal(p[1].astype(np.float32), expected[1])
