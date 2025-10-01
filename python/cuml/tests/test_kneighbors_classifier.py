# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

import cudf
import cupy as cp
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier as skKNN

import cuml
from cuml.neighbors import KNeighborsClassifier as cuKNN
from cuml.testing.utils import array_equal


def _build_train_test_data(X, y, datatype, train_ratio=0.9):

    train_selection = np.random.RandomState(42).choice(
        [True, False],
        X.shape[0],
        replace=True,
        p=[train_ratio, 1.0 - train_ratio],
    )

    X_train = X[train_selection]
    y_train = y[train_selection]
    X_test = X[~train_selection]
    y_test = y[~train_selection]

    if datatype == "dataframe":
        X_train = cudf.DataFrame(X_train)
        y_train = cudf.DataFrame(y_train.reshape(y_train.shape[0], 1))
        X_test = cudf.DataFrame(X_test)
        y_test = cudf.DataFrame(y_test.reshape(y_test.shape[0], 1))

    return X_train, X_test, y_train, y_test


@pytest.mark.parametrize("datatype", ["dataframe", "numpy"])
@pytest.mark.parametrize("nrows", [1000, 20000])
@pytest.mark.parametrize("ncols", [50, 100])
@pytest.mark.parametrize("n_neighbors", [2, 5, 10])
@pytest.mark.parametrize("n_clusters", [2, 5, 10])
def test_neighborhood_predictions(
    nrows, ncols, n_neighbors, n_clusters, datatype
):

    X, y = make_blobs(
        n_samples=nrows,
        centers=n_clusters,
        n_features=ncols,
        cluster_std=0.01,
        random_state=0,
    )

    X = X.astype(np.float32)

    X_train, X_test, y_train, y_test = _build_train_test_data(X, y, datatype)

    knn_cu = cuKNN(n_neighbors=n_neighbors)
    knn_cu.fit(X_train, y_train)

    predictions = knn_cu.predict(X_test)

    if datatype == "dataframe":
        assert isinstance(predictions, cudf.Series)
        assert array_equal(
            predictions.to_frame().astype(np.int32), y_test.astype(np.int32)
        )
    else:
        assert isinstance(predictions, np.ndarray)

        assert array_equal(
            predictions.astype(np.int32), y_test.astype(np.int32)
        )


@pytest.mark.parametrize("datatype", ["dataframe", "numpy"])
@pytest.mark.parametrize("nrows", [1000, 20000])
@pytest.mark.parametrize("ncols", [50, 100])
@pytest.mark.parametrize("n_neighbors", [2, 5, 10])
@pytest.mark.parametrize("n_clusters", [2, 5, 10])
@pytest.mark.parametrize("weighted", [False, True])
def test_score(nrows, ncols, n_neighbors, n_clusters, datatype, weighted):

    X, y = make_blobs(
        n_samples=nrows,
        centers=n_clusters,
        n_features=ncols,
        random_state=0,
        cluster_std=0.01,
    )

    X = X.astype(np.float32)
    X_train, X_test, y_train, y_test = _build_train_test_data(X, y, datatype)

    if weighted:
        sample_weight = np.random.default_rng(42).uniform(
            0.5, 1, size=len(X_test)
        )
    else:
        sample_weight = None

    knn_cu = cuKNN(n_neighbors=n_neighbors)
    knn_cu.fit(X_train, y_train)

    score = knn_cu.score(X_test, y_test, sample_weight=sample_weight)
    assert score >= (1.0 - 0.004)


@pytest.mark.parametrize("datatype", ["dataframe", "numpy"])
@pytest.mark.parametrize("nrows", [1000, 20000])
@pytest.mark.parametrize("ncols", [50, 100])
@pytest.mark.parametrize("n_neighbors", [2, 5, 10])
@pytest.mark.parametrize("n_clusters", [2, 5, 10])
def test_predict_proba(nrows, ncols, n_neighbors, n_clusters, datatype):

    X, y = make_blobs(
        n_samples=nrows,
        centers=n_clusters,
        n_features=ncols,
        cluster_std=0.01,
        random_state=0,
    )

    X = X.astype(np.float32)

    X_train, X_test, y_train, y_test = _build_train_test_data(X, y, datatype)

    knn_cu = cuKNN(n_neighbors=n_neighbors)
    knn_cu.fit(X_train, y_train)

    predictions = knn_cu.predict_proba(X_test)

    if datatype == "dataframe":
        assert isinstance(predictions, cudf.DataFrame)
        predictions = predictions.to_numpy()
        y_test = y_test.to_numpy().reshape(y_test.shape[0])
    else:
        assert isinstance(predictions, np.ndarray)

    y_hat = np.argmax(predictions, axis=1)

    assert array_equal(y_hat.astype(np.int32), y_test.astype(np.int32))
    assert array_equal(predictions.sum(axis=1), np.ones(y_test.shape[0]))


@pytest.mark.parametrize("datatype", ["dataframe", "numpy"])
def test_predict_proba_large_n_classes(datatype):

    nrows = 10000
    ncols = 100
    n_neighbors = 10
    n_clusters = 10000

    X, y = make_blobs(
        n_samples=nrows,
        centers=n_clusters,
        n_features=ncols,
        cluster_std=0.01,
        random_state=0,
    )

    X = X.astype(np.float32)

    X_train, X_test, y_train, y_test = _build_train_test_data(X, y, datatype)

    knn_cu = cuKNN(n_neighbors=n_neighbors)
    knn_cu.fit(X_train, y_train)

    predictions = knn_cu.predict_proba(X_test)

    if datatype == "dataframe":
        predictions = predictions.to_numpy()

    assert np.rint(np.sum(predictions)) == len(y_test)


@pytest.mark.parametrize("datatype", ["dataframe", "numpy"])
def test_predict_large_n_classes(datatype):

    nrows = 10000
    ncols = 100
    n_neighbors = 2
    n_clusters = 1000

    X, y = make_blobs(
        n_samples=nrows,
        centers=n_clusters,
        n_features=ncols,
        cluster_std=0.01,
        random_state=0,
    )

    X = X.astype(np.float32)

    X_train, X_test, y_train, y_test = _build_train_test_data(X, y, datatype)

    knn_cu = cuKNN(n_neighbors=n_neighbors)
    knn_cu.fit(X_train, y_train)

    y_hat = knn_cu.predict(X_test)

    if datatype == "dataframe":
        y_hat = y_hat.to_numpy()
        y_test = y_test.to_numpy().ravel()

    assert array_equal(y_hat.astype(np.int32), y_test.astype(np.int32))


# Ignore FutureWarning: Using `__dataframe__` is deprecated
@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("n_samples", [100])
@pytest.mark.parametrize("n_features", [40])
@pytest.mark.parametrize("n_neighbors", [4])
@pytest.mark.parametrize("n_query", [100])
def test_predict_non_gaussian(n_samples, n_features, n_neighbors, n_query):

    np.random.seed(123)

    X_host_train = pd.DataFrame(
        np.random.uniform(0, 1, (n_samples, n_features))
    )
    y_host_train = pd.DataFrame(np.random.randint(0, 5, (n_samples, 1)))
    X_host_test = pd.DataFrame(np.random.uniform(0, 1, (n_query, n_features)))

    X_device_train = cudf.DataFrame(X_host_train)
    y_device_train = cudf.DataFrame(y_host_train)

    X_device_test = cudf.DataFrame(X_host_test)

    knn_sk = skKNN(algorithm="brute", n_neighbors=n_neighbors, n_jobs=1)
    knn_sk.fit(X_host_train, y_host_train.values.ravel())

    sk_result = knn_sk.predict(X_host_test)

    knn_cuml = cuKNN(n_neighbors=n_neighbors)
    knn_cuml.fit(X_device_train, y_device_train)

    with cuml.using_output_type("numpy"):
        cuml_result = knn_cuml.predict(X_device_test)

        assert np.array_equal(cuml_result, sk_result)


@pytest.mark.parametrize("n_classes", [2, 5])
@pytest.mark.parametrize("n_rows", [1000])
@pytest.mark.parametrize("n_cols", [25, 50])
@pytest.mark.parametrize("n_neighbors", [3, 5])
@pytest.mark.parametrize("datatype", ["numpy", "dataframe"])
def test_nonmonotonic_labels(n_classes, n_rows, n_cols, datatype, n_neighbors):

    X, y = make_blobs(
        n_samples=n_rows,
        centers=n_classes,
        n_features=n_cols,
        cluster_std=0.01,
        random_state=0,
    )

    X = X.astype(np.float32)

    # Draw labels from non-monotonically increasing set
    classes = np.arange(0, n_classes * 5, 5)
    for i in range(n_classes):
        y[y == i] = classes[i]

    X_train, X_test, y_train, y_test = _build_train_test_data(X, y, datatype)

    knn_cu = cuKNN(n_neighbors=n_neighbors)
    knn_cu.fit(X_train, y_train)

    p = knn_cu.predict(X_test)

    if datatype == "dataframe":
        assert isinstance(p, cudf.Series)
        p = p.to_frame().to_numpy().reshape(p.shape[0])
        y_test = y_test.to_numpy().reshape(y_test.shape[0])

    assert array_equal(p.astype(np.int32), y_test.astype(np.int32))


@pytest.mark.parametrize("output_type", ["numpy", "cupy"])
@pytest.mark.parametrize("multioutput", [False, True])
def test_classes(output_type, multioutput):
    X = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 0]], dtype="float32")
    if multioutput:
        y = np.array([[3, 4], [1, 2], [3, 5]], dtype="int32")
    else:
        y = np.array([3, 1, 3], dtype="int32")
    knn_cu = cuKNN(output_type=output_type).fit(X, y)
    knn_sk = skKNN().fit(X, y)

    def check(cu, sk):
        if output_type == "cupy":
            assert isinstance(cu, cp.ndarray)
            cu = cu.get()
        else:
            assert isinstance(cu, np.ndarray)
        np.testing.assert_array_equal(cu, sk)

    assert knn_cu.outputs_2d_ == knn_sk.outputs_2d_

    if multioutput:
        assert isinstance(knn_cu.classes_, list)
        assert len(knn_cu.classes_) == 2
        for cu, sk in zip(knn_cu.classes_, knn_sk.classes_):
            check(cu, sk)
    else:
        check(knn_cu.classes_, knn_sk.classes_)


@pytest.mark.parametrize("input_type", ["cudf", "numpy", "cupy"])
@pytest.mark.parametrize("output_type", ["cudf", "numpy", "cupy"])
def test_predict_multioutput(input_type, output_type):

    X = np.array([[0, 0, 1, 0], [1, 0, 1, 0]]).astype(np.float32)
    y = np.array([[15, 2], [5, 4]]).astype(np.int32)

    if input_type == "cudf":
        X = cudf.DataFrame(X)
        y = cudf.DataFrame(y)
    elif input_type == "cupy":
        X = cp.asarray(X)
        y = cp.asarray(y)

    knn_cu = cuKNN(n_neighbors=1, output_type=output_type)
    knn_cu.fit(X, y)

    p = knn_cu.predict(X)

    if output_type == "cudf":
        assert isinstance(p, cudf.DataFrame)
    elif output_type == "numpy":
        assert isinstance(p, np.ndarray)
    elif output_type == "cupy":
        assert isinstance(p, cp.ndarray)

    assert array_equal(p.astype(np.int32), y)


@pytest.mark.parametrize("input_type", ["cudf", "numpy", "cupy"])
@pytest.mark.parametrize("output_type", ["cudf", "numpy", "cupy"])
def test_predict_proba_multioutput(input_type, output_type):

    X = np.array([[0, 0, 1, 0], [1, 0, 1, 0]]).astype(np.float32)
    y = np.array([[15, 2], [5, 4]]).astype(np.int32)

    if input_type == "cudf":
        X = cudf.DataFrame(X)
        y = cudf.DataFrame(y)
    elif input_type == "cupy":
        X = cp.asarray(X)
        y = cp.asarray(y)

    expected = (
        np.array([[0.0, 1.0], [1.0, 0.0]]).astype(np.float32),
        np.array([[1.0, 0.0], [0.0, 1.0]]).astype(np.float32),
    )

    knn_cu = cuKNN(n_neighbors=1, output_type=output_type)
    knn_cu.fit(X, y)

    p = knn_cu.predict_proba(X)

    assert isinstance(p, list)

    for i in p:
        if output_type == "cudf":
            assert isinstance(i, cudf.DataFrame)
        elif output_type == "numpy":
            assert isinstance(i, np.ndarray)
        elif output_type == "cupy":
            assert isinstance(i, cp.ndarray)

    assert array_equal(p[0].astype(np.float32), expected[0])
    assert array_equal(p[1].astype(np.float32), expected[1])
