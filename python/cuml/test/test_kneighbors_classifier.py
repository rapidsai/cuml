
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

from cuml.neighbors import KNeighborsClassifier as cuKNN

from sklearn.datasets import make_blobs
import numpy as np
from cuml.test.utils import array_equal


@pytest.mark.parametrize("nrows", [1000, 10000])
@pytest.mark.parametrize("ncols", [50, 100])
@pytest.mark.parametrize("n_neighbors", [2, 5, 10])
@pytest.mark.parametrize("n_clusters", [2, 5, 10])
def test_neighborhood_predictions(nrows, ncols, n_neighbors, n_clusters):

    X, y = make_blobs(n_samples=nrows,
                      centers=n_clusters,
                      n_features=ncols,
                      cluster_std=0.01,
                      random_state=0)

    X = X.astype(np.float32)

    knn_cu = cuKNN(n_neighbors=n_neighbors)
    knn_cu.fit(X, y)

    predictions = knn_cu.predict(X)

    assert array_equal(predictions.astype(np.int32), y.astype(np.int32))


@pytest.mark.parametrize("nrows", [1000, 10000])
@pytest.mark.parametrize("ncols", [50, 100])
@pytest.mark.parametrize("n_neighbors", [2, 5, 10])
@pytest.mark.parametrize("n_clusters", [2, 5, 10])
def test_score(nrows, ncols, n_neighbors, n_clusters):

    X, y = make_blobs(n_samples=nrows, centers=n_clusters,
                      n_features=ncols, random_state=0,
                      cluster_std=0.01)

    X = X.astype(np.float32)

    knn_cu = cuKNN(n_neighbors=n_neighbors)
    knn_cu.fit(X, y)

    assert knn_cu.score(X, y) == 1.0


@pytest.mark.parametrize("nrows", [1000, 10000])
@pytest.mark.parametrize("ncols", [50, 100])
@pytest.mark.parametrize("n_neighbors", [2, 5, 10])
@pytest.mark.parametrize("n_clusters", [2, 5, 10])
def test_predict_proba(nrows, ncols, n_neighbors, n_clusters):

    X, y = make_blobs(n_samples=nrows,
                      centers=n_clusters,
                      n_features=ncols,
                      cluster_std=0.01,
                      random_state=0)

    X = X.astype(np.float32)

    knn_cu = cuKNN(n_neighbors=n_neighbors)
    knn_cu.fit(X, y)

    predictions = knn_cu.predict_proba(X)

    assert len(predictions[predictions < 0]) == 0
    assert len(predictions[predictions > 1]) == 0

    y_hat = np.argmax(np.array(predictions), axis=1)

    assert array_equal(y_hat.astype(np.int32), y.astype(np.int32))


def test_nonmonotonic_labels():

    X = np.array([[0, 0, 1], [1, 0, 1]]).astype(np.float32)

    y = np.array([15, 5]).astype(np.int32)

    knn_cu = cuKNN(n_neighbors=1)
    knn_cu.fit(X, y)

    p = knn_cu.predict(X)

    assert array_equal(p.astype(np.int32), y)


def test_predict_multioutput():

    X = np.array([[0, 0, 1], [1, 0, 1]]).astype(np.float32)

    y = np.array([[15, 2], [5, 4]]).astype(np.int32)

    knn_cu = cuKNN(n_neighbors=1)
    knn_cu.fit(X, y)

    p = knn_cu.predict(X)

    assert array_equal(p.astype(np.int32), y)


def test_predict_proba_multioutput():

    X = np.array([[0, 0, 1], [1, 0, 1]]).astype(np.float32)

    y = np.array([[15, 2], [5, 4]]).astype(np.int32)

    expected = (np.array([[0., 1.], [1., 0.]]).astype(np.float32),
                np.array([[1., 0.], [0., 1.]]).astype(np.float32))

    knn_cu = cuKNN(n_neighbors=1)
    knn_cu.fit(X, y)

    p = knn_cu.predict_proba(X)

    assert isinstance(p, tuple)

    assert array_equal(p[0].astype(np.float32), expected[0])
    assert array_equal(p[1].astype(np.float32), expected[1])
