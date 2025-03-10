#
# Copyright (c) 2024, NVIDIA CORPORATION.
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
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score


@pytest.fixture(scope="module")
def clustering_data():
    X, y = make_blobs(
        n_samples=300,
        centers=3,
        cluster_std=[1.0, 2.5, 0.5],
        random_state=42,
    )
    return X, y


@pytest.mark.parametrize("eps", [0.1, 0.5, 1.0, 2.0])
def test_dbscan_eps(clustering_data, eps):
    X, y_true = clustering_data
    dbscan = DBSCAN(eps=eps).fit(X)
    y_pred = dbscan.labels_
    adjusted_rand_score(y_true, y_pred)


@pytest.mark.parametrize("min_samples", [1, 5, 10, 20])
def test_dbscan_min_samples(clustering_data, min_samples):
    X, y_true = clustering_data
    dbscan = DBSCAN(eps=0.5, min_samples=min_samples).fit(X)
    y_pred = dbscan.labels_
    adjusted_rand_score(y_true, y_pred)


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev"])
def test_dbscan_metric(clustering_data, metric):
    X, y_true = clustering_data
    dbscan = DBSCAN(eps=0.5, metric=metric).fit(X)
    y_pred = dbscan.labels_
    adjusted_rand_score(y_true, y_pred)


@pytest.mark.parametrize(
    "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
)
def test_dbscan_algorithm(clustering_data, algorithm):
    X, y_true = clustering_data
    dbscan = DBSCAN(eps=0.5, algorithm=algorithm).fit(X)
    y_pred = dbscan.labels_
    adjusted_rand_score(y_true, y_pred)


@pytest.mark.parametrize("leaf_size", [10, 30, 50])
def test_dbscan_leaf_size(clustering_data, leaf_size):
    X, y_true = clustering_data
    dbscan = DBSCAN(eps=0.5, leaf_size=leaf_size).fit(X)
    y_pred = dbscan.labels_
    adjusted_rand_score(y_true, y_pred)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_dbscan_p(clustering_data, p):
    X, y_true = clustering_data
    dbscan = DBSCAN(eps=0.5, metric="minkowski", p=p).fit(X)
    y_pred = dbscan.labels_
    adjusted_rand_score(y_true, y_pred)


def test_dbscan_consistency(clustering_data):
    X, y_true = clustering_data
    dbscan1 = DBSCAN(eps=0.5).fit(X)
    dbscan2 = DBSCAN(eps=0.5).fit(X)
    assert np.array_equal(
        dbscan1.labels_, dbscan2.labels_
    ), "Results should be consistent across runs"
