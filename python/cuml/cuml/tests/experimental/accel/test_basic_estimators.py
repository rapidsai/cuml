# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    ElasticNet,
    Ridge,
    Lasso,
)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.kernel_ridge import KernelRidge
from sklearn.manifold import TSNE
from sklearn.neighbors import (
    NearestNeighbors,
    KNeighborsClassifier,
    KNeighborsRegressor,
)
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    adjusted_rand_score,
    accuracy_score,
)
from scipy.sparse import random as sparse_random


def test_kmeans():
    X, y_true = make_blobs(n_samples=100, centers=3, random_state=42)
    clf = KMeans().fit(X)
    clf.predict(X)


def test_dbscan():
    X, y_true = make_blobs(n_samples=100, centers=3, random_state=42)
    clf = DBSCAN().fit(X)
    clf.labels_


def test_pca():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    pca = PCA().fit(X)
    pca.transform(X)


def test_truncated_svd():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    svd = TruncatedSVD().fit(X)
    svd.transform(X)


def test_linear_regression():
    X, y = make_regression(
        n_samples=100, n_features=20, noise=0.1, random_state=42
    )
    lr = LinearRegression().fit(X, y)
    lr.predict(X)


def test_logistic_regression():
    X, y = make_classification(
        n_samples=100, n_features=20, n_classes=2, random_state=42
    )
    clf = LogisticRegression().fit(X, y)
    clf.predict(X)


def test_elastic_net():
    X, y = make_regression(
        n_samples=100, n_features=20, noise=0.1, random_state=42
    )
    enet = ElasticNet().fit(X, y)
    enet.predict(X)


def test_ridge():
    X, y = make_regression(
        n_samples=100, n_features=20, noise=0.1, random_state=42
    )
    ridge = Ridge().fit(X, y)
    ridge.predict(X)


def test_lasso():
    X, y = make_regression(
        n_samples=100, n_features=20, noise=0.1, random_state=42
    )
    lasso = Lasso().fit(X, y)
    lasso.predict(X)


def test_tsne():
    X, _ = make_blobs(n_samples=100, centers=3, n_features=20, random_state=42)
    tsne = TSNE()
    tsne.fit_transform(X)


def test_nearest_neighbors():
    X, _ = make_blobs(n_samples=100, centers=3, n_features=20, random_state=42)
    nn = NearestNeighbors().fit(X)
    distances, indices = nn.kneighbors(X)
    assert distances.shape == (100, 5)
    assert indices.shape == (100, 5)


def test_k_neighbors_classifier():
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_classes=3,
        random_state=42,
        n_informative=6,
    )
    for weights in ["uniform", "distance"]:
        for metric in ["euclidean", "manhattan"]:
            knn = KNeighborsClassifier().fit(X, y)
            knn.predict(X)


def test_k_neighbors_regressor():
    X, y = make_regression(
        n_samples=100, n_features=20, noise=0.1, random_state=42
    )
    for weights in ["uniform", "distance"]:
        for metric in ["euclidean", "manhattan"]:
            knr = KNeighborsRegressor().fit(X, y)
            knr.predict(X)


def test_proxy_facade():
    # Check that the proxy estimator pretends to look like the
    # class it is proxying

    # A random estimator, shouldn't matter which one as all are proxied
    # the same way.
    # We need an instance to get access to the `_cpu_model_class`
    # but we want to compare to the PCA class
    pca = PCA()
    for attr in (
        "__module__",
        "__name__",
        "__qualname__",
        "__doc__",
        "__annotate__",
        "__type_params__",
    ):
        # if the original class has this attribute, the proxy should
        # have it as well and the values should match
        try:
            original_value = getattr(pca._cpu_model_class, attr)
        except AttributeError:
            pass
        else:
            proxy_value = getattr(PCA, attr)

            assert original_value == proxy_value


def test_defaults_args_only_methods():
    # Check that estimator methods that take no arguments work
    # These are slightly weird because basically everything else takes
    # a X as input.
    X = np.random.rand(1000, 3)
    y = X[:, 0] + np.sin(6 * np.pi * X[:, 1]) + 0.1 * np.random.randn(1000)

    nn = NearestNeighbors(metric="chebyshev", n_neighbors=3)
    nn.fit(X[:, 0].reshape((-1, 1)), y)
    nn.kneighbors()
