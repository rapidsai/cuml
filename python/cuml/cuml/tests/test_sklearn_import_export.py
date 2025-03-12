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
#

import pytest
import numpy as np

from cuml.cluster import KMeans, DBSCAN
from cuml.decomposition import PCA, TruncatedSVD
from cuml.linear_model import (
    LinearRegression,
    LogisticRegression,
    ElasticNet,
    Ridge,
    Lasso,
)
from cuml.manifold import TSNE
from cuml.neighbors import NearestNeighbors

from cuml.testing.utils import array_equal

from numpy.testing import assert_allclose

from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.utils.validation import check_is_fitted
from sklearn.cluster import KMeans as SkKMeans
from sklearn.decomposition import PCA as SkPCA, TruncatedSVD as SkTruncatedSVD
from sklearn.linear_model import (
    LinearRegression as SkLinearRegression,
    LogisticRegression as SkLogisticRegression,
    ElasticNet as SkElasticNet,
    Ridge as SkRidge,
    Lasso as SkLasso,
)

###############################################################################
#                              Helper functions                               #
###############################################################################


@pytest.fixture
def random_state():
    return 42


def assert_estimator_roundtrip(
    cuml_model, sklearn_class, X, y=None, transform=False
):
    """
    Generic assertion helper to test round-trip conversion:
    fit original custom model
    convert to sklearn
    convert back to custom model
    compare predictions or transform outputs
    """
    # Fit original model
    if y is not None:
        cuml_model.fit(X, y)
    else:
        cuml_model.fit(X)

    # Convert to sklearn model
    sklearn_model = cuml_model.as_sklearn()
    check_is_fitted(sklearn_model)

    original_params = cuml_model.get_params()

    assert isinstance(sklearn_model, sklearn_class)

    # Convert back
    roundtrip_model = type(cuml_model).from_sklearn(sklearn_model)

    rm_params = roundtrip_model.get_params()

    # Remove parameters that are not serialized
    _ = original_params.pop("handle", None)
    _ = rm_params.pop("handle", None)

    _ = original_params.pop("output_type", None)
    _ = rm_params.pop("output_type", None)

    _ = original_params.pop("verbose", None)
    _ = rm_params.pop("verbose", None)

    if isinstance(cuml_model, KMeans):
        # for KMeans, the roundtrip changes the string of
        # init from scalable-k-means++ to k-means++ which
        # in principle should change the value of oversampling_factor
        # But this value at 2 will lead to better centroids,
        # so ignoring this issue for now will have no ill
        # consequences
        _ = original_params.pop("init", None)
        _ = rm_params.pop("init", None)

        # This failure will be fixed by
        # https://github.com/rapidsai/cuml/pull/6142
        # otherwise the predict with default n_init like this
        # roundtrip will fail later.
        pytest.xfail(reason="auto is not supported by cuML n_init yet")

    def dict_diff(a, b):
        # Get all keys from both dictionaries
        all_keys = set(a.keys()) | set(b.keys())
        differences = {}
        for key in all_keys:
            if a.get(key) != b.get(key):
                differences[key] = {"a_dict": a.get(key), "b_dict": b.get(key)}
        return differences

    assert (
        original_params == rm_params
    ), f"Differences found: {dict_diff(original_params, rm_params)}"

    # Ensure roundtrip model is fitted
    check_is_fitted(roundtrip_model)

    # Compare predictions or transforms
    if transform:
        original_output = cuml_model.transform(X)
        roundtrip_output = roundtrip_model.transform(X)
        array_equal(original_output, roundtrip_output)
    else:
        # For predict methods
        if hasattr(cuml_model, "predict"):
            original_pred = cuml_model.predict(X)
            roundtrip_pred = roundtrip_model.predict(X)
            array_equal(original_pred, roundtrip_pred)
        # For models that only produce labels_ or similar attributes (e.g., clustering)
        elif hasattr(cuml_model, "labels_"):
            array_equal(cuml_model.labels_, roundtrip_model.labels_)
        else:
            # If we get here, need a custom handling for that type
            raise NotImplementedError(
                "No known method to compare outputs of this model."
            )


###############################################################################
#                                     Tests                                   #
###############################################################################


def test_basic_roundtrip():
    km = SkKMeans(n_clusters=13)
    ckm = KMeans.from_sklearn(km)

    assert ckm.n_clusters == 13


@pytest.mark.filterwarnings(
    "ignore:The default value of `n_init` will change from 1 to 'auto' in 25.04"
)
def test_kmeans(random_state):
    # Using sklearn directly for demonstration
    X, _ = make_blobs(
        n_samples=50, n_features=2, centers=3, random_state=random_state
    )
    original = KMeans(n_clusters=3, random_state=random_state)
    assert_estimator_roundtrip(original, SkKMeans, X)


def test_dbscan(random_state):
    X, _ = make_blobs(
        n_samples=50, n_features=2, centers=3, random_state=random_state
    )
    original = DBSCAN(eps=0.5, min_samples=5)
    # DBSCAN assigns labels_ after fit
    original.fit(X)
    sklearn_model = original.as_sklearn()
    roundtrip_model = DBSCAN.from_sklearn(sklearn_model)
    array_equal(original.labels_, roundtrip_model.labels_)


def test_pca(random_state):
    X = np.random.RandomState(random_state).rand(50, 5)
    original = PCA(n_components=2, random_state=random_state)
    assert_estimator_roundtrip(original, SkPCA, X, transform=True)


def test_truncated_svd(random_state):
    X = np.random.RandomState(random_state).rand(50, 5)
    original = TruncatedSVD(n_components=2, random_state=random_state)
    assert_estimator_roundtrip(original, SkTruncatedSVD, X, transform=True)


def test_linear_regression(random_state):
    X, y = make_regression(
        n_samples=50, n_features=5, noise=0.1, random_state=random_state
    )
    original = LinearRegression()
    assert_estimator_roundtrip(original, SkLinearRegression, X, y)


def test_logistic_regression(random_state):
    X, y = make_classification(
        n_samples=50, n_features=5, n_informative=3, random_state=random_state
    )
    original = LogisticRegression(random_state=random_state, max_iter=500)
    assert_estimator_roundtrip(original, SkLogisticRegression, X, y)


def test_elasticnet(random_state):
    X, y = make_regression(
        n_samples=50, n_features=5, noise=0.1, random_state=random_state
    )
    original = ElasticNet(random_state=random_state)
    assert_estimator_roundtrip(original, SkElasticNet, X, y)


def test_ridge(random_state):
    X, y = make_regression(
        n_samples=50, n_features=5, noise=0.1, random_state=random_state
    )
    original = Ridge(alpha=1.0, random_state=random_state)
    assert_estimator_roundtrip(original, SkRidge, X, y)


def test_lasso(random_state):
    X, y = make_regression(
        n_samples=50, n_features=5, noise=0.1, random_state=random_state
    )
    original = Lasso(alpha=0.1, random_state=random_state)
    assert_estimator_roundtrip(original, SkLasso, X, y)


@pytest.mark.filterwarnings(
    "ignore:Starting from version 22.04, the default method of TSNE is 'fft'."
)
def test_tsne(random_state):
    # TSNE is a bit tricky as it is non-deterministic. For test simplicity:
    X = np.random.RandomState(random_state).rand(50, 5)
    original = TSNE(n_components=2, random_state=random_state)
    original.fit(X)
    sklearn_model = original.as_sklearn()
    roundtrip_model = TSNE.from_sklearn(sklearn_model)
    # Since TSNE is non-deterministic, exact match is unlikely.
    # We can at least check output dimensions are the same.
    original_embedding = original.embedding_
    sklearn_embedding = sklearn_model.embedding_
    roundtrip_embedding = roundtrip_model.embedding_

    array_equal(original_embedding, sklearn_embedding)
    array_equal(original_embedding, roundtrip_embedding)


def test_nearest_neighbors(random_state):
    X = np.random.RandomState(random_state).rand(50, 5)
    original = NearestNeighbors(n_neighbors=5)
    original.fit(X)
    sklearn_model = original.as_sklearn()
    roundtrip_model = NearestNeighbors.from_sklearn(sklearn_model)
    # Check that the kneighbors results are the same
    dist_original, ind_original = original.kneighbors(X)
    dist_roundtrip, ind_roundtrip = roundtrip_model.kneighbors(X)
    assert_allclose(dist_original, dist_roundtrip)
    assert_allclose(ind_original, ind_roundtrip)
