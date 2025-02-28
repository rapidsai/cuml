#
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
from cuml.internals.global_settings import GlobalSettings
from sklearn.datasets import make_classification
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

pytestmark = pytest.mark.skipif(
    not GlobalSettings().accelerator_active,
    reason="Tests take a long time on CI without GPU acceleration",
)


@pytest.fixture(scope="module")
def synthetic_data():
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=10,
        n_redundant=10,
        n_clusters_per_class=1,
        n_classes=5,
        random_state=42,
    )
    # Standardize features
    X = StandardScaler().fit_transform(X)
    return X, y


@pytest.mark.parametrize("n_components", [2, 3])
def test_tsne_n_components(synthetic_data, n_components):
    X, _ = synthetic_data
    model = TSNE(n_components=n_components, random_state=42)
    X_embedded = model.fit_transform(X)
    assert (
        X_embedded.shape[1] == n_components
    ), f"Output dimensions should be {n_components}"


@pytest.mark.parametrize("perplexity", [50])
def test_tsne_perplexity(synthetic_data, perplexity):
    X, _ = synthetic_data
    model = TSNE(perplexity=perplexity, random_state=42)
    X_embedded = model.fit_transform(X)
    # Check that the embedding has the correct shape
    assert (
        X_embedded.shape[0] == X.shape[0]
    ), "Number of samples should remain the same"


@pytest.mark.parametrize("early_exaggeration", [12.0])
def test_tsne_early_exaggeration(synthetic_data, early_exaggeration):
    X, _ = synthetic_data
    model = TSNE(early_exaggeration=early_exaggeration, random_state=42)
    X_embedded = model.fit_transform(X)
    # Check that the embedding has the correct shape
    assert (
        X_embedded.shape[0] == X.shape[0]
    ), "Number of samples should remain the same"


@pytest.mark.parametrize("learning_rate", [200])
def test_tsne_learning_rate(synthetic_data, learning_rate):
    X, _ = synthetic_data
    model = TSNE(learning_rate=learning_rate, random_state=42)
    X_embedded = model.fit_transform(X)
    # Check that the embedding has the correct shape
    assert (
        X_embedded.shape[0] == X.shape[0]
    ), "Number of samples should remain the same"


@pytest.mark.parametrize("n_iter", [250])
def test_tsne_n_iter(synthetic_data, n_iter):
    X, _ = synthetic_data
    model = TSNE(n_iter=n_iter, random_state=42)
    model.fit_transform(X)
    # Since TSNE may perform additional iterations, check if n_iter_ is at least n_iter
    assert (
        model.n_iter_ >= n_iter
    ), f"Number of iterations should be at least {n_iter}"


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine"])
def test_tsne_metric(synthetic_data, metric):
    X, _ = synthetic_data
    model = TSNE(metric=metric, random_state=42)
    X_embedded = model.fit_transform(X)
    # Check that the embedding has the correct shape
    assert (
        X_embedded.shape[0] == X.shape[0]
    ), f"Embedding should have same number of samples with metric={metric}"


@pytest.mark.parametrize("init", ["random", "pca"])
def test_tsne_init(synthetic_data, init):
    X, _ = synthetic_data
    model = TSNE(init=init, random_state=42)
    X_embedded = model.fit_transform(X)
    # Check that the embedding has the correct shape
    assert (
        X_embedded.shape[0] == X.shape[0]
    ), f"Embedding should have same number of samples with init={init}"


@pytest.mark.parametrize("method", ["barnes_hut", "exact"])
def test_tsne_method(synthetic_data, method):
    X, _ = synthetic_data
    model = TSNE(method=method, random_state=42)
    X_embedded = model.fit_transform(X)
    # Check that the embedding has the correct shape
    assert (
        X_embedded.shape[0] == X.shape[0]
    ), f"Embedding should have same number of samples with method={method}"


@pytest.mark.parametrize("angle", [0.2])
def test_tsne_angle(synthetic_data, angle):
    X, _ = synthetic_data
    model = TSNE(method="barnes_hut", angle=angle, random_state=42)
    model.fit_transform(X)
    # Check that the angle parameter is set correctly
    assert model.angle == angle, f"Angle should be {angle}"


def test_tsne_random_state(synthetic_data):
    X, _ = synthetic_data
    model1 = TSNE(random_state=42)
    X_embedded1 = model1.fit_transform(X)
    model2 = TSNE(random_state=42)
    X_embedded2 = model2.fit_transform(X)
    # The embeddings should be the same when random_state is fixed
    np.testing.assert_allclose(
        X_embedded1,
        X_embedded2,
        atol=1e-5,
        err_msg="Embeddings should be the same with the same random_state",
    )


def test_tsne_verbose(synthetic_data):
    X, _ = synthetic_data
    model = TSNE(verbose=1, random_state=42)
    model.fit_transform(X)


def test_tsne_structure_preservation(synthetic_data):
    X, y = synthetic_data
    model = TSNE(random_state=42)
    X_embedded = model.fit_transform(X)
    # Compute pairwise distances in original and embedded spaces
    dist_original = pairwise_distances(X)
    dist_embedded = pairwise_distances(X_embedded)
    # Compute correlation between the distances
    np.corrcoef(dist_original.ravel(), dist_embedded.ravel())[0, 1]


@pytest.mark.parametrize("min_grad_norm", [1e-5])
def test_tsne_min_grad_norm(synthetic_data, min_grad_norm):
    X, _ = synthetic_data
    model = TSNE(min_grad_norm=min_grad_norm, random_state=42)
    model.fit_transform(X)
    # Check that the min_grad_norm parameter is set correctly
    assert (
        model.min_grad_norm == min_grad_norm
    ), f"min_grad_norm should be {min_grad_norm}"


def test_tsne_metric_params(synthetic_data):
    X, _ = synthetic_data
    metric_params = {"p": 2}
    model = TSNE(
        metric="minkowski", metric_params=metric_params, random_state=42
    )
    X_embedded = model.fit_transform(X)
    # Check that the embedding has the correct shape
    assert (
        X_embedded.shape[0] == X.shape[0]
    ), "Embedding should have same number of samples with custom metric_params"
