#
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import trustworthiness
from umap import UMAP


@pytest.fixture(scope="module")
def manifold_data():
    X, _ = make_swiss_roll(n_samples=100, noise=0.05, random_state=42)
    return X


@pytest.mark.parametrize("n_neighbors", [5])
def test_umap_n_neighbors(manifold_data, n_neighbors):
    X = manifold_data
    umap = UMAP(n_neighbors=n_neighbors, random_state=42)
    X_embedded = umap.fit_transform(X)
    trust = trustworthiness(X, X_embedded, n_neighbors=5)
    print(f"Trustworthiness with n_neighbors={n_neighbors}: {trust}")


@pytest.mark.parametrize("min_dist", [0.0, 0.5])
def test_umap_min_dist(manifold_data, min_dist):
    X = manifold_data
    umap = UMAP(min_dist=min_dist, random_state=42)
    X_embedded = umap.fit_transform(X)
    trust = trustworthiness(X, X_embedded, n_neighbors=5)
    print(f"Trustworthiness with min_dist={min_dist}: {trust}")


@pytest.mark.parametrize(
    "metric",
    [
        "euclidean",
        "manhattan",
        "chebyshev",
        "cosine",
        # These metrics are currently not supported in cuml,
        # we test them here to make sure no exception is raised
        "sokalsneath",
        "rogerstanimoto",
        "sokalmichener",
        "yule",
        "ll_dirichlet",
        "russellrao",
        "kulsinski",
        "dice",
        "wminkowski",
        "mahalanobis",
        "haversine",
    ],
)
@pytest.mark.filterwarnings(
    "ignore:gradient function is not yet implemented:UserWarning"
)
@pytest.mark.filterwarnings(
    "ignore:n_jobs value .* overridden to .* by setting random_state:UserWarning"
)
@pytest.mark.filterwarnings(
    "ignore:overflow encountered in cast:RuntimeWarning"
)
def test_umap_metric(manifold_data, metric):
    X = manifold_data
    # haversine only works for 2D data
    if metric == "haversine":
        X = X[:, :2]

    umap = UMAP(metric=metric, random_state=42)
    X_embedded = umap.fit_transform(X)
    trust = trustworthiness(X, X_embedded, n_neighbors=5)
    print(f"Trustworthiness with metric={metric}: {trust}")


@pytest.mark.parametrize("n_components", [2, 3])
def test_umap_n_components(manifold_data, n_components):
    X = manifold_data
    umap = UMAP(n_components=n_components, random_state=42)
    X_embedded = umap.fit_transform(X)
    trust = trustworthiness(X, X_embedded, n_neighbors=5)
    print(f"Trustworthiness with n_components={n_components}: {trust}")


@pytest.mark.parametrize("spread", [0.5, 1.5])
def test_umap_spread(manifold_data, spread):
    X = manifold_data
    umap = UMAP(spread=spread, random_state=42)
    X_embedded = umap.fit_transform(X)
    trust = trustworthiness(X, X_embedded, n_neighbors=5)
    print(f"Trustworthiness with spread={spread}: {trust}")


@pytest.mark.parametrize("negative_sample_rate", [5])
def test_umap_negative_sample_rate(manifold_data, negative_sample_rate):
    X = manifold_data
    umap = UMAP(negative_sample_rate=negative_sample_rate, random_state=42)
    X_embedded = umap.fit_transform(X)
    trust = trustworthiness(X, X_embedded, n_neighbors=5)
    print(
        f"Trustworthiness with negative_sample_rate={negative_sample_rate}: {trust}"
    )


@pytest.mark.parametrize("learning_rate", [0.1, 10.0])
def test_umap_learning_rate(manifold_data, learning_rate):
    X = manifold_data
    umap = UMAP(learning_rate=learning_rate, random_state=42)
    X_embedded = umap.fit_transform(X)
    trust = trustworthiness(X, X_embedded, n_neighbors=5)
    print(f"Trustworthiness with learning_rate={learning_rate}: {trust}")


@pytest.mark.parametrize("init", ["spectral", "random"])
def test_umap_init(manifold_data, init):
    X = manifold_data
    umap = UMAP(init=init, random_state=42)
    X_embedded = umap.fit_transform(X)
    trust = trustworthiness(X, X_embedded, n_neighbors=5)
    print(f"Trustworthiness with init={init}: {trust}")


@pytest.mark.parametrize("n_epochs", [100, 200, 500])
def test_umap_n_epochs(manifold_data, n_epochs):
    X = manifold_data
    umap = UMAP(n_epochs=n_epochs, random_state=42)
    X_embedded = umap.fit_transform(X)
    trust = trustworthiness(X, X_embedded, n_neighbors=5)
    print(f"Trustworthiness with n_epochs={n_epochs}: {trust}")


@pytest.mark.parametrize("local_connectivity", [1, 2, 5])
def test_umap_local_connectivity(manifold_data, local_connectivity):
    X = manifold_data
    umap = UMAP(local_connectivity=local_connectivity, random_state=42)
    X_embedded = umap.fit_transform(X)
    trust = trustworthiness(X, X_embedded, n_neighbors=5)
    print(
        f"Trustworthiness with local_connectivity={local_connectivity}: {trust}"
    )


@pytest.mark.parametrize("repulsion_strength", [0.5, 1.0, 2.0])
def test_umap_repulsion_strength(manifold_data, repulsion_strength):
    X = manifold_data
    umap = UMAP(repulsion_strength=repulsion_strength, random_state=42)
    X_embedded = umap.fit_transform(X)
    trust = trustworthiness(X, X_embedded, n_neighbors=5)
    print(
        f"Trustworthiness with repulsion_strength={repulsion_strength}: {trust}"
    )


@pytest.mark.parametrize("metric_kwds", [{"p": 1}, {"p": 2}, {"p": 3}])
def test_umap_metric_kwds(manifold_data, metric_kwds):
    X = manifold_data
    umap = UMAP(metric="minkowski", metric_kwds=metric_kwds, random_state=42)
    X_embedded = umap.fit_transform(X)
    trust = trustworthiness(X, X_embedded, n_neighbors=5)
    print(f"Trustworthiness with metric_kwds={metric_kwds}: {trust}")


@pytest.mark.parametrize("angular_rp_forest", [True, False])
def test_umap_angular_rp_forest(manifold_data, angular_rp_forest):
    X = manifold_data
    umap = UMAP(angular_rp_forest=angular_rp_forest, random_state=42)
    X_embedded = umap.fit_transform(X)
    trust = trustworthiness(X, X_embedded, n_neighbors=5)
    print(
        f"Trustworthiness with angular_rp_forest={angular_rp_forest}: {trust}"
    )


@pytest.mark.parametrize("densmap", [True, False])
@pytest.mark.filterwarnings(
    "ignore:n_jobs value .* overridden to .* by setting random_state:UserWarning"
)
def test_umap_densmap(manifold_data, densmap):
    X = manifold_data
    umap = UMAP(densmap=densmap, random_state=42)
    X_embedded = umap.fit_transform(X)
    trust = trustworthiness(X, X_embedded, n_neighbors=5)
    print(f"Trustworthiness with densmap={densmap}: {trust}")


@pytest.mark.parametrize("output_metric", ["euclidean", "manhattan"])
def test_umap_output_metric(manifold_data, output_metric):
    X = manifold_data
    umap = UMAP(output_metric=output_metric, random_state=42)
    X_embedded = umap.fit_transform(X)
    trust = trustworthiness(X, X_embedded, n_neighbors=5)
    print(f"Trustworthiness with output_metric={output_metric}: {trust}")
