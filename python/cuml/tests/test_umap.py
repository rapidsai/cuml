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


# Please install UMAP before running the code
# use 'conda install -c conda-forge umap-learn' command to install it

import copy

import cupy as cp
import cupyx
import joblib
import numpy as np
import pytest
import scipy.sparse as scipy_sparse
import umap
from pylibraft.common import DeviceResourcesSNMG
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.manifold import trustworthiness
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

from cuml.internals import GraphBasedDimRedCallback, logger
from cuml.manifold.umap import UMAP as cuUMAP
from cuml.metrics import pairwise_distances
from cuml.testing.utils import (
    array_equal,
    quality_param,
    stress_param,
    unit_param,
)

dataset_names = ["iris", "digits", "wine", "blobs"]


@pytest.mark.parametrize(
    "nrows", [unit_param(500), quality_param(5000), stress_param(500000)]
)
@pytest.mark.parametrize(
    "n_feats", [unit_param(20), quality_param(100), stress_param(1000)]
)
@pytest.mark.parametrize("build_algo", ["brute_force_knn", "nn_descent"])
def test_blobs_cluster(nrows, n_feats, build_algo):

    data, labels = datasets.make_blobs(
        n_samples=nrows, n_features=n_feats, centers=5, random_state=0
    )
    embedding = cuUMAP(build_algo=build_algo).fit_transform(
        data, convert_dtype=True
    )

    if nrows < 500000:
        score = adjusted_rand_score(labels, KMeans(5).fit_predict(embedding))
        assert score == 1.0


@pytest.mark.parametrize(
    "nrows",
    [
        pytest.param(
            500,
            marks=[
                pytest.mark.unit,
                pytest.mark.xfail(
                    reason="https://github.com/rapidsai/cuvs/issues/184"
                ),
            ],
        ),
        quality_param(5000),
        stress_param(500000),
    ],
)
@pytest.mark.parametrize(
    "n_feats", [unit_param(10), quality_param(100), stress_param(1000)]
)
@pytest.mark.parametrize("build_algo", ["brute_force_knn", "nn_descent"])
def test_umap_fit_transform_score(nrows, n_feats, build_algo):

    n_samples = nrows
    n_features = n_feats

    data, labels = make_blobs(
        n_samples=n_samples, n_features=n_features, centers=10, random_state=42
    )

    model = umap.UMAP(n_neighbors=10, min_dist=0.1)
    cuml_model = cuUMAP(n_neighbors=10, min_dist=0.01, build_algo=build_algo)

    embedding = model.fit_transform(data)
    cuml_embedding = cuml_model.fit_transform(data, convert_dtype=True)

    assert not np.isnan(embedding).any()
    assert not np.isnan(cuml_embedding).any()

    if nrows < 500000:
        cuml_score = adjusted_rand_score(
            labels, KMeans(10).fit_predict(cuml_embedding)
        )
        score = adjusted_rand_score(labels, KMeans(10).fit_predict(embedding))

        assert array_equal(score, cuml_score, 1e-2, with_sign=True)


def test_supervised_umap_trustworthiness_on_iris():
    iris = datasets.load_iris()
    data = iris.data
    embedding = cuUMAP(
        n_neighbors=10, random_state=0, min_dist=0.01
    ).fit_transform(data, iris.target, convert_dtype=True)
    trust = trustworthiness(iris.data, embedding, n_neighbors=10)
    assert trust >= 0.97


def test_semisupervised_umap_trustworthiness_on_iris():
    iris = datasets.load_iris()
    data = iris.data
    target = iris.target.copy()
    target[25:75] = -1
    embedding = cuUMAP(
        n_neighbors=10, random_state=0, min_dist=0.01
    ).fit_transform(data, target, convert_dtype=True)

    trust = trustworthiness(iris.data, embedding, n_neighbors=10)
    assert trust >= 0.97


def test_umap_trustworthiness_on_iris():
    iris = datasets.load_iris()
    data = iris.data
    embedding = cuUMAP(
        n_neighbors=10, min_dist=0.01, random_state=0
    ).fit_transform(data, convert_dtype=True)
    trust = trustworthiness(iris.data, embedding, n_neighbors=10)
    assert trust >= 0.97


@pytest.mark.parametrize("target_metric", ["categorical", "euclidean"])
def test_umap_transform_on_iris(target_metric):

    iris = datasets.load_iris()

    iris_selection = np.random.RandomState(42).choice(
        [True, False], 150, replace=True, p=[0.75, 0.25]
    )
    data = iris.data[iris_selection]

    fitter = cuUMAP(
        n_neighbors=10,
        init="random",
        n_epochs=800,
        min_dist=0.01,
        random_state=42,
        target_metric=target_metric,
    )
    fitter.fit(data, convert_dtype=True)
    new_data = iris.data[~iris_selection]
    embedding = fitter.transform(new_data, convert_dtype=True)

    assert not np.isnan(embedding).any()

    trust = trustworthiness(new_data, embedding, n_neighbors=10)
    assert trust >= 0.85


@pytest.mark.parametrize("input_type", ["cupy", "scipy"])
@pytest.mark.parametrize("xform_method", ["fit", "fit_transform"])
@pytest.mark.parametrize("target_metric", ["categorical", "euclidean"])
def test_umap_transform_on_digits_sparse(
    target_metric, input_type, xform_method
):

    digits = datasets.load_digits()

    digits_selection = np.random.RandomState(42).choice(
        [True, False], 1797, replace=True, p=[0.75, 0.25]
    )

    if input_type == "cupy":
        sp_prefix = cupyx.scipy.sparse
    else:
        sp_prefix = scipy_sparse

    data = sp_prefix.csr_matrix(
        scipy_sparse.csr_matrix(digits.data[digits_selection])
    )

    fitter = cuUMAP(
        n_neighbors=15,
        verbose=logger.level_enum.info,
        init="random",
        n_epochs=0,
        min_dist=0.01,
        random_state=42,
        target_metric=target_metric,
    )

    new_data = sp_prefix.csr_matrix(
        scipy_sparse.csr_matrix(digits.data[~digits_selection])
    )

    if xform_method == "fit":
        fitter.fit(data, convert_dtype=True)
        embedding = fitter.transform(new_data, convert_dtype=True)
    else:
        embedding = fitter.fit_transform(new_data, convert_dtype=True)

    if input_type == "cupy":
        embedding = embedding.get()

    trust = trustworthiness(
        digits.data[~digits_selection], embedding, n_neighbors=15
    )
    assert trust >= 0.96


@pytest.mark.parametrize("target_metric", ["categorical", "euclidean"])
def test_umap_transform_on_digits(target_metric):

    digits = datasets.load_digits()

    digits_selection = np.random.RandomState(42).choice(
        [True, False], 1797, replace=True, p=[0.75, 0.25]
    )
    data = digits.data[digits_selection]

    fitter = cuUMAP(
        n_neighbors=15,
        verbose=logger.level_enum.debug,
        init="random",
        n_epochs=0,
        min_dist=0.01,
        random_state=42,
        target_metric=target_metric,
    )
    fitter.fit(data, convert_dtype=True)

    new_data = digits.data[~digits_selection]

    embedding = fitter.transform(new_data, convert_dtype=True)
    trust = trustworthiness(
        digits.data[~digits_selection], embedding, n_neighbors=15
    )
    assert trust >= 0.96


@pytest.mark.parametrize("target_metric", ["categorical", "euclidean"])
@pytest.mark.parametrize("name", dataset_names)
def test_umap_fit_transform_trust(name, target_metric):

    if name == "iris":
        iris = datasets.load_iris()
        data = iris.data
        labels = iris.target

    elif name == "digits":
        digits = datasets.load_digits(n_class=5)
        data = digits.data
        labels = digits.target

    elif name == "wine":
        wine = datasets.load_wine()
        data = wine.data
        labels = wine.target
    else:
        data, labels = make_blobs(
            n_samples=500, n_features=10, centers=10, random_state=42
        )

    model = umap.UMAP(
        n_neighbors=10, min_dist=0.01, target_metric=target_metric
    )
    cuml_model = cuUMAP(
        n_neighbors=10, min_dist=0.01, target_metric=target_metric
    )
    embedding = model.fit_transform(data)
    cuml_embedding = cuml_model.fit_transform(data, convert_dtype=True)

    trust = trustworthiness(data, embedding, n_neighbors=10)
    cuml_trust = trustworthiness(data, cuml_embedding, n_neighbors=10)

    assert array_equal(trust, cuml_trust, 1e-1, with_sign=True)


@pytest.mark.parametrize("target_metric", ["categorical", "euclidean"])
@pytest.mark.parametrize("name", [unit_param("digits")])
@pytest.mark.parametrize("nrows", [quality_param(5000), stress_param(500000)])
@pytest.mark.parametrize("n_feats", [quality_param(100), stress_param(1000)])
@pytest.mark.parametrize("should_downcast", [True])
@pytest.mark.parametrize("input_type", ["dataframe", "ndarray"])
@pytest.mark.parametrize("build_algo", ["brute_force_knn", "nn_descent"])
def test_umap_data_formats(
    input_type,
    should_downcast,
    nrows,
    n_feats,
    name,
    target_metric,
    build_algo,
):

    dtype = np.float32 if not should_downcast else np.float64
    n_samples = nrows
    n_feats = n_feats

    if name == "digits":
        # use the digits dataset for unit test
        digits = datasets.load_digits(n_class=9)
        X = digits["data"].astype(dtype)

    else:
        X, y = datasets.make_blobs(
            n_samples=n_samples, n_features=n_feats, random_state=0
        )

    umap = cuUMAP(
        n_neighbors=3,
        n_components=2,
        target_metric=target_metric,
        build_algo=build_algo,
    )

    embeds = umap.fit_transform(X)
    assert type(embeds) is np.ndarray


@pytest.mark.parametrize("target_metric", ["categorical", "euclidean"])
@pytest.mark.filterwarnings("ignore:(.*)connected(.*):UserWarning:sklearn[.*]")
@pytest.mark.parametrize("build_algo", ["brute_force_knn", "nn_descent"])
def test_umap_fit_transform_score_default(target_metric, build_algo):

    n_samples = 500
    n_features = 20

    data, labels = make_blobs(
        n_samples=n_samples, n_features=n_features, centers=10, random_state=42
    )

    model = umap.UMAP(target_metric=target_metric)
    cuml_model = cuUMAP(target_metric=target_metric, build_algo=build_algo)

    embedding = model.fit_transform(data)
    cuml_embedding = cuml_model.fit_transform(data, convert_dtype=True)

    cuml_score = adjusted_rand_score(
        labels, KMeans(10).fit_predict(cuml_embedding)
    )
    score = adjusted_rand_score(labels, KMeans(10).fit_predict(embedding))

    assert array_equal(score, cuml_score, 1e-2, with_sign=True)


@pytest.mark.parametrize("build_algo", ["brute_force_knn", "nn_descent"])
def test_umap_fit_transform_against_fit_and_transform(build_algo):

    n_samples = 500
    n_features = 20

    data, labels = make_blobs(
        n_samples=n_samples, n_features=n_features, centers=10, random_state=42
    )

    """
    First test the default option does not hash the input
    """

    cuml_model = cuUMAP(build_algo=build_algo)

    ft_embedding = cuml_model.fit_transform(data, convert_dtype=True)
    fit_embedding_same_input = cuml_model.transform(data, convert_dtype=True)

    assert joblib.hash(ft_embedding) != joblib.hash(fit_embedding_same_input)

    """
    Next, test explicitly enabling feature hashes the input
    """

    cuml_model = cuUMAP(hash_input=True)

    ft_embedding = cuml_model.fit_transform(data, convert_dtype=True)
    fit_embedding_same_input = cuml_model.transform(data, convert_dtype=True)

    assert joblib.hash(ft_embedding) == joblib.hash(fit_embedding_same_input)

    fit_embedding_diff_input = cuml_model.transform(
        data[1:], convert_dtype=True
    )
    assert joblib.hash(ft_embedding) != joblib.hash(fit_embedding_diff_input)


@pytest.mark.parametrize(
    "n_components,random_state",
    [
        unit_param(2, None),
        unit_param(2, 8),
        unit_param(2, np.random.RandomState(42)),
        unit_param(21, None),
        unit_param(21, np.random.RandomState(42)),
        unit_param(25, 8),
        unit_param(50, None),
        stress_param(50, 8),
    ],
)
def test_umap_fit_transform_reproducibility(n_components, random_state):

    n_samples = 8000
    n_features = 200

    if random_state is None:
        n_components *= 2

    data, labels = make_blobs(
        n_samples=n_samples, n_features=n_features, centers=10, random_state=42
    )

    def get_embedding(n_components, random_state):
        reducer = cuUMAP(
            init="random", n_components=n_components, random_state=random_state
        )
        return reducer.fit_transform(data, convert_dtype=True)

    state = copy.deepcopy(random_state)
    cuml_embedding1 = get_embedding(n_components, state)
    state = copy.deepcopy(random_state)
    cuml_embedding2 = get_embedding(n_components, state)

    assert not np.isnan(cuml_embedding1).any()
    assert not np.isnan(cuml_embedding2).any()

    # Reproducibility threshold raised until intermittent failure is fixed
    # Ref: https://github.com/rapidsai/cuml/issues/1903
    mean_diff = np.mean(np.abs(cuml_embedding1 - cuml_embedding2))
    if random_state is not None:
        assert mean_diff == 0.0
    else:
        assert mean_diff > 0.5


@pytest.mark.parametrize(
    "n_components,random_state",
    [
        unit_param(2, None),
        unit_param(2, 8),
        unit_param(2, np.random.RandomState(42)),
        unit_param(21, None),
        unit_param(25, 8),
        unit_param(25, np.random.RandomState(42)),
        unit_param(50, None),
        stress_param(50, 8),
    ],
)
def test_umap_transform_reproducibility(n_components, random_state):

    n_samples = 5000
    n_features = 200

    if random_state is None:
        n_components *= 2

    data, labels = make_blobs(
        n_samples=n_samples, n_features=n_features, centers=10, random_state=42
    )

    selection = np.random.RandomState(42).choice(
        [True, False], n_samples, replace=True, p=[0.5, 0.5]
    )
    fit_data = data[selection]
    transform_data = data[~selection]

    def get_embedding(n_components, random_state):
        reducer = cuUMAP(
            init="random", n_components=n_components, random_state=random_state
        )
        reducer.fit(fit_data, convert_dtype=True)
        return reducer.transform(transform_data, convert_dtype=True)

    state = copy.deepcopy(random_state)
    cuml_embedding1 = get_embedding(n_components, state)
    state = copy.deepcopy(random_state)
    cuml_embedding2 = get_embedding(n_components, state)

    assert not np.isnan(cuml_embedding1).any()
    assert not np.isnan(cuml_embedding2).any()

    # Reproducibility threshold raised until intermittent failure is fixed
    # Ref: https://github.com/rapidsai/cuml/issues/1903
    mean_diff = np.mean(np.abs(cuml_embedding1 - cuml_embedding2))
    if random_state is not None:
        assert mean_diff == 0.0
    else:
        assert mean_diff > 0.5


def test_umap_fit_transform_trustworthiness_with_consistency_enabled():
    iris = datasets.load_iris()
    data = iris.data
    algo = cuUMAP(
        n_neighbors=10,
        min_dist=0.01,
        init="random",
        random_state=42,
    )
    embedding = algo.fit_transform(data, convert_dtype=True)
    trust = trustworthiness(iris.data, embedding, n_neighbors=10)
    assert trust >= 0.97


def test_umap_transform_trustworthiness_with_consistency_enabled():
    iris = datasets.load_iris()
    data = iris.data
    selection = np.random.RandomState(42).choice(
        [True, False], data.shape[0], replace=True, p=[0.5, 0.5]
    )
    fit_data = data[selection]
    transform_data = data[~selection]
    model = cuUMAP(
        n_neighbors=10,
        min_dist=0.01,
        init="random",
        random_state=42,
    )
    model.fit(fit_data, convert_dtype=True)
    embedding = model.transform(transform_data, convert_dtype=True)
    trust = trustworthiness(transform_data, embedding, n_neighbors=10)
    assert trust >= 0.92


@pytest.mark.filterwarnings("ignore:(.*)zero(.*)::scipy[.*]|umap[.*]")
@pytest.mark.parametrize("build_algo", ["brute_force_knn", "nn_descent"])
def test_exp_decay_params(build_algo):
    def compare_exp_decay_params(a=None, b=None, min_dist=0.1, spread=1.0):
        cuml_model = cuUMAP(
            a=a, b=b, min_dist=min_dist, spread=spread, build_algo=build_algo
        )
        cuml_a = cuml_model.a
        cuml_b = cuml_model.b
        skl_model = umap.UMAP(a=a, b=b, min_dist=min_dist, spread=spread)
        skl_model.fit(np.zeros((1, 1)))
        sklearn_a, sklearn_b = skl_model._a, skl_model._b

        assert abs(cuml_a) - abs(sklearn_a) < 1e-6
        assert abs(cuml_b) - abs(sklearn_b) < 1e-6

    compare_exp_decay_params(min_dist=0.1, spread=1.0)
    compare_exp_decay_params(a=0.5, b=2.0)
    compare_exp_decay_params(a=0.5)
    compare_exp_decay_params(b=0.5)
    compare_exp_decay_params(min_dist=0.1, spread=10.0)


@pytest.mark.parametrize("n_neighbors", [5, 15])
@pytest.mark.parametrize("build_algo", ["brute_force_knn", "nn_descent"])
def test_umap_knn_graph(n_neighbors, build_algo):
    data, labels = datasets.make_blobs(
        n_samples=2000, n_features=10, centers=5, random_state=0
    )
    data = data.astype(np.float32)

    def fit_transform_embed(knn_graph=None):
        model = cuUMAP(
            random_state=42,
            init="random",
            n_neighbors=n_neighbors,
            build_algo=build_algo,
        )
        return model.fit_transform(
            data, knn_graph=knn_graph, convert_dtype=True
        )

    def transform_embed(knn_graph=None):
        model = cuUMAP(
            random_state=42,
            init="random",
            n_neighbors=n_neighbors,
            build_algo=build_algo,
        )
        model.fit(data, knn_graph=knn_graph, convert_dtype=True)
        return model.transform(data, convert_dtype=True)

    def test_trustworthiness(embedding):
        trust = trustworthiness(data, embedding, n_neighbors=n_neighbors)
        assert trust >= 0.92

    def test_equality(e1, e2):
        mean_diff = np.mean(np.abs(e1 - e2))
        assert mean_diff < 1.0

    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(data)
    knn_graph = neigh.kneighbors_graph(data, mode="distance")

    embedding1 = fit_transform_embed(None)
    embedding2 = fit_transform_embed(knn_graph.tocsr())
    embedding3 = fit_transform_embed(knn_graph.tocoo())
    embedding4 = fit_transform_embed(knn_graph.tocsc())
    embedding5 = transform_embed(knn_graph.tocsr())
    embedding6 = transform_embed(knn_graph.tocoo())
    embedding7 = transform_embed(knn_graph.tocsc())

    test_trustworthiness(embedding1)
    test_trustworthiness(embedding2)
    test_trustworthiness(embedding3)
    test_trustworthiness(embedding4)
    test_trustworthiness(embedding5)
    test_trustworthiness(embedding6)
    test_trustworthiness(embedding7)

    test_equality(embedding2, embedding3)
    test_equality(embedding3, embedding4)
    test_equality(embedding5, embedding6)
    test_equality(embedding6, embedding7)


@pytest.mark.parametrize(
    "precomputed_type", ["knn_graph", "tuple", "pairwise"]
)
@pytest.mark.parametrize(
    "sparse_input,build_algo",
    [
        (False, "brute_force_knn"),
        (True, "brute_force_knn"),
        (False, "nn_descent"),
    ],
)
def test_umap_precomputed_knn(precomputed_type, sparse_input, build_algo):
    data, labels = make_blobs(
        n_samples=2000, n_features=10, centers=5, random_state=0
    )
    data = data.astype(np.float32)

    if sparse_input:
        sparsification = np.random.choice(
            [0.0, 1.0], p=[0.1, 0.9], size=data.shape
        )
        data = np.multiply(data, sparsification)
        data = scipy_sparse.csr_matrix(data)

    n_neighbors = 8

    if precomputed_type == "knn_graph":
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(data)
        precomputed_knn = nn.kneighbors_graph(data, mode="distance")
    elif precomputed_type == "tuple":
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(data)
        precomputed_knn = nn.kneighbors(data, return_distance=True)
        precomputed_knn = (precomputed_knn[1], precomputed_knn[0])
    elif precomputed_type == "pairwise":
        precomputed_knn = pairwise_distances(data)

    model = cuUMAP(
        n_neighbors=n_neighbors,
        precomputed_knn=precomputed_knn,
        build_algo=build_algo,
    )
    embedding = model.fit_transform(data)
    trust = trustworthiness(data, embedding, n_neighbors=n_neighbors)
    assert trust >= 0.92


def correctness_sparse(a, b, atol=0.1, rtol=0.2, threshold=0.95):
    n_ref_zeros = (a == 0).sum()
    n_ref_non_zero_elms = a.size - n_ref_zeros
    n_correct = (cp.abs(a - b) <= (atol + rtol * cp.abs(b))).sum()
    correctness = (n_correct - n_ref_zeros) / n_ref_non_zero_elms
    return correctness >= threshold


@pytest.mark.parametrize("n_rows", [200, 800])
@pytest.mark.parametrize("n_features", [8, 32])
@pytest.mark.parametrize("n_neighbors", [8, 16])
@pytest.mark.filterwarnings(
    "ignore:Spectral initialisation failed.*:UserWarning"
)
@pytest.mark.filterwarnings(
    "ignore:Graph is not fully connected.*:UserWarning"
)
def test_fuzzy_simplicial_set(n_rows, n_features, n_neighbors):
    n_clusters = 30
    random_state = 42

    X, _ = make_blobs(
        n_samples=n_rows,
        centers=n_clusters,
        n_features=n_features,
        random_state=random_state,
    )

    model = cuUMAP(n_neighbors=n_neighbors)
    model.fit(X)
    cu_fss_graph = model.graph_

    model = umap.UMAP(n_neighbors=n_neighbors)
    model.fit(X)
    ref_fss_graph = model.graph_

    cu_fss_graph = cp.array(cu_fss_graph.todense())
    ref_fss_graph = cp.array(ref_fss_graph.todense())
    assert correctness_sparse(
        ref_fss_graph, cu_fss_graph, atol=0.1, rtol=0.2, threshold=0.95
    )


@pytest.mark.parametrize(
    "metric,build_algo,supported",
    [
        ("l2", "brute_force_knn", True),
        ("euclidean", "brute_force_knn", True),
        ("sqeuclidean", "brute_force_knn", True),
        ("l1", "brute_force_knn", True),
        ("manhattan", "brute_force_knn", True),
        ("minkowski", "brute_force_knn", True),
        ("chebyshev", "brute_force_knn", True),
        ("cosine", "brute_force_knn", True),
        ("correlation", "brute_force_knn", True),
        ("jaccard", "brute_force_knn", False),
        ("hamming", "brute_force_knn", True),
        ("canberra", "brute_force_knn", True),
        ("l2", "nn_descent", True),
        ("euclidean", "nn_descent", True),
        ("sqeuclidean", "nn_descent", True),
        ("l1", "nn_descent", False),
        ("manhattan", "nn_descent", False),
        ("minkowski", "nn_descent", False),
        ("chebyshev", "nn_descent", False),
        ("cosine", "nn_descent", True),
        ("correlation", "nn_descent", False),
        ("jaccard", "nn_descent", False),
        ("hamming", "nn_descent", False),
        ("canberra", "nn_descent", False),
    ],
)
@pytest.mark.filterwarnings(
    "ignore:gradient function is not yet implemented.*:UserWarning"
)
def test_umap_distance_metrics_fit_transform_trust(
    metric, build_algo, supported
):
    data, labels = make_blobs(
        n_samples=500, n_features=64, centers=5, random_state=42
    )

    if metric == "jaccard":
        data = data >= 0

    umap_model = umap.UMAP(
        n_neighbors=10, min_dist=0.01, metric=metric, init="random"
    )
    cuml_model = cuUMAP(
        n_neighbors=10,
        min_dist=0.01,
        metric=metric,
        init="random",
        build_algo=build_algo,
    )
    if not supported:
        with pytest.raises(NotImplementedError):
            cuml_model.fit_transform(data)
        return

    umap_embedding = umap_model.fit_transform(data)

    cuml_embedding = cuml_model.fit_transform(data)

    umap_trust = trustworthiness(
        data, umap_embedding, n_neighbors=10, metric=metric
    )
    cuml_trust = trustworthiness(
        data, cuml_embedding, n_neighbors=10, metric=metric
    )

    assert array_equal(umap_trust, cuml_trust, 0.05, with_sign=True)


@pytest.mark.parametrize(
    "metric,supported,umap_learn_supported",
    [
        ("l2", True, False),
        ("euclidean", True, True),
        ("sqeuclidean", True, False),
        ("l1", True, True),
        ("manhattan", True, True),
        ("minkowski", True, True),
        ("chebyshev", True, True),
        ("cosine", True, True),
        ("correlation", True, True),
        ("jaccard", True, True),
        ("hamming", True, True),
        ("canberra", True, True),
    ],
)
@pytest.mark.filterwarnings(
    "ignore:gradient function is not yet implemented.*:UserWarning"
)
def test_umap_distance_metrics_fit_transform_trust_on_sparse_input(
    metric, supported, umap_learn_supported
):
    if metric == "jaccard":
        n_features = 1000
    else:
        n_features = 64

    data, labels = make_blobs(
        n_samples=1000, n_features=n_features, centers=5, random_state=42
    )

    data_selection = np.random.RandomState(42).choice(
        [True, False], 1000, replace=True, p=[0.75, 0.25]
    )

    if metric == "jaccard":
        data = data >= 0

    new_data = scipy_sparse.csr_matrix(data[~data_selection])

    if umap_learn_supported:
        umap_model = umap.UMAP(
            n_neighbors=10, min_dist=0.01, metric=metric, init="random"
        )
        umap_embedding = umap_model.fit_transform(new_data)
        umap_trust = trustworthiness(
            data[~data_selection],
            umap_embedding,
            n_neighbors=10,
            metric=metric,
        )

    cuml_model = cuUMAP(
        n_neighbors=10, min_dist=0.01, metric=metric, init="random"
    )

    if not supported:
        with pytest.raises(NotImplementedError):
            cuml_model.fit_transform(new_data)
        return

    cuml_embedding = cuml_model.fit_transform(new_data)
    cuml_trust = trustworthiness(
        data[~data_selection], cuml_embedding, n_neighbors=10, metric=metric
    )

    if umap_learn_supported:
        assert array_equal(umap_trust, cuml_trust, 0.05, with_sign=True)


@pytest.mark.parametrize("num_clusters", [3, 5])
@pytest.mark.parametrize("fit_then_transform", [False, True])
@pytest.mark.parametrize("metric", ["l2", "sqeuclidean", "cosine"])
@pytest.mark.parametrize("do_snmg", [True, False])
def test_umap_trustworthiness_on_batch_nnd(
    num_clusters, fit_then_transform, metric, do_snmg
):
    digits = datasets.load_digits()

    umap_handle = None
    if do_snmg:
        umap_handle = DeviceResourcesSNMG()

    cuml_model = cuUMAP(
        handle=umap_handle,
        n_neighbors=10,
        min_dist=0.01,
        build_algo="nn_descent",
        build_kwds={"nnd_n_clusters": num_clusters},
        metric=metric,
    )

    if fit_then_transform:
        cuml_model.fit(digits.data, convert_dtype=True)
        cuml_embedding = cuml_model.transform(digits.data)
    else:
        cuml_embedding = cuml_model.fit_transform(
            digits.data, convert_dtype=True
        )

    cuml_trust = trustworthiness(
        digits.data, cuml_embedding, n_neighbors=10, metric=metric
    )

    assert cuml_trust > 0.9


def test_callback():
    class Callback(GraphBasedDimRedCallback):
        preprocess_event, epoch_event, train_event = False, 0, False

        def __init__(self, skip_init=False):
            if not skip_init:
                super().__init__()

        def check(self):
            assert self.preprocess_event
            assert self.epoch_event > 10
            assert self.train_event

        def on_preprocess_end(self, embeddings):
            self.preprocess_event = True

        def on_epoch_end(self, embeddings):
            self.epoch_event += 1

        def on_train_end(self, embeddings):
            self.train_event = True

    digits = datasets.load_digits()

    callback = Callback()
    reducer = cuUMAP(n_components=2, callback=callback)
    reducer.fit(digits.data)
    callback.check()


def test_umap_small_fit_large_transform():
    data, _ = make_blobs(
        n_samples=10_000, n_features=8, centers=5, random_state=0
    )
    train, infer = data[:1000], data[1000:]

    model = cuUMAP(build_algo="brute_force_knn", init="random")
    model.fit(train)
    embeddings = model.transform(infer)

    trust = trustworthiness(infer, embeddings, n_neighbors=10)
    assert trust >= 0.9


@pytest.mark.parametrize("n_neighbors", [5, 15])
@pytest.mark.parametrize("n_components", [2, 5])
def test_umap_outliers(n_neighbors, n_components):
    all_neighbors = pytest.importorskip("cuvs.neighbors.all_neighbors")
    nn_descent = pytest.importorskip("cuvs.neighbors.nn_descent")

    k = n_neighbors
    n_rows = 50_000

    # This dataset was specifically chosen because UMAP produces outliers
    # on this dataset before the outlier fix.
    data, _ = make_moons(n_samples=n_rows, noise=0.0, random_state=42)
    data = data.astype(np.float32)

    # precompute knn for faster testing with CPU UMAP
    nn_descent_params = nn_descent.IndexParams(
        metric="euclidean",
        graph_degree=k,
        intermediate_graph_degree=k * 2,
    )
    params = all_neighbors.AllNeighborsParams(
        algo="nn_descent",
        metric="euclidean",
        nn_descent_params=nn_descent_params,
    )
    indices, distances = all_neighbors.build(
        data,
        k,
        params,
        distances=cp.empty((n_rows, k), dtype=cp.float32),
    )
    indices = cp.asnumpy(indices)
    distances = cp.asnumpy(distances)

    gpu_umap = cuUMAP(
        precomputed_knn=(indices, distances),
        build_algo="nn_descent",
        init="spectral",
        n_neighbors=n_neighbors,
        n_components=n_components,
    )
    gpu_umap_embeddings = gpu_umap.fit_transform(data)

    cpu_umap = umap.UMAP(
        precomputed_knn=(indices, distances),
        init="spectral",
        n_neighbors=n_neighbors,
        n_components=n_components,
    )
    cpu_umap_embeddings = cpu_umap.fit_transform(data)

    # test to see if there are values in the final embedding that are too out of range
    # compared to the cpu umap output.
    lower_bound = 3 * cpu_umap_embeddings.min()
    upper_bound = 3 * cpu_umap_embeddings.max()

    assert np.all(
        (gpu_umap_embeddings >= lower_bound)
        & (gpu_umap_embeddings <= upper_bound)
    )


@pytest.mark.parametrize("precomputed_type", ["tuple", "knn_graph"])
@pytest.mark.parametrize("k_provided,k_requested", [(15, 10), (20, 8)])
def test_umap_precomputed_knn_trimming(
    precomputed_type, k_provided, k_requested
):
    """
    Test that precomputed KNN data with more neighbors than requested
    is automatically trimmed instead of raising an error.
    """
    data, labels = make_blobs(
        n_samples=500, n_features=10, centers=5, random_state=0
    )
    data = data.astype(np.float32)

    # Build KNN graph with more neighbors than we'll request
    nn = NearestNeighbors(n_neighbors=k_provided)
    nn.fit(data)

    if precomputed_type == "tuple":
        distances, indices = nn.kneighbors(data, return_distance=True)
        precomputed_knn = (indices, distances)
    elif precomputed_type == "knn_graph":
        precomputed_knn = nn.kneighbors_graph(data, mode="distance")

    # This should work now - the excess neighbors should be trimmed
    model = cuUMAP(
        n_neighbors=k_requested,
        precomputed_knn=precomputed_knn,
        random_state=42,
        init="random",
    )
    embedding = model.fit_transform(data)

    # Verify the embedding is valid
    assert embedding.shape == (data.shape[0], 2)
    assert not np.isnan(embedding).any()

    # Verify trustworthiness with the requested number of neighbors
    trust = trustworthiness(data, embedding, n_neighbors=k_requested)
    assert trust >= 0.85


@pytest.mark.parametrize("precomputed_type", ["tuple", "knn_graph"])
def test_umap_precomputed_knn_insufficient_neighbors(precomputed_type):
    """
    Test that precomputed KNN data with fewer neighbors than requested
    raises an appropriate error.
    """
    data, labels = make_blobs(
        n_samples=500, n_features=10, centers=5, random_state=0
    )
    data = data.astype(np.float32)

    k_provided = 5
    k_requested = 10

    # Build KNN graph with fewer neighbors than we'll request
    nn = NearestNeighbors(n_neighbors=k_provided)
    nn.fit(data)

    if precomputed_type == "tuple":
        distances, indices = nn.kneighbors(data, return_distance=True)
        precomputed_knn = (indices, distances)
    elif precomputed_type == "knn_graph":
        precomputed_knn = nn.kneighbors_graph(data, mode="distance")

    # This should raise an error during initialization
    with pytest.raises(ValueError, match=".*fewer neighbors.*"):
        cuUMAP(
            n_neighbors=k_requested,
            precomputed_knn=precomputed_knn,
            random_state=42,
            init="random",
        )
