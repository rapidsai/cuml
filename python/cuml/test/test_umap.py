# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

import numpy as np
import pytest
import umap
import copy

import cupyx
import scipy.sparse

from cuml.manifold.umap import UMAP as cuUMAP
from cuml.test.utils import array_equal, unit_param, \
    quality_param, stress_param
from sklearn.neighbors import NearestNeighbors

import joblib

from cuml.common import logger

from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.manifold import trustworthiness
from sklearn.metrics import adjusted_rand_score

dataset_names = ['iris', 'digits', 'wine', 'blobs']


@pytest.mark.parametrize('nrows', [unit_param(500), quality_param(5000),
                         stress_param(500000)])
@pytest.mark.parametrize('n_feats', [unit_param(20), quality_param(100),
                         stress_param(1000)])
def test_blobs_cluster(nrows, n_feats):

    data, labels = datasets.make_blobs(
        n_samples=nrows, n_features=n_feats, centers=5, random_state=0)
    embedding = cuUMAP().fit_transform(data, convert_dtype=True)

    if nrows < 500000:
        score = adjusted_rand_score(labels,
                                    KMeans(5).fit_predict(embedding))
        assert score == 1.0


@pytest.mark.parametrize('nrows', [unit_param(500), quality_param(5000),
                         stress_param(500000)])
@pytest.mark.parametrize('n_feats', [unit_param(10), quality_param(100),
                         stress_param(1000)])
def test_umap_fit_transform_score(nrows, n_feats):

    n_samples = nrows
    n_features = n_feats

    data, labels = make_blobs(n_samples=n_samples, n_features=n_features,
                              centers=10, random_state=42)

    model = umap.UMAP(n_neighbors=10, min_dist=0.1)
    cuml_model = cuUMAP(n_neighbors=10, min_dist=0.01)

    embedding = model.fit_transform(data)
    cuml_embedding = cuml_model.fit_transform(data, convert_dtype=True)

    assert not np.isnan(embedding).any()
    assert not np.isnan(cuml_embedding).any()

    if nrows < 500000:
        cuml_score = adjusted_rand_score(labels,
                                         KMeans(10).fit_predict(
                                             cuml_embedding))
        score = adjusted_rand_score(labels,
                                    KMeans(10).fit_predict(embedding))

        assert array_equal(score, cuml_score, 1e-2, with_sign=True)


def test_supervised_umap_trustworthiness_on_iris():
    iris = datasets.load_iris()
    data = iris.data
    embedding = cuUMAP(n_neighbors=10, random_state=0,
                       min_dist=0.01).fit_transform(
        data, iris.target, convert_dtype=True)
    trust = trustworthiness(iris.data, embedding, n_neighbors=10)
    assert trust >= 0.97


def test_semisupervised_umap_trustworthiness_on_iris():
    iris = datasets.load_iris()
    data = iris.data
    target = iris.target.copy()
    target[25:75] = -1
    embedding = cuUMAP(n_neighbors=10, random_state=0,
                       min_dist=0.01).fit_transform(
        data, target, convert_dtype=True)

    trust = trustworthiness(iris.data, embedding, n_neighbors=10)
    assert trust >= 0.97


def test_umap_trustworthiness_on_iris():
    iris = datasets.load_iris()
    data = iris.data
    embedding = cuUMAP(n_neighbors=10, min_dist=0.01,
                       random_state=0).fit_transform(
        data, convert_dtype=True)
    trust = trustworthiness(iris.data, embedding, n_neighbors=10)
    assert trust >= 0.97


@pytest.mark.parametrize('target_metric', ["categorical", "euclidean"])
def test_umap_transform_on_iris(target_metric):

    iris = datasets.load_iris()

    iris_selection = np.random.RandomState(42).choice(
        [True, False], 150, replace=True, p=[0.75, 0.25])
    data = iris.data[iris_selection]

    fitter = cuUMAP(n_neighbors=10, init="random", n_epochs=800, min_dist=0.01,
                    random_state=42, target_metric=target_metric)
    fitter.fit(data, convert_dtype=True)
    new_data = iris.data[~iris_selection]
    embedding = fitter.transform(new_data, convert_dtype=True)

    assert not np.isnan(embedding).any()

    trust = trustworthiness(new_data, embedding, n_neighbors=10)
    assert trust >= 0.85


@pytest.mark.parametrize('input_type', ['cupy', 'scipy'])
@pytest.mark.parametrize('xform_method', ['fit', 'fit_transform'])
@pytest.mark.parametrize('target_metric', ["categorical", "euclidean"])
def test_umap_transform_on_digits_sparse(target_metric, input_type,
                                         xform_method):

    digits = datasets.load_digits()

    digits_selection = np.random.RandomState(42).choice(
        [True, False], 1797, replace=True, p=[0.75, 0.25])

    if input_type == 'cupy':
        sp_prefix = cupyx.scipy.sparse
    else:
        sp_prefix = scipy.sparse

    data = sp_prefix.csr_matrix(
        scipy.sparse.csr_matrix(digits.data[digits_selection]))

    fitter = cuUMAP(n_neighbors=15,
                    verbose=logger.level_info,
                    init="random",
                    n_epochs=0,
                    min_dist=0.01,
                    random_state=42,
                    target_metric=target_metric)

    new_data = sp_prefix.csr_matrix(
        scipy.sparse.csr_matrix(digits.data[~digits_selection]))

    if xform_method == 'fit':
        fitter.fit(data, convert_dtype=True)
        embedding = fitter.transform(new_data, convert_dtype=True)
    else:
        embedding = fitter.fit_transform(new_data, convert_dtype=True)

    if input_type == 'cupy':
        embedding = embedding.get()

    trust = trustworthiness(digits.data[~digits_selection], embedding,
                            n_neighbors=15)
    assert trust >= 0.96


@pytest.mark.parametrize('target_metric', ["categorical", "euclidean"])
def test_umap_transform_on_digits(target_metric):

    digits = datasets.load_digits()

    digits_selection = np.random.RandomState(42).choice(
        [True, False], 1797, replace=True, p=[0.75, 0.25])
    data = digits.data[digits_selection]

    fitter = cuUMAP(n_neighbors=15,
                    verbose=logger.level_debug,
                    init="random",
                    n_epochs=0,
                    min_dist=0.01,
                    random_state=42,
                    target_metric=target_metric)
    fitter.fit(data, convert_dtype=True)

    new_data = digits.data[~digits_selection]

    embedding = fitter.transform(new_data, convert_dtype=True)
    trust = trustworthiness(digits.data[~digits_selection], embedding,
                            n_neighbors=15)
    assert trust >= 0.96


@pytest.mark.parametrize('target_metric', ["categorical", "euclidean"])
@pytest.mark.parametrize('name', dataset_names)
def test_umap_fit_transform_trust(name, target_metric):

    if name == 'iris':
        iris = datasets.load_iris()
        data = iris.data
        labels = iris.target

    elif name == 'digits':
        digits = datasets.load_digits(n_class=5)
        data = digits.data
        labels = digits.target

    elif name == 'wine':
        wine = datasets.load_wine()
        data = wine.data
        labels = wine.target
    else:
        data, labels = make_blobs(n_samples=500, n_features=10,
                                  centers=10, random_state=42)

    model = umap.UMAP(n_neighbors=10, min_dist=0.01,
                      target_metric=target_metric)
    cuml_model = cuUMAP(n_neighbors=10, min_dist=0.01,
                        target_metric=target_metric)
    embedding = model.fit_transform(data)
    cuml_embedding = cuml_model.fit_transform(data, convert_dtype=True)

    trust = trustworthiness(data, embedding, n_neighbors=10)
    cuml_trust = trustworthiness(data, cuml_embedding, n_neighbors=10)

    assert array_equal(trust, cuml_trust, 1e-1, with_sign=True)


@pytest.mark.parametrize('target_metric', ["categorical", "euclidean"])
@pytest.mark.parametrize('name', [unit_param('digits')])
@pytest.mark.parametrize('nrows', [quality_param(5000),
                         stress_param(500000)])
@pytest.mark.parametrize('n_feats', [quality_param(100),
                         stress_param(1000)])
@pytest.mark.parametrize('should_downcast', [True])
@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
def test_umap_data_formats(input_type, should_downcast,
                           nrows, n_feats, name, target_metric):

    dtype = np.float32 if not should_downcast else np.float64
    n_samples = nrows
    n_feats = n_feats

    if name == 'digits':
        # use the digits dataset for unit test
        digits = datasets.load_digits(n_class=9)
        X = digits["data"].astype(dtype)

    else:
        X, y = datasets.make_blobs(n_samples=n_samples,
                                   n_features=n_feats, random_state=0)

    umap = cuUMAP(n_neighbors=3, n_components=2, target_metric=target_metric)

    embeds = umap.fit_transform(X)
    assert type(embeds) == np.ndarray


@pytest.mark.parametrize('target_metric', ["categorical", "euclidean"])
@pytest.mark.filterwarnings("ignore:(.*)connected(.*):UserWarning:sklearn[.*]")
def test_umap_fit_transform_score_default(target_metric):

    n_samples = 500
    n_features = 20

    data, labels = make_blobs(n_samples=n_samples, n_features=n_features,
                              centers=10, random_state=42)

    model = umap.UMAP(target_metric=target_metric)
    cuml_model = cuUMAP(target_metric=target_metric)

    embedding = model.fit_transform(data)
    cuml_embedding = cuml_model.fit_transform(data, convert_dtype=True)

    cuml_score = adjusted_rand_score(labels,
                                     KMeans(10).fit_predict(
                                        cuml_embedding))
    score = adjusted_rand_score(labels,
                                KMeans(10).fit_predict(embedding))

    assert array_equal(score, cuml_score, 1e-2, with_sign=True)


def test_umap_fit_transform_against_fit_and_transform():

    n_samples = 500
    n_features = 20

    data, labels = make_blobs(n_samples=n_samples, n_features=n_features,
                              centers=10, random_state=42)

    """
    First test the default option does not hash the input
    """

    cuml_model = cuUMAP()

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

    fit_embedding_diff_input = cuml_model.transform(data[1:],
                                                    convert_dtype=True)
    assert joblib.hash(ft_embedding) != joblib.hash(fit_embedding_diff_input)


@pytest.mark.parametrize('n_components,random_state',
                         [unit_param(2, None),
                          unit_param(2, 8),
                          unit_param(2, np.random.RandomState(42)),
                          unit_param(21, None),
                          unit_param(21, np.random.RandomState(42)),
                          unit_param(25, 8),
                          unit_param(50, None),
                          stress_param(50, 8)])
def test_umap_fit_transform_reproducibility(n_components, random_state):

    n_samples = 8000
    n_features = 200

    if random_state is None:
        n_components *= 2

    data, labels = make_blobs(n_samples=n_samples, n_features=n_features,
                              centers=10, random_state=42)

    def get_embedding(n_components, random_state):
        reducer = cuUMAP(init="random",
                         n_components=n_components,
                         random_state=random_state)
        return reducer.fit_transform(data, convert_dtype=True)

    state = copy.copy(random_state)
    cuml_embedding1 = get_embedding(n_components, state)
    state = copy.copy(random_state)
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


@pytest.mark.parametrize('n_components,random_state',
                         [unit_param(2, None),
                          unit_param(2, 8),
                          unit_param(2, np.random.RandomState(42)),
                          unit_param(21, None),
                          unit_param(25, 8),
                          unit_param(25, np.random.RandomState(42)),
                          unit_param(50, None),
                          stress_param(50, 8)])
def test_umap_transform_reproducibility(n_components, random_state):

    n_samples = 5000
    n_features = 200

    if random_state is None:
        n_components *= 2

    data, labels = make_blobs(n_samples=n_samples, n_features=n_features,
                              centers=10, random_state=42)

    selection = np.random.RandomState(42).choice(
        [True, False], n_samples, replace=True, p=[0.5, 0.5])
    fit_data = data[selection]
    transform_data = data[~selection]

    def get_embedding(n_components, random_state):
        reducer = cuUMAP(init="random",
                         n_components=n_components,
                         random_state=random_state)
        reducer.fit(fit_data, convert_dtype=True)
        return reducer.transform(transform_data, convert_dtype=True)

    state = copy.copy(random_state)
    cuml_embedding1 = get_embedding(n_components, state)
    state = copy.copy(random_state)
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
    algo = cuUMAP(n_neighbors=10, min_dist=0.01, init="random",
                  random_state=42)
    embedding = algo.fit_transform(data, convert_dtype=True)
    trust = trustworthiness(iris.data, embedding, n_neighbors=10)
    assert trust >= 0.97


def test_umap_transform_trustworthiness_with_consistency_enabled():
    iris = datasets.load_iris()
    data = iris.data
    selection = np.random.RandomState(42).choice(
        [True, False], data.shape[0], replace=True, p=[0.5, 0.5])
    fit_data = data[selection]
    transform_data = data[~selection]
    model = cuUMAP(n_neighbors=10, min_dist=0.01, init="random",
                   random_state=42)
    model.fit(fit_data, convert_dtype=True)
    embedding = model.transform(transform_data, convert_dtype=True)
    trust = trustworthiness(transform_data, embedding, n_neighbors=10)
    assert trust >= 0.92


@pytest.mark.filterwarnings("ignore:(.*)zero(.*)::scipy[.*]|umap[.*]")
def test_exp_decay_params():
    def compare_exp_decay_params(a=None, b=None, min_dist=0.1, spread=1.0):
        cuml_model = cuUMAP(a=a, b=b, min_dist=min_dist, spread=spread)
        state = cuml_model.__getstate__()
        cuml_a, cuml_b = state['a'], state['b']
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


@pytest.mark.parametrize('n_neighbors', [5, 15])
def test_umap_knn_parameters(n_neighbors):
    data, labels = datasets.make_blobs(
        n_samples=2000, n_features=10, centers=5, random_state=0)
    data = data.astype(np.float32)

    def fit_transform_embed(knn_graph=None):
        model = cuUMAP(random_state=42,
                       init='random',
                       n_neighbors=n_neighbors)
        return model.fit_transform(data, knn_graph=knn_graph,
                                   convert_dtype=True)

    def transform_embed(knn_graph=None):
        model = cuUMAP(random_state=42,
                       init='random',
                       n_neighbors=n_neighbors)
        model.fit(data, knn_graph=knn_graph, convert_dtype=True)
        return model.transform(data, knn_graph=knn_graph,
                               convert_dtype=True)

    def test_trustworthiness(embedding):
        trust = trustworthiness(data, embedding, n_neighbors=n_neighbors)
        assert trust >= 0.92

    def test_equality(e1, e2):
        mean_diff = np.mean(np.abs(e1 - e2))
        print("mean diff: %s" % mean_diff)
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
