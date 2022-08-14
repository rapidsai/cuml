# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

import numpy as np
import pytest
import scipy
import cupyx

from cuml.manifold import TSNE
from cuml.testing.utils import array_equal, stress_param
from cuml.neighbors import NearestNeighbors as cuKNN

from sklearn.datasets import make_blobs
from sklearn.manifold import trustworthiness
from sklearn import datasets
from sklearn.manifold import TSNE as skTSNE

pytestmark = pytest.mark.filterwarnings("ignore:Method 'fft' is "
                                        "experimental::")

DEFAULT_N_NEIGHBORS = 90
DEFAULT_PERPLEXITY = 30

test_datasets = {
                 "digits": datasets.load_digits(),
                 "boston": datasets.load_boston(),
                 "diabetes": datasets.load_diabetes(),
                 "cancer": datasets.load_breast_cancer(),
}


def validate_embedding(X, Y, score=0.74, n_neighbors=DEFAULT_N_NEIGHBORS):
    """Compares TSNE embedding trustworthiness, NAN and verbosity"""
    nans = np.sum(np.isnan(Y))
    trust = trustworthiness(X, Y, n_neighbors=n_neighbors)

    print("Trust=%s" % trust)
    assert trust > score
    assert nans == 0


@pytest.mark.parametrize('dataset', test_datasets.values())
@pytest.mark.parametrize('type_knn_graph', ['cuml', 'sklearn'])
@pytest.mark.parametrize('method', ['fft', 'barnes_hut'])
def test_tsne_knn_graph_used(dataset, type_knn_graph, method):

    X = dataset.data

    neigh = cuKNN(n_neighbors=DEFAULT_N_NEIGHBORS,
                  metric="euclidean").fit(X)
    knn_graph = neigh.kneighbors_graph(X, mode="distance").astype('float32')

    if type_knn_graph == 'cuml':
        knn_graph = cupyx.scipy.sparse.csr_matrix(knn_graph)

    tsne = TSNE(random_state=1,
                n_neighbors=DEFAULT_N_NEIGHBORS,
                method=method,
                perplexity=DEFAULT_PERPLEXITY,
                learning_rate_method='none',
                min_grad_norm=1e-12)

    # Perform tsne with normal knn_graph
    Y = tsne.fit_transform(X, True, knn_graph)

    trust_normal = trustworthiness(X, Y, n_neighbors=DEFAULT_N_NEIGHBORS)

    X_garbage = np.ones(X.shape)
    knn_graph_garbage = neigh.kneighbors_graph(
        X_garbage, mode="distance").astype('float32')

    if type_knn_graph == 'cuml':
        knn_graph_garbage = cupyx.scipy.sparse.csr_matrix(knn_graph_garbage)

    tsne = TSNE(random_state=1,
                n_neighbors=DEFAULT_N_NEIGHBORS,
                method=method,
                perplexity=DEFAULT_PERPLEXITY,
                learning_rate_method='none',
                min_grad_norm=1e-12)

    # Perform tsne with garbage knn_graph
    Y = tsne.fit_transform(X, True, knn_graph_garbage)

    trust_garbage = trustworthiness(X, Y, n_neighbors=DEFAULT_N_NEIGHBORS)
    assert (trust_normal - trust_garbage) > 0.15

    Y = tsne.fit_transform(X, True, knn_graph_garbage)
    trust_garbage = trustworthiness(X, Y, n_neighbors=DEFAULT_N_NEIGHBORS)
    assert (trust_normal - trust_garbage) > 0.15

    Y = tsne.fit_transform(X, True, knn_graph_garbage)
    trust_garbage = trustworthiness(X, Y, n_neighbors=DEFAULT_N_NEIGHBORS)
    assert (trust_normal - trust_garbage) > 0.15


@pytest.mark.parametrize('dataset', test_datasets.values())
@pytest.mark.parametrize('type_knn_graph', ['cuml', 'sklearn'])
@pytest.mark.parametrize('method', ['fft', 'barnes_hut'])
def test_tsne_knn_parameters(dataset, type_knn_graph, method):

    X = dataset.data

    from sklearn.preprocessing import normalize

    X = normalize(X, norm='l1')

    neigh = cuKNN(n_neighbors=DEFAULT_N_NEIGHBORS,
                  metric="euclidean").fit(X)
    knn_graph = neigh.kneighbors_graph(X, mode="distance").astype('float32')

    if type_knn_graph == 'cuml':
        knn_graph = cupyx.scipy.sparse.csr_matrix(knn_graph)

    tsne = TSNE(n_components=2,
                random_state=1,
                n_neighbors=DEFAULT_N_NEIGHBORS,
                learning_rate_method='none',
                method=method,
                min_grad_norm=1e-12,
                perplexity=DEFAULT_PERPLEXITY)

    embed = tsne.fit_transform(X, True, knn_graph)
    validate_embedding(X, embed)

    embed = tsne.fit_transform(X, True, knn_graph.tocoo())
    validate_embedding(X, embed)

    embed = tsne.fit_transform(X, True, knn_graph.tocsc())
    validate_embedding(X, embed)


@pytest.mark.parametrize('dataset', test_datasets.values())
@pytest.mark.parametrize('method', ['fft', 'barnes_hut'])
def test_tsne(dataset, method):
    """
    This tests how TSNE handles a lot of input data across time.
    (1) Numpy arrays are passed in
    (2) Params are changed in the TSNE class
    (3) The class gets re-used across time
    (4) Trustworthiness is checked
    (5) Tests NAN in TSNE output for learning rate explosions
    (6) Tests verbosity
    """
    X = dataset.data

    tsne = TSNE(n_components=2,
                random_state=1,
                n_neighbors=DEFAULT_N_NEIGHBORS,
                learning_rate_method='none',
                method=method,
                min_grad_norm=1e-12,
                perplexity=DEFAULT_PERPLEXITY)

    Y = tsne.fit_transform(X)
    validate_embedding(X, Y)


@pytest.mark.parametrize('nrows', [stress_param(2400000)])
@pytest.mark.parametrize('ncols', [stress_param(225)])
@pytest.mark.parametrize('method', ['fft', 'barnes_hut'])
def test_tsne_large(nrows, ncols, method):
    """
    This tests how TSNE handles large input
    """
    X, y = make_blobs(n_samples=nrows,
                      centers=8,
                      n_features=ncols,
                      random_state=1).astype(np.float32)

    tsne = TSNE(random_state=1,
                exaggeration_iter=1,
                n_iter=2, method=method,
                min_grad_norm=1e-12)
    Y = tsne.fit_transform(X)
    nans = np.sum(np.isnan(Y))
    assert nans == 0


def test_components_exception():
    with pytest.raises(ValueError):
        TSNE(n_components=3)


@pytest.mark.parametrize('input_type', ['cupy', 'scipy'])
@pytest.mark.parametrize('method', ['fft', 'barnes_hut'])
def test_tsne_fit_transform_on_digits_sparse(input_type, method):

    digits = test_datasets['digits'].data

    if input_type == 'cupy':
        sp_prefix = cupyx.scipy.sparse
    else:
        sp_prefix = scipy.sparse

    fitter = TSNE(n_components=2,
                  random_state=1,
                  method=method,
                  min_grad_norm=1e-12,
                  n_neighbors=DEFAULT_N_NEIGHBORS,
                  learning_rate_method="none",
                  perplexity=DEFAULT_PERPLEXITY)

    new_data = sp_prefix.csr_matrix(
        scipy.sparse.csr_matrix(digits)).astype('float32')

    embedding = fitter.fit_transform(new_data, convert_dtype=True)

    if input_type == 'cupy':
        embedding = embedding.get()

    trust = trustworthiness(digits, embedding,
                            n_neighbors=DEFAULT_N_NEIGHBORS)
    assert trust >= 0.85


@pytest.mark.parametrize('type_knn_graph', ['cuml', 'sklearn'])
@pytest.mark.parametrize('input_type', ['cupy', 'scipy'])
@pytest.mark.parametrize('method', ['fft', 'barnes_hut'])
def test_tsne_knn_parameters_sparse(type_knn_graph, input_type, method):

    digits = test_datasets["digits"].data

    neigh = cuKNN(n_neighbors=DEFAULT_N_NEIGHBORS,
                  metric="euclidean").fit(digits)
    knn_graph = neigh.kneighbors_graph(
        digits, mode="distance").astype('float32')

    if type_knn_graph == 'cuml':
        knn_graph = cupyx.scipy.sparse.csr_matrix(knn_graph)

    if input_type == 'cupy':
        sp_prefix = cupyx.scipy.sparse
    else:
        sp_prefix = scipy.sparse

    tsne = TSNE(n_components=2,
                n_neighbors=DEFAULT_N_NEIGHBORS,
                random_state=1,
                learning_rate_method='none',
                method=method,
                min_grad_norm=1e-12,
                perplexity=DEFAULT_PERPLEXITY)

    new_data = sp_prefix.csr_matrix(
        scipy.sparse.csr_matrix(digits))

    Y = tsne.fit_transform(new_data, True, knn_graph)
    if input_type == 'cupy':
        Y = Y.get()
    validate_embedding(digits, Y, 0.85)

    Y = tsne.fit_transform(new_data, True, knn_graph.tocoo())
    if input_type == 'cupy':
        Y = Y.get()
    validate_embedding(digits, Y, 0.85)

    Y = tsne.fit_transform(new_data, True, knn_graph.tocsc())
    if input_type == 'cupy':
        Y = Y.get()
    validate_embedding(digits, Y, 0.85)


@pytest.mark.parametrize('metric', ['l2', 'euclidean', 'sqeuclidean',
                                    'cityblock', 'l1', 'manhattan',
                                    'minkowski', 'chebyshev',
                                    'cosine', 'correlation'])
def test_tsne_distance_metrics(metric):

    data, labels = make_blobs(n_samples=1000, n_features=64,
                              centers=5, random_state=42)

    tsne = TSNE(n_components=2,
                random_state=1,
                n_neighbors=DEFAULT_N_NEIGHBORS,
                method='exact',
                learning_rate_method='none',
                min_grad_norm=1e-12,
                perplexity=DEFAULT_PERPLEXITY,
                metric=metric)

    sk_tsne = skTSNE(n_components=2,
                     random_state=1,
                     min_grad_norm=1e-12,
                     method='exact',
                     perplexity=DEFAULT_PERPLEXITY,
                     metric=metric)

    cuml_embedding = tsne.fit_transform(data)
    sk_embedding = sk_tsne.fit_transform(data)
    nans = np.sum(np.isnan(cuml_embedding))
    cuml_trust = trustworthiness(data, cuml_embedding, metric=metric)
    sk_trust = trustworthiness(data, sk_embedding, metric=metric)

    assert cuml_trust > 0.85
    assert nans == 0
    assert array_equal(sk_trust, cuml_trust, 0.05, with_sign=True)


@pytest.mark.parametrize('method', ['fft', 'barnes_hut', 'exact'])
@pytest.mark.parametrize('metric', ['l2', 'euclidean', 'cityblock',
                                    'l1', 'manhattan', 'cosine'])
def test_tsne_distance_metrics_on_sparse_input(method, metric):

    data, labels = make_blobs(n_samples=1000, n_features=64,
                              centers=5, random_state=42)
    data_sparse = scipy.sparse.csr_matrix(data)

    cuml_tsne = TSNE(n_components=2,
                     random_state=1,
                     n_neighbors=DEFAULT_N_NEIGHBORS,
                     method=method,
                     learning_rate_method='none',
                     min_grad_norm=1e-12,
                     perplexity=DEFAULT_PERPLEXITY,
                     metric=metric)

    if method == 'fft':
        sk_tsne = skTSNE(n_components=2,
                         random_state=1,
                         min_grad_norm=1e-12,
                         method='barnes_hut',
                         perplexity=DEFAULT_PERPLEXITY,
                         metric=metric)

    else:
        sk_tsne = skTSNE(n_components=2,
                         random_state=1,
                         min_grad_norm=1e-12,
                         method=method,
                         perplexity=DEFAULT_PERPLEXITY,
                         metric=metric)

    cuml_embedding = cuml_tsne.fit_transform(data_sparse)
    nans = np.sum(np.isnan(cuml_embedding))
    sk_embedding = sk_tsne.fit_transform(data_sparse)
    cu_trust = trustworthiness(data, cuml_embedding, metric=metric)
    sk_trust = trustworthiness(data, sk_embedding, metric=metric)

    assert cu_trust > 0.85
    assert nans == 0
    assert array_equal(sk_trust, cu_trust, 0.05, with_sign=True)
