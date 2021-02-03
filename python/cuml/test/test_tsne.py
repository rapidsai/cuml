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

import numpy as np
import pytest
import scipy
import cupyx

from cuml.manifold import TSNE
from cuml.test.utils import stress_param
from cuml.neighbors import NearestNeighbors as cuKNN

from sklearn.neighbors import NearestNeighbors as skKNN
from sklearn.datasets import make_blobs
from cuml.metrics import trustworthiness
from sklearn import datasets


DEFAULT_N_NEIGHBORS = 25

test_datasets = {"digits": datasets.load_digits(),
                 "boston": datasets.load_boston(),
                 "cancer": datasets.load_breast_cancer(),
                 "diabetes": datasets.load_diabetes()}


def validate_embedding(X, Y, score=0.76, n_neighbors=DEFAULT_N_NEIGHBORS):
    """Compares TSNE embedding trustworthiness, NAN and verbosity"""
    nans = np.sum(np.isnan(Y))
    trust = trustworthiness(X, Y, n_neighbors=n_neighbors)
    assert trust > score
    assert nans == 0


@pytest.mark.parametrize('dataset', test_datasets.values())
@pytest.mark.parametrize('type_knn_graph', ['cuml', 'sklearn'])
@pytest.mark.parametrize('method', ['fft', 'barnes_hut'])
def test_tsne_knn_graph_used(dataset, type_knn_graph, method):

    X = dataset.data

    neigh = cuKNN(n_neighbors=DEFAULT_N_NEIGHBORS).fit(X)
    knn_graph = neigh.kneighbors_graph(X, mode="distance").astype('float32')

    if type_knn_graph == 'cuml':
        knn_graph = cupyx.scipy.sparse.csr_matrix(knn_graph)

    tsne = TSNE(random_state=1,
                n_neighbors=DEFAULT_N_NEIGHBORS,
                method=method,
                learning_rate_method='none')

    # Perform tsne with normal knn_graph
    Y = tsne.fit_transform(X, True, knn_graph)
    trust_normal = trustworthiness(X, Y, n_neighbors=DEFAULT_N_NEIGHBORS)

    X_garbage = np.ones(X.shape)
    knn_graph_garbage = neigh.kneighbors_graph(
        X_garbage, mode="distance").astype('float32')

    if type_knn_graph == 'cuml':
        knn_graph_garbage = cupyx.scipy.sparse.csr_matrix(knn_graph_garbage)

    # Perform tsne with garbage knn_graph
    Y1 = tsne.fit_transform(X, True, knn_graph_garbage)
    trust_garbage = trustworthiness(X, Y1, n_neighbors=DEFAULT_N_NEIGHBORS)
    assert (trust_normal - trust_garbage) > 0.15

    Y2 = tsne.fit_transform(X, True, knn_graph_garbage.tocoo())
    trust_garbage = trustworthiness(X, Y2, n_neighbors=DEFAULT_N_NEIGHBORS)
    assert (trust_normal - trust_garbage) > 0.15

    Y3 = tsne.fit_transform(X, True, knn_graph_garbage.tocsc())
    trust_garbage = trustworthiness(X, Y3, n_neighbors=DEFAULT_N_NEIGHBORS)
    assert (trust_normal - trust_garbage) > 0.15


@pytest.mark.parametrize('dataset', test_datasets.values())
@pytest.mark.parametrize('type_knn_graph', ['cuml', 'sklearn'])
@pytest.mark.parametrize('method', ['fft', 'barnes_hut'])
def test_tsne_knn_parameters(dataset, type_knn_graph, method):

    X = dataset.data

    neigh = cuKNN(n_neighbors=DEFAULT_N_NEIGHBORS).fit(X)
    knn_graph = neigh.kneighbors_graph(X, mode="distance").astype('float32')

    if type_knn_graph == 'cuml':
        knn_graph = cupyx.scipy.sparse.csr_matrix(knn_graph)

    for i in range(3):
        tsne = TSNE(random_state=1,
                    n_neighbors=DEFAULT_N_NEIGHBORS,
                    method=method)
        embed1 = tsne.fit_transform(X, True, knn_graph)
        validate_embedding(X, embed1)

        embed2 = tsne.fit_transform(X, True, knn_graph.tocoo())
        validate_embedding(X, embed2)

        embed3 = tsne.fit_transform(X, True, knn_graph.tocsc())
        validate_embedding(X, embed3)


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

    for i in range(3):
        tsne = TSNE(n_components=2,
                    random_state=1,
                    n_neighbors=DEFAULT_N_NEIGHBORS,
                    method=method)

        Y = tsne.fit_transform(X)
        validate_embedding(X, Y)

        # Again
        tsne = TSNE(n_components=2,
                    random_state=1,
                    n_neighbors=DEFAULT_N_NEIGHBORS,
                    method=method)

        Y = tsne.fit_transform(X)
        validate_embedding(X, Y)


@pytest.mark.parametrize('dataset', test_datasets.values())
@pytest.mark.parametrize('method', ['fft', 'barnes_hut'])
def test_tsne_default(dataset, method):

    X = dataset.data

    for i in range(3):
        tsne = TSNE(random_state=1,
                    n_neighbors=DEFAULT_N_NEIGHBORS,
                    method=method)
        Y = tsne.fit_transform(X)
        validate_embedding(X, Y)
        del Y


@pytest.mark.parametrize('nrows', [stress_param(2400000)])
@pytest.mark.parametrize('ncols', [stress_param(250)])
@pytest.mark.parametrize('method', ['fft', 'barnes_hut'])
def test_tsne_large(nrows, ncols, method):
    """
    This tests how TSNE handles large input
    """
    X, y = make_blobs(n_samples=nrows,
                      centers=8,
                      n_features=ncols,
                      random_state=1)

    X = X.astype(np.float32)

    tsne = TSNE(random_state=1,
                exaggeration_iter=1,
                n_neighbors=DEFAULT_N_NEIGHBORS,
                n_iter=2, method=method)
    Y = tsne.fit_transform(X)
    nans = np.sum(np.isnan(Y))
    assert nans == 0


def test_components_exception():
    with pytest.raises(ValueError):
        TSNE(n_components=3)


@pytest.mark.parametrize('input_type', ['cupy', 'scipy'])
@pytest.mark.parametrize('method', ['fft', 'barnes_hut'])
def test_tsne_fit_transform_on_digits_sparse(input_type, method):

    digits = datasets.load_digits()

    if input_type == 'cupy':
        sp_prefix = cupyx.scipy.sparse
    else:
        sp_prefix = scipy.sparse

    fitter = TSNE(n_components=2,
                  n_neighbors=DEFAULT_N_NEIGHBORS,
                  random_state=1,
                  method=method)

    new_data = sp_prefix.csr_matrix(
        scipy.sparse.csr_matrix(digits.data)).astype('float32')

    embedding = fitter.fit_transform(new_data, convert_dtype=True)

    if input_type == 'cupy':
        embedding = embedding.get()

    trust = trustworthiness(digits, embedding,
                            n_neighbors=90)
    assert trust >= 0.85


@pytest.mark.parametrize('type_knn_graph', ['cuml', 'sklean'])
@pytest.mark.parametrize('input_type', ['cupy', 'scipy'])
@pytest.mark.parametrize('method', ['fft', 'barnes_hut'])
def test_tsne_knn_parameters_sparse(type_knn_graph, input_type, method):

    digits = test_datasets["digits"].data

    neigh = cuKNN(n_neighbors=DEFAULT_N_NEIGHBORS).fit(digits)
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
                method=method)

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
    del Y
