# Copyright (c) 2021, NVIDIA CORPORATION.
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
from sklearn.manifold.t_sne import trustworthiness
from sklearn import datasets

import cuml.common.logger as logger


dataset_names = ['digits', 'boston', 'iris', 'breast_cancer',
                 'diabetes']


def check_embedding(X, Y, score=0.76):
    """Compares TSNE embedding trustworthiness, NAN and verbosity"""
    nans = np.sum(np.isnan(Y))
    trust = trustworthiness(X, Y)
    print("Trust = ", trust)
    assert trust > score
    assert nans == 0


@pytest.mark.parametrize('name', dataset_names)
@pytest.mark.parametrize('type_knn_graph', ['sklearn', 'cuml'])
@pytest.mark.parametrize('method', ['barnes_hut', 'fft'])
def test_tsne_knn_parameters(name, type_knn_graph, method):

    datasets
    X = eval("datasets.load_{}".format(name))().data

    neigh = skKNN(n_neighbors=90) if type_knn_graph == 'sklearn' \
        else cuKNN(n_neighbors=90)

    neigh.fit(X)
    knn_graph = neigh.kneighbors_graph(X, mode="distance").astype('float32')

    for i in range(3):
        print("iteration = ", i)
        tsne = TSNE(method=method)
        print("Direct knn graph input")
        Y = tsne.fit_transform(X, True, knn_graph)
        check_embedding(X, Y)

        print("KNN graph COO")
        Y = tsne.fit_transform(X, True, knn_graph.tocoo())
        check_embedding(X, Y)

        print("KNN graph CSC")
        Y = tsne.fit_transform(X, True, knn_graph.tocsc())
        check_embedding(X, Y)
        del Y


@pytest.mark.parametrize('name', dataset_names)
@pytest.mark.parametrize('type_knn_graph', ['sklearn', 'cuml'])
@pytest.mark.parametrize('method', ['fft'])
def test_tsne_knn_graph_used(name, type_knn_graph, method):

    datasets
    X = eval("datasets.load_{}".format(name))().data

    neigh = skKNN(n_neighbors=90) if type_knn_graph == 'sklearn' \
        else cuKNN(n_neighbors=90)

    neigh.fit(X)
    knn_graph = neigh.kneighbors_graph(X, mode="distance")
    tsne = TSNE(method=method)

    # Perform tsne with normal knn_graph
    Y = tsne.fit_transform(X, True, knn_graph)
    trust_normal = trustworthiness(X, Y)
    print("Trust = ", trust_normal)

    X_garbage = np.ones(X.shape)
    knn_graph_garbage = neigh.kneighbors_graph(X_garbage, mode="distance")

    # Perform tsne with garbage knn_graph
    Y = tsne.fit_transform(X, True, knn_graph_garbage)
    trust_garbage = trustworthiness(X, Y)
    print("Trust = ", trust_garbage)
    assert (trust_normal - trust_garbage) > 0.15

    Y = tsne.fit_transform(X, True, knn_graph_garbage.tocoo())
    trust_garbage = trustworthiness(X, Y)
    print("Trust = ", trust_garbage)
    assert (trust_normal - trust_garbage) > 0.15

    Y = tsne.fit_transform(X, True, knn_graph_garbage.tocsc())
    trust_garbage = trustworthiness(X, Y)
    print("Trust = ", trust_garbage)
    assert (trust_normal - trust_garbage) > 0.15




@pytest.mark.parametrize('name', dataset_names)
@pytest.mark.parametrize('method', ['fft'])
def test_tsne(name, method):
    """
    This tests how TSNE handles a lot of input data across time.
    (1) Numpy arrays are passed in
    (2) Params are changed in the TSNE class
    (3) The class gets re-used across time
    (4) Trustworthiness is checked
    (5) Tests NAN in TSNE output for learning rate explosions
    (6) Tests verbosity
    """
    datasets
    X = eval("datasets.load_{}".format(name))().data

    for i in range(3):
        print("iteration = ", i)

        tsne = TSNE(2, random_state=i, verbose=False,
                    learning_rate=2+i)

        # Reuse
        Y = tsne.fit_transform(X)
        check_embedding(X, Y)
        del Y

        # Again
        tsne = TSNE(2, random_state=i+2, verbose=logger.level_debug,
                    learning_rate=2+i+2, method=method)

        # Reuse
        Y = tsne.fit_transform(X)
        check_embedding(X, Y)
        del Y


@pytest.mark.parametrize('name', dataset_names)
@pytest.mark.parametrize('method', ['fft'])
def test_tsne_default(name, method):

    datasets
    X = eval("datasets.load_{}".format(name))().data

    for i in range(3):
        print("iteration = ", i)

        tsne = TSNE(method=method)
        Y = tsne.fit_transform(X)
        check_embedding(X, Y)
        del Y


@pytest.mark.parametrize('nrows', [stress_param(2400000)])
@pytest.mark.parametrize('ncols', [stress_param(250)])
@pytest.mark.parametrize('method', ['fft'])
def test_tsne_large(nrows, ncols, method):
    """
    This tests how TSNE handles large input
    """
    X, y = make_blobs(n_samples=nrows, centers=8,
                      n_features=ncols, random_state=0)

    X = X.astype(np.float32)

    tsne = TSNE(random_state=0, exaggeration_iter=1, n_iter=2, method=method)
    Y = tsne.fit_transform(X)
    nans = np.sum(np.isnan(Y))
    assert nans == 0



def test_components_exception():
    with pytest.raises(ValueError):
        TSNE(n_components=3)


@pytest.mark.parametrize('input_type', ['cupy', 'scipy'])
@pytest.mark.parametrize('method', ['fft'])
def test_tsne_transform_on_digits_sparse(input_type, method):

    datasets
    digits = datasets.load_digits()

    digits_selection = np.random.RandomState(42).choice(
        [True, False], 1797, replace=True, p=[0.60, 0.40])

    if input_type == 'cupy':
        sp_prefix = cupyx.scipy.sparse
    else:
        sp_prefix = scipy.sparse

    fitter = TSNE(2, n_neighbors=15,
                  random_state=1,
                  learning_rate=500,
                  angle=0.8, method=method)

    new_data = sp_prefix.csr_matrix(
        scipy.sparse.csr_matrix(digits.data[~digits_selection]))

    embedding = fitter.fit_transform(new_data, convert_dtype=True)

    if input_type == 'cupy':
        embedding = embedding.get()

    trust = trustworthiness(digits.data[~digits_selection], embedding, 15)
    assert trust >= 0.85


@pytest.mark.parametrize('type_knn_graph', ['sklearn', 'cuml'])
@pytest.mark.parametrize('input_type', ['cupy', 'scipy'])
@pytest.mark.parametrize('method', ['fft'])
def test_tsne_knn_parameters_sparse(type_knn_graph, input_type, method):

    datasets
    digits = datasets.load_digits()

    neigh = skKNN(n_neighbors=90) if type_knn_graph == 'sklearn' \
        else cuKNN(n_neighbors=90)

    digits_selection = np.random.RandomState(42).choice(
        [True, False], 1797, replace=True, p=[0.60, 0.40])

    selected_digits = digits.data[~digits_selection]

    neigh.fit(selected_digits)
    knn_graph = neigh.kneighbors_graph(selected_digits, mode="distance")

    if input_type == 'cupy':
        sp_prefix = cupyx.scipy.sparse
    else:
        sp_prefix = scipy.sparse

    tsne = TSNE(2, n_neighbors=15,
                random_state=1,
                learning_rate=500,
                angle=0.8, method=method)

    new_data = sp_prefix.csr_matrix(
        scipy.sparse.csr_matrix(selected_digits))

    Y = tsne.fit_transform(new_data, True, knn_graph)
    if input_type == 'cupy':
        Y = Y.get()
    check_embedding(selected_digits, Y, 0.85)

    Y = tsne.fit_transform(new_data, True, knn_graph.tocoo())
    if input_type == 'cupy':
        Y = Y.get()
    check_embedding(selected_digits, Y, 0.85)

    Y = tsne.fit_transform(new_data, True, knn_graph.tocsc())
    if input_type == 'cupy':
        Y = Y.get()
    check_embedding(selected_digits, Y, 0.85)
    del Y
