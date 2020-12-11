
# Copyright (c) 2019, NVIDIA CORPORATION.
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

from cuml.test.utils import array_equal, unit_param, quality_param, \
    stress_param
from cuml.neighbors import NearestNeighbors as cuKNN

from sklearn.neighbors import NearestNeighbors as skKNN
from sklearn.datasets import make_blobs

from cuml.common import logger

import cupy as cp
import cupyx
import cudf
import pandas as pd
import numpy as np
from scipy.sparse import isspmatrix_csr

import sklearn
import cuml
from cuml.common import has_scipy


def predict(neigh_ind, _y, n_neighbors):
    import scipy.stats as stats

    neigh_ind = neigh_ind.astype(np.int32)

    ypred, count = stats.mode(_y[neigh_ind], axis=1)
    return ypred.ravel(), count.ravel() * 1.0 / n_neighbors


def valid_metrics(algo="brute", cuml_algo=None):
    cuml_algo = algo if cuml_algo is None else cuml_algo
    cuml_metrics = cuml.neighbors.VALID_METRICS[cuml_algo]
    sklearn_metrics = sklearn.neighbors.VALID_METRICS[algo]
    return [value for value in cuml_metrics if value in sklearn_metrics]


@pytest.mark.parametrize("datatype", ["dataframe", "numpy"])
@pytest.mark.parametrize("nrows", [500, 1000, 10000])
@pytest.mark.parametrize("ncols", [100, 1000])
@pytest.mark.parametrize("n_neighbors", [10, 50])
@pytest.mark.parametrize("n_clusters", [2, 10])
def test_neighborhood_predictions(nrows, ncols, n_neighbors, n_clusters,
                                  datatype):
    if not has_scipy():
        pytest.skip('Skipping test_neighborhood_predictions because ' +
                    'Scipy is missing')

    X, y = make_blobs(n_samples=nrows, centers=n_clusters,
                      n_features=ncols, random_state=0)

    X = X.astype(np.float32)

    if datatype == "dataframe":
        X = cudf.DataFrame(X)

    knn_cu = cuKNN()
    knn_cu.fit(X)
    neigh_ind = knn_cu.kneighbors(X, n_neighbors=n_neighbors,
                                  return_distance=False)

    if datatype == "dataframe":
        assert isinstance(neigh_ind, cudf.DataFrame)
        neigh_ind = neigh_ind.as_gpu_matrix().copy_to_host()
    else:
        assert isinstance(neigh_ind, np.ndarray)

    labels, probs = predict(neigh_ind, y, n_neighbors)

    assert array_equal(labels, y)


def test_return_dists():
    n_samples = 50
    n_feats = 50
    k = 5

    X, y = make_blobs(n_samples=n_samples,
                      n_features=n_feats, random_state=0)

    knn_cu = cuKNN()
    knn_cu.fit(X)

    ret = knn_cu.kneighbors(X, k, return_distance=False)
    assert not isinstance(ret, tuple)
    assert ret.shape == (n_samples, k)

    ret = knn_cu.kneighbors(X, k, return_distance=True)
    assert isinstance(ret, tuple)
    assert len(ret) == 2


@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
@pytest.mark.parametrize('nrows', [unit_param(500), quality_param(5000),
                         stress_param(500000)])
@pytest.mark.parametrize('n_feats', [unit_param(3), quality_param(100),
                         stress_param(1000)])
@pytest.mark.parametrize('k', [unit_param(3), quality_param(30),
                         stress_param(50)])
@pytest.mark.parametrize("metric", valid_metrics())
def test_knn_separate_index_search(input_type, nrows, n_feats, k, metric):
    X, _ = make_blobs(n_samples=nrows,
                      n_features=n_feats, random_state=0)

    X_index = X[:100]
    X_search = X[101:]

    p = 5  # Testing 5-norm of the minkowski metric only
    knn_sk = skKNN(metric=metric, p=p)  # Testing
    knn_sk.fit(X_index)
    D_sk, I_sk = knn_sk.kneighbors(X_search, k)

    X_orig = X_index

    if input_type == "dataframe":
        X_index = cudf.DataFrame(X_index)
        X_search = cudf.DataFrame(X_search)

    knn_cu = cuKNN(metric=metric, p=p)
    knn_cu.fit(X_index)
    D_cuml, I_cuml = knn_cu.kneighbors(X_search, k)

    if input_type == "dataframe":
        assert isinstance(D_cuml, cudf.DataFrame)
        assert isinstance(I_cuml, cudf.DataFrame)
        D_cuml_arr = D_cuml.as_gpu_matrix().copy_to_host()
        I_cuml_arr = I_cuml.as_gpu_matrix().copy_to_host()
    else:
        assert isinstance(D_cuml, np.ndarray)
        assert isinstance(I_cuml, np.ndarray)
        D_cuml_arr = D_cuml
        I_cuml_arr = I_cuml

    with cuml.using_output_type("numpy"):
        # Assert the cuml model was properly reverted
        np.testing.assert_allclose(knn_cu.X_m, X_orig,
                                   atol=1e-3, rtol=1e-3)

    if metric == 'braycurtis':
        diff = D_cuml_arr - D_sk
        # Braycurtis has a few differences, but this is computed by FAISS.
        # So long as the indices all match below, the small discrepancy
        # should be okay.
        assert len(diff[diff > 1e-2]) / X_search.shape[0] < 0.06
    else:
        np.testing.assert_allclose(D_cuml_arr, D_sk, atol=1e-3,
                                   rtol=1e-3)
    assert I_cuml_arr.all() == I_sk.all()


@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
@pytest.mark.parametrize('nrows', [unit_param(500), quality_param(5000),
                         stress_param(500000)])
@pytest.mark.parametrize('n_feats', [unit_param(3), quality_param(100),
                         stress_param(1000)])
@pytest.mark.parametrize('k', [unit_param(3), quality_param(30),
                         stress_param(50)])
@pytest.mark.parametrize("metric", valid_metrics())
def test_knn_x_none(input_type, nrows, n_feats, k, metric):
    X, _ = make_blobs(n_samples=nrows,
                      n_features=n_feats, random_state=0)

    p = 5  # Testing 5-norm of the minkowski metric only
    knn_sk = skKNN(metric=metric, p=p)  # Testing
    knn_sk.fit(X)
    D_sk, I_sk = knn_sk.kneighbors(X=None, n_neighbors=k)

    X_orig = X

    if input_type == "dataframe":
        X = cudf.DataFrame(X)

    knn_cu = cuKNN(metric=metric, p=p, output_type="numpy")
    knn_cu.fit(X)
    D_cuml, I_cuml = knn_cu.kneighbors(X=None, n_neighbors=k)

    # Assert the cuml model was properly reverted
    cp.testing.assert_allclose(knn_cu.X_m, X_orig,
                               atol=1e-5, rtol=1e-4)

    # Allow a max relative diff of 10% and absolute diff of 1%
    cp.testing.assert_allclose(D_cuml, D_sk, atol=5e-2,
                               rtol=1e-1)
    assert I_cuml.all() == I_sk.all()


def test_knn_fit_twice():
    """
    Test that fitting a model twice does not fail.
    This is necessary since the NearestNeighbors class
    needs to free Cython allocated heap memory when
    fit() is called more than once.
    """

    n_samples = 1000
    n_feats = 50
    k = 5

    X, y = make_blobs(n_samples=n_samples,
                      n_features=n_feats, random_state=0)

    knn_cu = cuKNN()
    knn_cu.fit(X)
    knn_cu.fit(X)

    knn_cu.kneighbors(X, k)

    del knn_cu


@pytest.mark.parametrize('input_type', ['ndarray'])
@pytest.mark.parametrize('nrows', [unit_param(500), quality_param(5000),
                         stress_param(500000)])
@pytest.mark.parametrize('n_feats', [unit_param(20), quality_param(100),
                         stress_param(1000)])
def test_nn_downcast_fails(input_type, nrows, n_feats):
    X, y = make_blobs(n_samples=nrows,
                      n_features=n_feats, random_state=0)

    knn_cu = cuKNN()
    if input_type == 'dataframe':
        X_pd = pd.DataFrame({'fea%d' % i: X[0:, i] for i in range(X.shape[1])})
        X_cudf = cudf.DataFrame.from_pandas(X_pd)
        knn_cu.fit(X_cudf, convert_dtype=True)

    with pytest.raises(Exception):
        knn_cu.fit(X, convert_dtype=False)

    # Test fit() fails when downcast corrupted data
    X = np.array([[np.finfo(np.float32).max]], dtype=np.float64)
    knn_cu = cuKNN()
    with pytest.raises(Exception):
        knn_cu.fit(X, convert_dtype=False)


@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
@pytest.mark.parametrize('nrows', [unit_param(10), quality_param(100),
                         stress_param(1000)])
@pytest.mark.parametrize('n_feats', [unit_param(5), quality_param(30),
                         stress_param(100)])
@pytest.mark.parametrize("p", [2, 5])
@pytest.mark.parametrize('k', [unit_param(3), quality_param(10),
                         stress_param(30)])
@pytest.mark.parametrize("metric", valid_metrics())
@pytest.mark.parametrize("mode", ['connectivity', 'distance'])
@pytest.mark.parametrize("output_type", ['cupy', 'numpy'])
@pytest.mark.parametrize("as_instance", [True, False])
def test_knn_graph(input_type, nrows, n_feats, p, k, metric, mode,
                   output_type, as_instance):
    X, _ = make_blobs(n_samples=nrows,
                      n_features=n_feats, random_state=0)

    if as_instance:
        sparse_sk = sklearn.neighbors.kneighbors_graph(X, k, mode,
                                                       metric=metric, p=p,
                                                       include_self='auto')
    else:
        knn_sk = skKNN(metric=metric, p=p)
        knn_sk.fit(X)
        sparse_sk = knn_sk.kneighbors_graph(X, k, mode)

    if input_type == "dataframe":
        X = cudf.DataFrame(X)

    if as_instance:
        sparse_cu = cuml.neighbors.kneighbors_graph(X, k, mode,
                                                    metric=metric, p=p,
                                                    include_self='auto',
                                                    output_type=output_type)
    else:
        knn_cu = cuKNN(metric=metric, p=p, output_type=output_type)
        knn_cu.fit(X)
        sparse_cu = knn_cu.kneighbors_graph(X, k, mode)

    assert np.array_equal(sparse_sk.data.shape, sparse_cu.data.shape)
    assert np.array_equal(sparse_sk.indices.shape, sparse_cu.indices.shape)
    assert np.array_equal(sparse_sk.indptr.shape, sparse_cu.indptr.shape)
    assert np.array_equal(sparse_sk.toarray().shape, sparse_cu.toarray().shape)

    if output_type == 'cupy':
        assert cupyx.scipy.sparse.isspmatrix_csr(sparse_cu)
    else:
        assert isspmatrix_csr(sparse_cu)


@pytest.mark.parametrize("metric", valid_metrics(cuml_algo="sparse"))
@pytest.mark.parametrize('nrows', [1, 10, 35])
@pytest.mark.parametrize('ncols', [10, 35])
@pytest.mark.parametrize('density', [0.8])
@pytest.mark.parametrize('n_neighbors', [1, 4])
@pytest.mark.parametrize('batch_size_index', [10, 20000])
@pytest.mark.parametrize('batch_size_query', [10, 20000])
def test_nearest_neighbors_sparse(nrows, ncols,
                                  density,
                                  metric,
                                  n_neighbors,
                                  batch_size_index,
                                  batch_size_query):

    if nrows == 1 and n_neighbors > 1:
        return

    a = cp.sparse.random(nrows, ncols, format='csr', density=density,
                         random_state=32)

    logger.set_level(logger.level_info)
    nn = cuKNN(metric=metric, n_neighbors=n_neighbors, algorithm="brute",
               verbose=logger.level_debug,
               algo_params={"batch_size_index": batch_size_index,
                            "batch_size_query": batch_size_query})
    nn.fit(a)

    cuD, cuI = nn.kneighbors(a)

    sknn = skKNN(metric=metric, n_neighbors=n_neighbors,
                 algorithm="brute", n_jobs=-1)
    sk_X = a.get()
    sknn.fit(sk_X)

    skD, skI = sknn.kneighbors(sk_X)

    cp.testing.assert_allclose(cuI, skI, atol=1e-4, rtol=1e-4)
    cp.testing.assert_allclose(cuD, skD, atol=1e-3, rtol=1e-3)
