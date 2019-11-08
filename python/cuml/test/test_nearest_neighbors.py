
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
from sklearn.datasets.samples_generator import make_blobs
import cudf
import pandas as pd
import numpy as np
from cuml.test.utils import array_equal

import scipy.stats as stats


def predict(neigh_ind, _y, n_neighbors):

    neigh_ind = neigh_ind.astype(np.int32)

    ypred, count = stats.mode(_y[neigh_ind], axis=1)
    return ypred.ravel(), count.ravel() * 1.0 / n_neighbors


@pytest.mark.parametrize("nrows", [500, 1000, 10000])
@pytest.mark.parametrize("ncols", [100, 1000])
@pytest.mark.parametrize("n_neighbors", [10, 50])
@pytest.mark.parametrize("n_clusters", [2, 10])
def test_neighborhood_predictions(nrows, ncols, n_neighbors, n_clusters):

    X, y = make_blobs(n_samples=nrows, centers=n_clusters,
                      n_features=ncols, random_state=0)

    X = X.astype(np.float32)

    knn_cu = cuKNN()
    knn_cu.fit(X)
    neigh_ind = knn_cu.kneighbors(X, n_neighbors=n_neighbors,
                                  return_distance=False)

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


@pytest.mark.parametrize('input_type', ['ndarray'])
@pytest.mark.parametrize('nrows', [unit_param(20), quality_param(5000),
                         stress_param(500000)])
@pytest.mark.parametrize('n_feats', [unit_param(3), quality_param(100),
                         stress_param(1000)])
@pytest.mark.parametrize('k', [unit_param(3), quality_param(30),
                         stress_param(50)])
def test_cuml_against_sklearn(input_type, nrows, n_feats, k):
    n_samples = nrows
    X, y = make_blobs(n_samples=n_samples,
                      n_features=n_feats, random_state=0)

    knn_cu = cuKNN()

    if input_type == 'ndarray':

        knn_cu.fit(X)
        D_cuml, I_cuml = knn_cu.kneighbors(X, k)

        print(str(D_cuml))
        print(str(I_cuml))
        assert type(D_cuml) == np.ndarray
        assert type(I_cuml) == np.ndarray

    D_cuml_arr = D_cuml
    I_cuml_arr = I_cuml

    if nrows < 500000:
        knn_sk = skKNN(metric="euclidean")
        knn_sk.fit(X)
        D_sk, I_sk = knn_sk.kneighbors(X, k)

        assert array_equal(D_cuml_arr, D_sk, 1e-2, with_sign=True)
        assert I_cuml_arr.all() == I_sk.all()


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
@pytest.mark.parametrize('nrows', [unit_param(20), quality_param(5000),
                         stress_param(500000)])
@pytest.mark.parametrize('n_feats', [unit_param(3), quality_param(100),
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
