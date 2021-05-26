# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

import random

import cuml
import cupy as cp
import cuml.common.logger as logger
import numpy as np
import pytest

from cuml.datasets import make_blobs

from cuml.test.utils import get_pattern, unit_param, \
    quality_param, stress_param, array_equal

from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


dataset_names = ['blobs', 'noisy_circles', 'noisy_moons', 'varied', 'aniso']


@pytest.fixture
def get_data_consistency_test():
    cluster_std = 1.0
    nrows = 1000
    ncols = 50
    nclusters = 8

    X, y = make_blobs(nrows,
                      ncols,
                      nclusters,
                      cluster_std=cluster_std,
                      shuffle=False,
                      random_state=0)
    return X, y


@pytest.fixture
def random_state():
    random_state = random.randint(0, 1e6)
    with logger.set_level(logger.level_debug):
        logger.debug("Random seed: {}".format(random_state))
    return random_state


@pytest.mark.xfail
def test_n_init_cluster_consistency(random_state):

    nclusters = 8
    X, y = get_data_consistency_test()

    cuml_kmeans = cuml.KMeans(init="k-means++",
                              n_clusters=nclusters,
                              n_init=10,
                              random_state=random_state,
                              output_type='numpy')

    cuml_kmeans.fit(X)
    initial_clusters = cuml_kmeans.cluster_centers_

    cuml_kmeans = cuml.KMeans(init="k-means++",
                              n_clusters=nclusters,
                              n_init=10,
                              random_state=random_state,
                              output_type='numpy')

    cuml_kmeans.fit(X)

    assert array_equal(initial_clusters, cuml_kmeans.cluster_centers_)


@pytest.mark.parametrize('nrows', [1000, 10000])
@pytest.mark.parametrize('ncols', [25])
@pytest.mark.parametrize('nclusters', [2, 5])
def test_traditional_kmeans_plus_plus_init(nrows, ncols, nclusters,
                                           random_state):

    # Using fairly high variance between points in clusters
    cluster_std = 1.0

    X, y = make_blobs(int(nrows),
                      ncols,
                      nclusters,
                      cluster_std=cluster_std,
                      shuffle=False,
                      random_state=0)

    cuml_kmeans = cuml.KMeans(init="k-means++",
                              n_clusters=nclusters,
                              n_init=10,
                              random_state=random_state,
                              output_type='numpy')

    cuml_kmeans.fit(X)
    cu_score = cuml_kmeans.score(X)

    kmeans = cluster.KMeans(random_state=random_state,
                            n_clusters=nclusters)
    kmeans.fit(cp.asnumpy(X))
    sk_score = kmeans.score(cp.asnumpy(X))

    assert abs(cu_score - sk_score) <= cluster_std * 1.5


@pytest.mark.parametrize('nrows', [100, 500])
@pytest.mark.parametrize('ncols', [3, 5, 25])
@pytest.mark.parametrize('nclusters', [3, 5, 10])
def test_weighted_kmeans(nrows, ncols, nclusters,
                         random_state):

    # Using fairly high variance between points in clusters
    cluster_std = 10000.0
    np.random.seed(random_state)

    wt = np.array([0.00001 for j in range(nrows)])

    # Open the space really large

    bound = nclusters * 100000

    centers = np.random.uniform(-bound, bound,
                                size=(nclusters, ncols))

    X, y = make_blobs(nrows,
                      ncols,
                      centers=centers,
                      cluster_std=cluster_std,
                      shuffle=False,
                      random_state=0)

    # Choose one sample from each label and increase its weight
    for i in range(nclusters):
        wt[cp.argmax(cp.array(y) == i).item()] = 5000.0

    cuml_kmeans = cuml.KMeans(init="k-means++",
                              n_clusters=nclusters,
                              n_init=10,
                              random_state=random_state,
                              output_type='numpy')

    cuml_kmeans.fit(X, sample_weight=wt)

    for i in range(nrows):

        label = cuml_kmeans.labels_[i]
        actual_center = cuml_kmeans.cluster_centers_[label]

        diff = sum(abs(X[i].copy_to_host() - actual_center))

        # The large weight should be the centroid
        if wt[i] > 1.0:
            assert diff < 1.0

        # Otherwise it should be pretty far away
        else:
            assert diff > 1000.0


@pytest.mark.parametrize('nrows', [1000, 10000])
@pytest.mark.parametrize('ncols', [25])
@pytest.mark.parametrize('nclusters', [2, 5])
@pytest.mark.parametrize('cluster_std', [1.0, 0.1, 0.01])
def test_kmeans_clusters_blobs(nrows, ncols, nclusters,
                               random_state, cluster_std):

    X, y = make_blobs(int(nrows), ncols, nclusters,
                      cluster_std=cluster_std,
                      shuffle=False,
                      random_state=0)

    cuml_kmeans = cuml.KMeans(init="k-means||",
                              n_clusters=nclusters,
                              random_state=random_state,
                              output_type='numpy')

    preds = cuml_kmeans.fit_predict(X)

    assert adjusted_rand_score(cp.asnumpy(preds), cp.asnumpy(y)) >= 0.99


@pytest.mark.parametrize('name', dataset_names)
@pytest.mark.parametrize('nrows', [unit_param(1000),
                                   quality_param(5000)])
def test_kmeans_sklearn_comparison(name, nrows, random_state):

    default_base = {'quantile': .3,
                    'eps': .3,
                    'damping': .9,
                    'preference': -200,
                    'n_neighbors': 10,
                    'n_clusters': 3}

    pat = get_pattern(name, nrows)

    params = default_base.copy()
    params.update(pat[1])

    cuml_kmeans = cuml.KMeans(n_clusters=params['n_clusters'],
                              output_type='numpy',
                              init="k-means++",
                              random_state=random_state,
                              n_init=10)

    X, y = pat[0]

    X = StandardScaler().fit_transform(X)

    cu_y_pred = cuml_kmeans.fit_predict(X)
    cu_score = adjusted_rand_score(cu_y_pred, y)
    kmeans = cluster.KMeans(random_state=random_state,
                            n_clusters=params['n_clusters'])
    sk_y_pred = kmeans.fit_predict(X)
    sk_score = adjusted_rand_score(sk_y_pred, y)

    assert sk_score - 1e-2 <= cu_score <= sk_score + 1e-2


@pytest.mark.parametrize('name', dataset_names)
@pytest.mark.parametrize('nrows', [unit_param(500),
                                   quality_param(5000),
                                   stress_param(500000)])
def test_kmeans_sklearn_comparison_default(name, nrows, random_state):

    default_base = {'quantile': .3,
                    'eps': .3,
                    'damping': .9,
                    'preference': -200,
                    'n_neighbors': 10,
                    'n_clusters': 3}

    pat = get_pattern(name, nrows)

    params = default_base.copy()
    params.update(pat[1])

    cuml_kmeans = cuml.KMeans(n_clusters=params['n_clusters'],
                              random_state=random_state,
                              n_init=10,
                              output_type='numpy')

    X, y = pat[0]

    X = StandardScaler().fit_transform(X)

    cu_y_pred = cuml_kmeans.fit_predict(X)
    cu_score = adjusted_rand_score(cu_y_pred, y)
    kmeans = cluster.KMeans(random_state=random_state,
                            n_clusters=params['n_clusters'])
    sk_y_pred = kmeans.fit_predict(X)
    sk_score = adjusted_rand_score(sk_y_pred, y)

    assert sk_score - 1e-2 <= cu_score <= sk_score + 1e-2


@pytest.mark.parametrize(
    'max_iter, oversampling_factor, max_samples_per_batch, init', [
        (100, 0.5, 1 << 10, 'preset'),
        (1000, 1.0, 1 << 15, 'preset'),
        (500, 1.5, 1 << 5, 'k-means||'),
        (1000, 1.0, 1 << 10, 'random'),
        # Redundant case to better exercise 'k-means||'
        (1000, 1.0, 1 << 15, 'k-means||')
    ]
)
@pytest.mark.parametrize('n_clusters', [unit_param(10),
                                        unit_param(100),
                                        stress_param(1000)])
def test_all_kmeans_params(n_clusters, max_iter, init,
                           oversampling_factor, max_samples_per_batch,
                           random_state):

    np.random.seed(0)
    X = np.random.rand(1000, 10)

    if init == 'preset':
        init = np.random.rand(n_clusters, 10)

    cuml_kmeans = cuml.KMeans(n_clusters=n_clusters,
                              max_iter=max_iter,
                              init=init,
                              random_state=random_state,
                              oversampling_factor=oversampling_factor,
                              max_samples_per_batch=max_samples_per_batch,
                              output_type='cupy')

    cuml_kmeans.fit_predict(X)


@pytest.mark.parametrize('nrows', [unit_param(500),
                                   quality_param(5000),
                                   stress_param(500000)])
@pytest.mark.parametrize("ncols", [10, 30])
@pytest.mark.parametrize("nclusters", [unit_param(5), quality_param(10),
                                       stress_param(50)])
def test_score(nrows, ncols, nclusters, random_state):

    X, y = make_blobs(int(nrows), ncols, nclusters,
                      cluster_std=1.0,
                      shuffle=False,
                      random_state=0)

    cuml_kmeans = cuml.KMeans(init="k-means||",
                              n_clusters=nclusters,
                              random_state=random_state,
                              output_type='numpy')

    cuml_kmeans.fit(X)

    actual_score = cuml_kmeans.score(X)
    predictions = cuml_kmeans.predict(X)

    centers = cuml_kmeans.cluster_centers_

    expected_score = 0.0
    for idx, label in enumerate(predictions):
        x = X[idx, :]
        y = cp.array(centers[label, :], dtype=cp.float32)

        sq_euc_dist = cp.sum(cp.square((x - y)))
        expected_score += sq_euc_dist

    expected_score *= -1

    cp.testing.assert_allclose(
        actual_score, expected_score, atol=0.1, rtol=1e-5)
