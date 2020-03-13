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

import cuml
import numpy as np
import pytest

from cuml.datasets import make_blobs

from cuml.test.utils import get_pattern, unit_param, \
    quality_param, stress_param, array_equal

from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


dataset_names = ['blobs', 'noisy_circles', 'noisy_moons', 'varied', 'aniso']

SCORE_EPS = 0.06


@pytest.mark.xfail
@pytest.mark.parametrize('random_state', [i for i in range(10)])
def test_n_init_cluster_consistency(random_state):

    cluster_std = 1.0

    nrows = 100000
    ncols = 100
    nclusters = 8

    X, y = make_blobs(nrows,
                      ncols,
                      nclusters,
                      cluster_std=cluster_std,
                      shuffle=False,
                      random_state=0)

    cuml_kmeans = cuml.KMeans(verbose=0, init="k-means++",
                              n_clusters=nclusters,
                              n_init=10,
                              random_state=random_state,
                              output_type='numpy')

    cuml_kmeans.fit(X)
    initial_clusters = cuml_kmeans.cluster_centers_

    cuml_kmeans = cuml.KMeans(verbose=0, init="k-means++",
                              n_clusters=nclusters,
                              n_init=10,
                              random_state=random_state,
                              output_type='numpy')

    cuml_kmeans.fit(X)

    assert array_equal(initial_clusters, cuml_kmeans.cluster_centers_)


@pytest.mark.parametrize('nrows', [1000, 10000])
@pytest.mark.parametrize('ncols', [10, 50])
@pytest.mark.parametrize('nclusters', [2, 5])
@pytest.mark.parametrize('random_state', [i for i in range(50)])
def test_kmeans_sequential_plus_plus_init(nrows, ncols, nclusters,
                                          random_state):

    # Using fairly high variance between points in clusters
    cluster_std = 1.0

    X, y = make_blobs(nrows,
                      ncols,
                      nclusters,
                      cluster_std=cluster_std,
                      shuffle=False,
                      random_state=0)

    cuml_kmeans = cuml.KMeans(verbose=0, init="k-means++",
                              n_clusters=nclusters,
                              n_init=10,
                              random_state=random_state,
                              output_type='numpy')

    cuml_kmeans.fit(X)
    cu_score = cuml_kmeans.score(X)

    kmeans = cluster.KMeans(random_state=random_state,
                            n_clusters=nclusters)
    kmeans.fit(X.copy_to_host())
    sk_score = kmeans.score(X.copy_to_host())

    assert abs(cu_score - sk_score) <= cluster_std * 1.5


@pytest.mark.parametrize('nrows', [1000, 10000])
@pytest.mark.parametrize('ncols', [10, 50])
@pytest.mark.parametrize('nclusters', [2, 5])
@pytest.mark.parametrize('cluster_std', [1.0, 0.1, 0.01])
@pytest.mark.parametrize('random_state', [i for i in range(25)])
def test_kmeans_clusters_blobs(nrows, ncols, nclusters,
                               random_state, cluster_std):

    X, y = make_blobs(nrows, ncols, nclusters,
                      cluster_std=cluster_std,
                      shuffle=False,
                      random_state=random_state,)

    cuml_kmeans = cuml.KMeans(verbose=0, init="k-means||",
                              n_clusters=nclusters,
                              random_state=random_state,
                              output_type='numpy')

    preds = cuml_kmeans.fit_predict(X)

    assert adjusted_rand_score(preds, y) >= 0.99


@pytest.mark.parametrize('name', dataset_names)
@pytest.mark.parametrize('nrows', [unit_param(1000),
                                   quality_param(5000)])
def test_kmeans_sklearn_comparison(name, nrows):

    random_state = 12

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
    kmeans = cluster.KMeans(random_state=12,
                            n_clusters=params['n_clusters'])
    sk_y_pred = kmeans.fit_predict(X)
    sk_score = adjusted_rand_score(sk_y_pred, y)

    assert sk_score - 1e-2 <= cu_score <= sk_score + 1e-2


@pytest.mark.parametrize('name', dataset_names)
@pytest.mark.parametrize('nrows', [unit_param(500),
                                   quality_param(5000),
                                   stress_param(500000)])
def test_kmeans_sklearn_comparison_default(name, nrows):

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
                              random_state=12,
                              output_type='numpy')

    X, y = pat[0]

    X = StandardScaler().fit_transform(X)

    cu_y_pred = cuml_kmeans.fit_predict(X)
    cu_score = adjusted_rand_score(cu_y_pred, y)
    kmeans = cluster.KMeans(random_state=12,
                            n_clusters=params['n_clusters'])
    sk_y_pred = kmeans.fit_predict(X)
    sk_score = adjusted_rand_score(sk_y_pred, y)

    assert sk_score - 1e-2 <= cu_score <= sk_score + 1e-2


@pytest.mark.parametrize('n_rows', [unit_param(100),
                                    stress_param(500000)])
@pytest.mark.parametrize('n_clusters', [unit_param(10),
                                        unit_param(100),
                                        stress_param(1000)])
@pytest.mark.parametrize('max_iter', [100, 500, 1000])
@pytest.mark.parametrize('oversampling_factor', [0.5, 1.0, 1.5])
@pytest.mark.parametrize('max_samples_per_batch', [1 << 15, 1 << 10, 1 << 5])
@pytest.mark.parametrize('init', ['k-means||',
                                  'random',
                                  'preset'])
def test_all_kmeans_params(n_rows, n_clusters, max_iter, init,
                           oversampling_factor, max_samples_per_batch):

    np.random.seed(0)
    X = np.random.rand(1000, 10)

    if init == 'preset':
        init = np.random.rand(n_clusters, 10)

    cuml_kmeans = cuml.KMeans(n_clusters=n_clusters,
                              max_iter=max_iter,
                              init=init,
                              random_state=12,
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
def test_score(nrows, ncols, nclusters):

    X, y = make_blobs(nrows, ncols, nclusters,
                      cluster_std=0.01,
                      shuffle=False,
                      random_state=10)

    cuml_kmeans = cuml.KMeans(verbose=0, init="k-means||",
                              n_clusters=nclusters,
                              random_state=10,
                              output_type='numpy')

    cuml_kmeans.fit(X)

    actual_score = cuml_kmeans.score(X)

    predictions = cuml_kmeans.predict(X)

    centers = cuml_kmeans.cluster_centers_

    expected_score = 0
    for idx, label in enumerate(predictions):
        x = X[idx]
        y = centers[label]

        dist = np.sqrt(np.sum((x - y)**2))
        expected_score += dist**2

    assert actual_score + SCORE_EPS \
        >= (-1*expected_score) \
        >= actual_score - SCORE_EPS
