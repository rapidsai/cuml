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
import cupy as cp
import pytest

from cuml.datasets import make_blobs

from cuml.test.utils import get_pattern, clusters_equal, unit_param, \
    quality_param, stress_param

from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


dataset_names = ['blobs', 'noisy_circles', 'noisy_moons', 'varied', 'aniso']

SCORE_EPS = 0.06


@pytest.mark.parametrize('name', dataset_names)
@pytest.mark.parametrize('nrows', [unit_param(1000),
                                   quality_param(5000),
                                   stress_param(500000)])
def test_kmeans_sklearn_comparison(name, nrows):

    default_base = {'quantile': .3,
                    'eps': .3,
                    'damping': .9,
                    'preference': -200,
                    'n_neighbors': 10,
                    'n_clusters': 3}

    pat = get_pattern(name, nrows)

    params = default_base.copy()
    params.update(pat[1])

    cuml_kmeans = cuml.KMeans(n_clusters=params['n_clusters'])

    X, y = pat[0]

    X = StandardScaler().fit_transform(X)

    cu_y_pred = cuml_kmeans.fit_predict(X).to_array()

    if nrows < 500000:
        kmeans = cluster.KMeans(n_clusters=params['n_clusters'])
        sk_y_pred = kmeans.fit_predict(X)

        # Noisy circles clusters are rotated in the results,
        # since we are comparing 2 we just need to compare that both clusters
        # have approximately the same number of points.
        calculation = (np.sum(sk_y_pred) - np.sum(cu_y_pred))/len(sk_y_pred)
        score_test = (cuml_kmeans.score(X) - kmeans.score(X)) < 2e-3
        if name == 'noisy_circles':
            assert (calculation < 4e-3) and score_test

        else:
            if name == 'aniso':
                # aniso dataset border points tend to differ in the frontier
                # between clusters when compared to sklearn
                tol = 2e-2
            else:
                # We allow up to 5 points to be different for the other
                # datasets to be robust to small behavior changes
                # between library versions/ small changes. Visually it is
                # very clear that the algorithm work. Will add option
                # to plot if desired in a future version.
                tol = 1e-2
            assert (clusters_equal(sk_y_pred, cu_y_pred,
                    params['n_clusters'], tol=tol)) and score_test


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

    cuml_kmeans = cuml.KMeans(n_clusters=params['n_clusters'])

    X, y = pat[0]

    X = StandardScaler().fit_transform(X)

    cu_y_pred = cuml_kmeans.fit_predict(X)
    cu_score = adjusted_rand_score(cu_y_pred, y)
    kmeans = cluster.KMeans(random_state=12, n_clusters=params['n_clusters'])
    sk_y_pred = kmeans.fit_predict(X)
    sk_score = adjusted_rand_score(sk_y_pred, y)

    # cuML score should be in a close neighborhood around scikit-learn's
    assert sk_score - 0.03 <= cu_score <= sk_score + 0.03


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
                              oversampling_factor=oversampling_factor,
                              max_samples_per_batch=max_samples_per_batch)

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
                      random_state=10)

    cuml_kmeans = cuml.KMeans(verbose=1, init="k-means||",
                              n_clusters=nclusters,
                              random_state=10)

    cuml_kmeans.fit(X)

    actual_score = cuml_kmeans.score(X)

    predictions = cuml_kmeans.predict(X)

    centers = cp.array(cuml_kmeans.cluster_centers_.as_gpu_matrix())

    expected_score = 0
    for idx, label in enumerate(predictions):

        x = X[idx]
        y = centers[label]

        dist = np.sqrt(np.sum((x - y)**2))
        expected_score += dist**2

    assert actual_score + SCORE_EPS \
        >= (-1*expected_score) \
        >= actual_score - SCORE_EPS
