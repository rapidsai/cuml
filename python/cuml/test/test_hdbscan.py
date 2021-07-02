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

import pytest


from cuml.experimental.cluster import HDBSCAN
from sklearn.datasets import make_blobs

from cuml.metrics import adjusted_rand_score
from cuml.test.utils import get_pattern

import numpy as np

from cuml.common import logger

import hdbscan

from sklearn import datasets

import cupy as cp

import rmm

IS_A100 = 'A100' in rmm._cuda.gpu.deviceGetName(0)

test_datasets = {
 "digits": datasets.load_digits(),
 "boston": datasets.load_boston(),
 "diabetes": datasets.load_diabetes(),
 "cancer": datasets.load_breast_cancer(),
}

dataset_names = ['noisy_circles', 'noisy_moons', 'varied']


def assert_cluster_counts(sk_agg, cuml_agg, digits=25):
    sk_unique, sk_counts = np.unique(sk_agg.labels_, return_counts=True)
    sk_counts = np.sort(sk_counts)

    cu_unique, cu_counts = cp.unique(cuml_agg.labels_, return_counts=True)
    cu_counts = cp.sort(cu_counts).get()

    np.testing.assert_almost_equal(sk_counts, cu_counts, decimal=-1 * digits)


@pytest.mark.skipif(IS_A100, reason="Skipping test on A100")
@pytest.mark.parametrize('nrows', [500])
@pytest.mark.parametrize('ncols', [25])
@pytest.mark.parametrize('nclusters', [2, 5])
@pytest.mark.parametrize('min_samples', [25, 60])
@pytest.mark.parametrize('allow_single_cluster', [True, False])
@pytest.mark.parametrize('min_cluster_size', [30, 50])
@pytest.mark.parametrize('cluster_selection_epsilon', [0.0])
@pytest.mark.parametrize('max_cluster_size', [0])
@pytest.mark.parametrize('cluster_selection_method', ['eom', 'leaf'])
@pytest.mark.parametrize('connectivity', ['knn'])
def test_hdbscan_blobs(nrows, ncols, nclusters,
                       connectivity,
                       cluster_selection_epsilon,
                       cluster_selection_method,
                       allow_single_cluster,
                       min_cluster_size,
                       max_cluster_size,
                       min_samples):

    X, y = make_blobs(int(nrows),
                      ncols,
                      nclusters,
                      cluster_std=0.7,
                      shuffle=False,
                      random_state=42)

    cuml_agg = HDBSCAN(verbose=logger.level_info,
                       allow_single_cluster=allow_single_cluster,
                       n_neighbors=min_samples+1,
                       min_samples=min_samples,
                       max_cluster_size=max_cluster_size,
                       min_cluster_size=min_cluster_size,
                       cluster_selection_epsilon=cluster_selection_epsilon,
                       cluster_selection_method=cluster_selection_method)

    cuml_agg.fit(X)
    sk_agg = hdbscan.HDBSCAN(
        allow_single_cluster=allow_single_cluster,
        approx_min_span_tree=False,
        gen_min_span_tree=True,
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        algorithm="generic")

    sk_agg.fit(cp.asnumpy(X))

    assert_cluster_counts(sk_agg, cuml_agg)

    assert(adjusted_rand_score(cuml_agg.labels_, sk_agg.labels_) >= 0.95)
    assert(len(np.unique(sk_agg.labels_)) == len(cp.unique(cuml_agg.labels_)))

    assert np.allclose(np.sort(sk_agg.cluster_persistence_),
           np.sort(cuml_agg.cluster_persistence_), rtol=0.01, atol=0.01)


@pytest.mark.skipif(IS_A100, reason="Skipping test on A100")
@pytest.mark.parametrize('dataset', test_datasets.values())
@pytest.mark.parametrize('cluster_selection_epsilon', [0.0, 50.0, 150.0])
@pytest.mark.parametrize('min_samples_cluster_size_bounds', [(150, 150, 0),
                                                             (50, 25, 0)])
@pytest.mark.parametrize('allow_single_cluster', [True, False])
@pytest.mark.parametrize('cluster_selection_method', ['eom', 'leaf'])
@pytest.mark.parametrize('connectivity', ['knn'])
def test_hdbscan_sklearn_datasets(dataset,
                                  connectivity,
                                  cluster_selection_epsilon,
                                  cluster_selection_method,
                                  min_samples_cluster_size_bounds,
                                  allow_single_cluster):

    min_samples, min_cluster_size, max_cluster_size = \
        min_samples_cluster_size_bounds

    X = dataset.data

    cuml_agg = HDBSCAN(verbose=logger.level_info,
                       allow_single_cluster=allow_single_cluster,
                       n_neighbors=min_samples,
                       gen_min_span_tree=True,
                       min_samples=min_samples,
                       max_cluster_size=max_cluster_size,
                       min_cluster_size=min_cluster_size,
                       cluster_selection_epsilon=cluster_selection_epsilon,
                       cluster_selection_method=cluster_selection_method)

    cuml_agg.fit(X)

    sk_agg = hdbscan.HDBSCAN(
        allow_single_cluster=allow_single_cluster,
        approx_min_span_tree=False,
        gen_min_span_tree=True,
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        algorithm="generic")

    sk_agg.fit(cp.asnumpy(X))

    assert_cluster_counts(sk_agg, cuml_agg)

    assert(len(np.unique(sk_agg.labels_)) == len(cp.unique(cuml_agg.labels_)))
    assert(adjusted_rand_score(cuml_agg.labels_, sk_agg.labels_) > 0.85)

    assert np.allclose(np.sort(sk_agg.cluster_persistence_),
           np.sort(cuml_agg.cluster_persistence_), rtol=0.1, atol=0.1)


@pytest.mark.skipif(IS_A100, reason="Skipping test on A100")
@pytest.mark.parametrize('nrows', [1000])
@pytest.mark.parametrize('dataset', dataset_names)
@pytest.mark.parametrize('min_samples', [15])
@pytest.mark.parametrize('cluster_selection_epsilon', [0.0])
@pytest.mark.parametrize('min_cluster_size', [25])
@pytest.mark.parametrize('allow_single_cluster', [True, False])
@pytest.mark.parametrize('max_cluster_size', [0])
# TODO: Need to test leaf selection method
@pytest.mark.parametrize('cluster_selection_method', ['eom'])
@pytest.mark.parametrize('connectivity', ['knn'])
def test_hdbscan_cluster_patterns(dataset, nrows,
                                  connectivity,
                                  cluster_selection_epsilon,
                                  cluster_selection_method,
                                  min_cluster_size,
                                  allow_single_cluster,
                                  max_cluster_size,
                                  min_samples):

    # This also tests duplicate data points
    X, y = get_pattern(dataset, nrows)[0]

    cuml_agg = HDBSCAN(verbose=logger.level_info,
                       allow_single_cluster=allow_single_cluster,
                       n_neighbors=min_samples,
                       min_samples=min_samples,
                       max_cluster_size=max_cluster_size,
                       min_cluster_size=min_cluster_size,
                       cluster_selection_epsilon=cluster_selection_epsilon,
                       cluster_selection_method=cluster_selection_method)

    cuml_agg.fit(X)

    sk_agg = hdbscan.HDBSCAN(
        allow_single_cluster=allow_single_cluster,
        approx_min_span_tree=False,
        gen_min_span_tree=True,
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        algorithm="generic")

    sk_agg.fit(cp.asnumpy(X))

    assert_cluster_counts(sk_agg, cuml_agg)

    assert(len(np.unique(sk_agg.labels_)) == len(cp.unique(cuml_agg.labels_)))
    assert(adjusted_rand_score(cuml_agg.labels_, sk_agg.labels_) > 0.95)

    assert np.allclose(np.sort(sk_agg.cluster_persistence_),
           np.sort(cuml_agg.cluster_persistence_), rtol=0.1, atol=0.1)


def test_hdbscan_plots():

    X, y = make_blobs(int(100),
                      100,
                      10,
                      cluster_std=0.7,
                      shuffle=False,
                      random_state=42)

    cuml_agg = HDBSCAN(gen_min_span_tree=True)
    cuml_agg.fit(X)

    assert cuml_agg.condensed_tree_ is not None
    assert cuml_agg.minimum_spanning_tree_ is not None
    assert cuml_agg.single_linkage_tree_ is not None

    cuml_agg = HDBSCAN(gen_min_span_tree=False)
    cuml_agg.fit(X)

    assert cuml_agg.minimum_spanning_tree_ is None
