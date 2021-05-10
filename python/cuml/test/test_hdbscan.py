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

import pytest


from cuml.cluster import HDBSCAN
from sklearn.datasets import make_blobs

from cuml.metrics import adjusted_rand_score
from cuml.test.utils import get_pattern

from cuml.common import logger

import hdbscan

from sklearn import datasets

import cupy as cp

test_datasets = {
 "digits": datasets.load_digits(),
 "boston": datasets.load_boston(),
 "diabetes": datasets.load_diabetes(),
 # "cancer": datasets.load_breast_cancer(),
}

dataset_names = ['noisy_circles', 'noisy_moons', 'varied']#, 'aniso']


@pytest.mark.parametrize('nrows', [100])
@pytest.mark.parametrize('ncols', [25])
@pytest.mark.parametrize('nclusters', [1, 10, 50])
@pytest.mark.parametrize('min_samples', [25])
@pytest.mark.parametrize('allow_single_cluster', [True, False])
@pytest.mark.parametrize('min_cluster_size', [10])
@pytest.mark.parametrize('cluster_selection_epsilon', [0.0])
@pytest.mark.parametrize('max_cluster_size', [0])
@pytest.mark.parametrize('cluster_selection_method', ['eom'])
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
                      cluster_std=1.0,
                      shuffle=False,
                      random_state=42)

    logger.set_level(logger.level_debug)
    cuml_agg = HDBSCAN(verbose=logger.level_debug,
                       allow_single_cluster=allow_single_cluster,
                       n_neighbors=min_samples,
                       min_samples=min_samples,
                       max_cluster_size=max_cluster_size,
                       min_cluster_size=min_cluster_size,
                       cluster_selection_epsilon=cluster_selection_epsilon,
                       cluster_selection_method=cluster_selection_method)

    cuml_agg.fit(X)
    sk_agg = hdbscan.HDBSCAN(allow_single_cluster=allow_single_cluster,
                             approx_min_span_tree=False,
                             gen_min_span_tree=True,
                             min_samples=min_samples,
                             #max_cluster_size=max_cluster_size,
                             min_cluster_size=min_cluster_size,
                             cluster_selection_epsilon=cluster_selection_epsilon,
                             cluster_selection_method=cluster_selection_method,
                             algorithm="generic")

    sk_agg.fit(cp.asnumpy(X))

    print("cu condensed: %s" % cuml_agg.condensed_lambdas_[:101])
    print("cu condensed: %s" % cuml_agg.condensed_parent_[:101])
    print("cu condensed: %s" % cuml_agg.condensed_child_[:101])

    print("sk labels: %s" % sk_agg.labels_)


    import numpy as np
    print("unique labels: %s" % np.unique(sk_agg.labels_))

    print("sk condensed: %s" % sk_agg.condensed_tree_.to_numpy())

    t = sk_agg.condensed_tree_.to_numpy()

    print("parent min: %s" % t['parent'].min())

    print("condensed tree size: %s" % t.shape)

    print("sk condensed parent max: %s" % t["lambda_val"][t['parent'] == 100].max())

    print("Cluster tree: %s" % t[t['child_size']>1])

    # Cluster assignments should be exact, even though the actual
    # labels may differ
    # TODO: Investigating a couiple very small label differences
    assert(adjusted_rand_score(cuml_agg.labels_, sk_agg.labels_) >= 0.95)


@pytest.mark.parametrize('dataset', [test_datasets["diabetes"]])
@pytest.mark.parametrize('min_samples', [25])
@pytest.mark.parametrize('cluster_selection_epsilon', [0.0])
@pytest.mark.parametrize('min_cluster_size', [50])
@pytest.mark.parametrize('allow_single_cluster', [True])
@pytest.mark.parametrize('max_cluster_size', [0])
@pytest.mark.parametrize('cluster_selection_method', ['eom'])
@pytest.mark.parametrize('connectivity', ['knn'])
def test_hdbscan_sklearn_datasets(dataset,
                                  connectivity,
                                  cluster_selection_epsilon,
                                  cluster_selection_method,
                                  min_cluster_size,
                                  allow_single_cluster,
                                  max_cluster_size,
                                  min_samples):

    X = dataset.data

    print("points: %s" % X.shape[0])

    logger.set_level(logger.level_debug)
    cuml_agg = HDBSCAN(verbose=logger.level_debug,
                       allow_single_cluster=allow_single_cluster,
                       n_neighbors=min_samples,
                       min_samples=min_samples,
                       max_cluster_size=max_cluster_size,
                       min_cluster_size=min_cluster_size,
                       cluster_selection_epsilon=cluster_selection_epsilon,
                       cluster_selection_method=cluster_selection_method)

    cuml_agg.fit(X)

    # print("condensed_parents: %s" % cuml_agg.condensed_parent_[:])
    # print("condensed child: %s" % cuml_agg.condensed_child_)

    sk_agg = hdbscan.HDBSCAN(allow_single_cluster=allow_single_cluster,
                             approx_min_span_tree=False,
                             gen_min_span_tree=True,
                             min_samples=min_samples,
                             #max_cluster_size=max_cluster_size,
                             min_cluster_size=min_cluster_size,
                             cluster_selection_epsilon=cluster_selection_epsilon,
                             cluster_selection_method=cluster_selection_method,
                             algorithm="generic")
    sk_agg.fit(cp.asnumpy(X))

    print("sk labels: %s" % sk_agg.labels_[:25])

    print("cu condensed: %s" % cuml_agg.condensed_lambdas_[:101])
    print("cu condensed: %s" % cuml_agg.condensed_parent_[:101])
    print("cu condensed: %s" % cuml_agg.condensed_child_[:101])

    print("sk condensed: %s" % sk_agg.condensed_tree_.to_numpy())

    import numpy as np

    print("sk counts: %s" % str(np.unique(sk_agg.labels_, return_counts=True)))
    print("cu counts: %s" % str(np.unique(cuml_agg.labels_, return_counts=True)))

    cu_asmnt = np.sort(np.unique(cuml_agg.labels_, return_counts=True)[1])
    sk_asmnt = np.sort(np.unique(sk_agg.labels_, return_counts=True)[1])

    print("cu stabilities: %s" % cuml_agg.stabilities_.to_output("numpy"))
    print("sk stabiliies: %s" % sk_agg.cluster_persistence_)

    # np.testing.assert_equal(cu_asmnt, sk_asmnt)

    # Cluster assignments should be exact, even though the actual
    # labels may differ.
    #
    # TODO: Investigating a couple very small label differences
    assert(adjusted_rand_score(cuml_agg.labels_, sk_agg.labels_) > 0.95)


@pytest.mark.parametrize('nrows', [1000])
@pytest.mark.parametrize('dataset', dataset_names)
@pytest.mark.parametrize('min_samples', [25])
@pytest.mark.parametrize('cluster_selection_epsilon', [0.0])
@pytest.mark.parametrize('min_cluster_size', [10])
@pytest.mark.parametrize('allow_single_cluster', [True])
@pytest.mark.parametrize('max_cluster_size', [0])
@pytest.mark.parametrize('cluster_selection_method', ['eom', 'leaf'])
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

    logger.set_level(logger.level_debug)
    cuml_agg = HDBSCAN(verbose=logger.level_debug,
                       allow_single_cluster=allow_single_cluster,
                       n_neighbors=min_samples,
                       min_samples=min_samples,
                       max_cluster_size=max_cluster_size,
                       min_cluster_size=min_cluster_size,
                       cluster_selection_epsilon=cluster_selection_epsilon,
                       cluster_selection_method=cluster_selection_method)

    cuml_agg.fit(X)

    sk_agg = hdbscan.HDBSCAN(allow_single_cluster=allow_single_cluster,
                             approx_min_span_tree=False,
                             gen_min_span_tree=True,
                             min_samples=min_samples,
                             #max_cluster_size=max_cluster_size,
                             min_cluster_size=min_cluster_size,
                             cluster_selection_epsilon=cluster_selection_epsilon,
                             cluster_selection_method=cluster_selection_method,
                             algorithm="generic")
    sk_agg.fit(cp.asnumpy(X))

    print("sk labels: %s" % sk_agg.labels_)

    # Cluster assignments should be exact, even though the actual
    # labels may differ.
    #
    # TODO: Investigating a couple very small label differences
    assert(adjusted_rand_score(cuml_agg.labels_, sk_agg.labels_) > 0.95)

