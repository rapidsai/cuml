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


from cuml.experimental.cluster import HDBSCAN
from sklearn.datasets import make_blobs

from cuml.metrics import adjusted_rand_score
from cuml.test.utils import get_pattern

import numpy as np

from cuml.common import logger

import hdbscan

from sklearn import datasets

import cupy as cp

test_datasets = {
 "digits": datasets.load_digits(),
 "boston": datasets.load_boston(),
 "diabetes": datasets.load_diabetes(),
 "cancer": datasets.load_breast_cancer(),
}

dataset_names = ['noisy_circles', 'noisy_moons', 'varied']#, 'aniso']


@pytest.mark.parametrize('nrows', [25])
@pytest.mark.parametrize('ncols', [25])
@pytest.mark.parametrize('nclusters', [2])
@pytest.mark.parametrize('min_samples', [3])
@pytest.mark.parametrize('allow_single_cluster', [True])
@pytest.mark.parametrize('min_cluster_size', [2])
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
                      cluster_std=0.7,
                      shuffle=False,
                      random_state=42)

    logger.set_level(logger.level_debug)
    cuml_agg = HDBSCAN(verbose=logger.level_debug,
                       allow_single_cluster=allow_single_cluster,
                       n_neighbors=min_samples+1,
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

    # import numpy as np
    # np.set_printoptions(threshold=np.inf)
    #
    sk_agg.fit(cp.asnumpy(X))
    #
    # print("cu condensed: %s" % cuml_agg.condensed_lambdas_[:101])
    # print("cu condensed: %s" % cuml_agg.condensed_parent_[:101])
    # print("cu condensed: %s" % cuml_agg.condensed_child_[:101])
    #
    # print("sk labels: %s" % sk_agg.labels_)
    #
    #
    # import numpy as np
    # print("unique labels: %s" % np.unique(sk_agg.labels_))
    #
    # print("sk condensed: %s" % sk_agg.condensed_tree_.to_numpy())
    #
    # t = sk_agg.condensed_tree_.to_numpy()
    #
    # print("parent min: %s" % t['parent'].min())
    #
    # print("condensed tree size: %s" % t.shape)
    #
    # print("sk condensed parent max: %s" % t["lambda_val"][t['parent'] == 100].max())
    #
    # print("Cluster tree: %s" % t[t['child_size']>1])

    # Cluster assignments should be exact, even though the actual
    # labels may differ
    # TODO: Investigating a couiple very small label differences
    assert(adjusted_rand_score(cuml_agg.labels_, sk_agg.labels_) >= 0.95)
    assert(len(np.unique(sk_agg.labels_)) == len(cp.unique(cuml_agg.labels_)))


@pytest.mark.parametrize('dataset', test_datasets.values())

# TODO: Fix crash when min_samples is changes (due to MST determinism precision error)
@pytest.mark.parametrize('cluster_selection_epsilon', [0.0])
@pytest.mark.parametrize('min_samples_cluster_size_bounds', [(150, 150, 0),
                                                             (15, 5, 0),
                                                             (50, 25, 0)
                                                            ])

# TODO: Fix small discrepancies in allow_single_cluster=False (single test failure)
@pytest.mark.parametrize('allow_single_cluster', [True, False])

# TODO: Verify/fix discrepancies in leaf selection method
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

    print("points: %s" % X.shape[0])

    logger.set_level(logger.level_debug)
    cuml_agg = HDBSCAN(verbose=logger.level_debug,
                       allow_single_cluster=allow_single_cluster,
                       n_neighbors=min_samples+1,
                       gen_min_span_tree=True,
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

    tree = sk_agg.condensed_tree_.to_numpy()
    cluster_tree = tree[tree['child_size'] > 1]
    print("sk_parent", cluster_tree['parent'] - 1797)
    print("sk_child", cluster_tree['child'] - 1797)
    print("sk_size", cluster_tree['child_size'])
    # is_cluster = np.zeros(cluster_tree['child'].max() - cluster_tree['parent'].min() + 1)
    # for c in cluster_tree['child']:
    #     if c not in cluster_tree['parent']:
    #         is_cluster[c - 1797] = 1
    # print("sk_is_cluster", is_cluster)

    # import numpy as np
    # np.set_printoptions(threshold=np.inf)
    #
    # print("sk labels: %s" % sk_agg.labels_[:25])
    #
    # print("cu condensed: %s" % cuml_agg.condensed_lambdas_)
    # print("cu condensed: %s" % cuml_agg.condensed_parent_)
    # print("cu condensed: %s" % cuml_agg.condensed_child_)
    # print("cu condensed: %s" % cuml_agg.condensed_sizes_)
    #
    # print("sk condensed: %s" % sk_agg.condensed_tree_.to_numpy())

    print("sk counts: %s" % str(np.unique(sk_agg.labels_, return_counts=True)))
    print("cu counts: %s" % str(np.unique(cuml_agg.labels_, return_counts=True)))
    #
    # cu_asmnt = np.sort(np.unique(cuml_agg.labels_, return_counts=True)[1])
    # sk_asmnt = np.sort(np.unique(sk_agg.labels_, return_counts=True)[1])
    #
    # print("cu stabilities: %s" % cuml_agg.stabilities_.to_output("numpy"))
    # print("sk stabiliies: %s" % sk_agg.cluster_persistence_)
    #
    #
    # t = cuml_agg.condensed_sizes_>1
    # print("cu cluster tree parent %s" % cuml_agg.condensed_parent_[t])
    # print("cu cluster tree child %s" % cuml_agg.condensed_child_[t])
    # print("cu cluster tree lambdas %s" % cuml_agg.condensed_lambdas_[t])
    # print("cu cluster tree sizes %s" % cuml_agg.condensed_sizes_[t])

    # t = sk_agg.condensed_tree_.to_numpy()

    # print("Cluster tree: %s" % t[t['child_size']>1])
    #
    # print("single linkage tree %s" % sk_agg.single_linkage_tree_.to_numpy())
    #
    # print("sk dendrogram parents: %s" % np.array2string(sk_agg.minimum_spanning_tree_.to_numpy()[:,0].astype('int32'), separator=","))
    # print("sk dendrogram children: %s" % np.array2string(sk_agg.minimum_spanning_tree_.to_numpy()[:,1].astype('int32'), separator=","))
    # print("sk dendrogram lambdas: %s" % np.array2string(sk_agg.minimum_spanning_tree_.to_numpy()[:,2].astype('float32'), separator=","))
    # print("cu children: %s" % cuml_agg.children_)
    # print("cu sizes: %s" % cuml_agg.sizes_)


    # print("cu mst_total: %s" % cp.sum(cuml_agg.mst_weights_))
    # print("cu mst_total: %s" % cuml_agg.mst_src_)
    # print("cu mst_total: %s" % cuml_agg.mst_dst_)

    # print("cu mst: %s" % cuml_agg.mst_weights_)
    # print("sk mst_total: %s" % np.sum(sk_agg.minimum_spanning_tree_.to_numpy()[:,2]))
    # print("sk mst: %s" % sk_agg.minimum_spanning_tree_.to_numpy()[:,0])
    # print("sk mst: %s" % sk_agg.minimum_spanning_tree_.to_numpy()[:,1])
    # print("sk mst: %s" % sk_agg.minimum_spanning_tree_.to_numpy()[:,2])

    # cu_mst_src = cp.asnumpy(cuml_agg.mst_src_).reshape((-1, 1))
    # cu_mst_dst = cp.asnumpy(cuml_agg.mst_dst_).reshape((-1, 1))
    # cu_mst_wt = np.array(cp.asnumpy(cuml_agg.mst_weights_).reshape((-1, 1)))

    # sk_mst_src = sk_agg.minimum_spanning_tree_.to_numpy()[:,0].reshape((-1, 1))
    # sk_mst_dst = sk_agg.minimum_spanning_tree_.to_numpy()[:,1].reshape((-1, 1))
    # sk_mst_wt = sk_agg.minimum_spanning_tree_.to_numpy()[:,2].reshape((-1, 1))

    # cu_mst = np.hstack((cu_mst_src, cu_mst_dst, cu_mst_wt))
    # sk_mst = np.hstack((sk_mst_src, sk_mst_dst, sk_mst_wt))

    # cu_mst_sym = np.hstack((cu_mst_dst, cu_mst_src, cu_mst_wt))
    # sk_mst_sym = np.hstack((sk_mst_dst, sk_mst_src, sk_mst_wt))

    # cu_mst = np.vstack((cu_mst, cu_mst_sym))
    # sk_mst = np.vstack((sk_mst, sk_mst_sym))

    # cu_mst_set = set([tuple(x[:2]) for x in cu_mst])
    # sk_mst_set = set([tuple(x[:2]) for x in sk_mst])
    # print(cu_mst_set)
    # print(sk_mst_set)
    # inter = cu_mst_set & sk_mst_set
    # print(inter)

    # new_cu_mst = np.array([x for x in cu_mst if tuple(x[:2]) not in inter])
    # new_sk_mst = np.array([x for x in sk_mst if tuple(x[:2]) not in inter])

    # cu_mst_indices = np.argsort(new_cu_mst[:, 2])
    # sk_mst_indices = np.argsort(new_sk_mst[:, 2])

    # new_cu_mst = new_cu_mst[cu_mst_indices, :]
    # new_sk_mst = new_sk_mst[sk_mst_indices, :]

    # print(np.sum(cu_mst[:, 2]))
    # print(np.sum(sk_mst[:, 2]))
    # print(new_cu_mst[np.around(new_cu_mst[:, 2], 4) != np.around(new_sk_mst[:, 2], 4), :])
    # print(new_cu_mst)
    # print(np.sum(new_cu_mst[:, 2]))
    # print(new_sk_mst)
    # print(new_sk_mst[np.around(new_cu_mst[:, 2], 4) != np.around(new_sk_mst[:, 2], 4), :])

    # print(sk_mst[np.abs(cu_mst[:, 2] - sk_mst[:, 2]) >= 1e-3, :])
    #
    # print("cu mst max: %s" % cp.sum(cuml_agg.mst_weights_))
    # print("sk mst max: %s" % np.sum(sk_agg.minimum_spanning_tree_.to_numpy()[:,2]))

    # print("sk linka: %s" % sk_agg.single_linkage_tree_.to_numpy())
    # print("cu linka: %s" % cuml_agg.single_linkage_tree_().to_numpy())
    #

    # print("sk mst weights: %s" % np.sort(sk_agg.minimum_spanning_tree_.to_numpy()[:,2]))
    # print("cu mst weights: %s" % np.sort(cuml_agg.minimum_spanning_tree_.to_numpy()[:,2]))

    # np.testing.assert_equal(cu_asmnt, sk_asmnt)

    # Cluster assignments should be exact, even though the actual
    # labels may differ.
    #
    # TODO: Investigating a couple very small label differences
    # assert(len(np.unique(sk_agg.labels_)) == len(cp.unique(cuml_agg.labels_)))
    assert(adjusted_rand_score(cuml_agg.labels_, sk_agg.labels_) > 0.95)
    # assert(len(np.unique(sk_agg.labels_)) == len(cp.unique(cuml_agg.labels_)))


@pytest.mark.parametrize('nrows', [1500])
@pytest.mark.parametrize('dataset', dataset_names)
@pytest.mark.parametrize('min_samples', [5, 20, 50])
@pytest.mark.parametrize('cluster_selection_epsilon', [0.0])
@pytest.mark.parametrize('min_cluster_size', [5, 30])
@pytest.mark.parametrize('allow_single_cluster', [True, False])
@pytest.mark.parametrize('max_cluster_size', [0])
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

"""
 [2.93000000e+02 1.27600000e+03 2.60576284e+01]
 [3.00000000e+00 1.31000000e+03 2.60576284e+01]
 [1.31000000e+03 3.00000000e+00 2.60576284e+01]
 [1.27600000e+03 2.93000000e+02 2.60576284e+01]
 [1.50700000e+03 5.94000000e+02 2.60576284e+01]
 [5.94000000e+02 1.50700000e+03 2.60576284e+01]

  [2.79000000e+02 1.31000000e+03 2.60576267e+01]
 [1.27600000e+03 1.68600000e+03 2.60576267e+01]
 [1.31000000e+03 2.79000000e+02 2.60576267e+01]
 [1.68600000e+03 1.27600000e+03 2.60576267e+01]
"""