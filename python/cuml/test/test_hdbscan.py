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

from cuml.common import logger

import hdbscan

import cupy as cp


# TODO: Tests that need to be written:
  # outlier points
  # multimodal data
  # different parameter settings
  # duplicate data points
  #


@pytest.mark.parametrize('nrows', [1000])
@pytest.mark.parametrize('ncols', [25, 50])
@pytest.mark.parametrize('nclusters', [2, 5, 10])
@pytest.mark.parametrize('k', [25])
@pytest.mark.parametrize('connectivity', ['knn'])
def test_hdbscan_sklearn_compare(nrows, ncols, nclusters,
                                        k, connectivity):

    X, y = make_blobs(int(nrows),
                      ncols,
                      nclusters,
                      cluster_std=1.0,
                      shuffle=False,
                      random_state=42)

    logger.set_level(logger.level_info)
    cuml_agg = HDBSCAN(verbose=logger.level_info, min_samples=k, n_neighbors=k, min_cluster_size=10)

    try:
        cuml_agg.fit(X)
    except Exception:
        cuml_agg.fit(X)

    sk_agg = hdbscan.HDBSCAN(min_samples=k, approx_min_span_tree=False, gen_min_span_tree=True, min_cluster_size=10, algorithm="generic")
    sk_agg.fit(cp.asnumpy(X))

    asort = cp.argsort(cuml_agg.condensed_child_.flatten())

    print("cu unique parents: %s" % cp.unique(cuml_agg.condensed_parent_.flatten()))

    print("cu condensed_parents: %s" % cuml_agg.condensed_parent_.flatten()[asort])
    print("cu condensed_children: %s" % cuml_agg.condensed_child_.flatten()[asort])

    print("cu_children_src: %s" % cuml_agg.children_[0,-10:])
    print("cu_children_dst: %s" % cuml_agg.children_[1,-10:])
    print("cu_sizes: %s" % cuml_agg.sizes_[-10:])
    print("cu_lambdas: %s" % cuml_agg.lambdas_[-10:])

    print("sk_children: %s" % sk_agg.single_linkage_tree_.to_numpy()[-10:])

    print("sk children shape %s" % str(sk_agg.single_linkage_tree_.to_numpy().shape))

    print("sk_children_1279: %s" % sk_agg.single_linkage_tree_.to_numpy()[279])
    print("sk_children_1996: %s" % sk_agg.single_linkage_tree_.to_numpy()[996])

    print("cu_children (inc): %s" % cuml_agg.children_.flatten()[:20])
    print("cu_sizes (inc): %s" % cuml_agg.sizes_[:10])

    print("cu mst_src: %s" % cuml_agg.mst_src_[:50])
    print("cu mst_dst: %s" % cuml_agg.mst_dst_[:50])
    print("cu mst_weight: %s" % cuml_agg.mst_weights_[:50])

    from hdbscan._hdbscan_tree import condense_tree
    import numpy as np

    # cuml_children = cuml_agg.children_[:, :nrows-1].astype("double")
    # cuml_children = np.vstack([cuml_children[:, 0], cuml_children[:, 1], cuml_agg.lambdas_.astype("double")[:nrows-1], cuml_agg.sizes_.astype("double")[:nrows-1]]).T
    #
    # print(str(cuml_children.shape))

    # print("condensed tree: %s" % condense_tree(cuml_children.T))

    print("sk_mst_weight: %s" % sk_agg.minimum_spanning_tree_.to_numpy()[np.argsort(sk_agg.minimum_spanning_tree_.to_numpy().T[2]),:])

    asort = np.argsort(sk_agg.condensed_tree_.to_numpy()["child"])

    print("sk unique parents: %s" % np.unique(sk_agg.condensed_tree_.to_numpy()["parent"]))

    print("sk condensed_parents: %s" % sk_agg.condensed_tree_.to_numpy()["parent"][asort])
    print("sk condensed_children: %s" % sk_agg.condensed_tree_.to_numpy()["child"][asort])

    print("sklabels: %s" % sk_agg.labels_)


    print("sk_children (int) %s" % sk_agg.single_linkage_tree_.to_numpy()[:10])

    # Cluster assignments should be exact, even though the actual
    # labels may differ
    assert(adjusted_rand_score(cuml_agg.labels_, sk_agg.labels_) == 1.0)
