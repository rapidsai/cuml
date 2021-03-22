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

from cuml.cluster import AgglomerativeClustering
from cuml.datasets import make_blobs

from cuml.metrics import adjusted_rand_score

from sklearn import cluster

from cuml.test.utils import array_equal

import cupy as cp


@pytest.mark.parametrize('nrows', [1000])
@pytest.mark.parametrize('ncols', [25])
@pytest.mark.parametrize('nclusters', [2, 5, 10, 50, 100])
@pytest.mark.parametrize('k', [3, 4])
def test_sklearn_compare(nrows, ncols, nclusters, k):

    X, y = make_blobs(int(nrows),
                      ncols,
                      nclusters,
                      cluster_std=1.0,
                      shuffle=False,
                      random_state=0)

    cuml_agg = AgglomerativeClustering(
        n_clusters=nclusters, affinity='euclidean', linkage='single',
        n_neighbors=k, connectivity='knn')

    cuml_agg.fit(X)

    print("cu labels: %s" % cuml_agg.labels_.to_output("numpy"))

    sk_agg = cluster.AgglomerativeClustering(
        n_clusters=nclusters, affinity='euclidean', linkage='single')

    sk_agg.fit(cp.asnumpy(X))

    cu_children = cp.asarray(cuml_agg.children_).T.get()
    sk_children = sk_agg.children_

    sk_children.sort(axis=1)
    cu_children.sort(axis=1)

    print("cu_children: " + str(cu_children))
    print("sk_children: " + str(sk_children))

    import numpy

    a = numpy.not_equal(cu_children, sk_children)

    print("NOT EQUAL: " + str(len(cu_children[a==True])))

    # assert array_equal(cu_children, sk_children)

    # Cluster assignments should be exact, even though the actual
    # labels may differ
    assert(adjusted_rand_score(cuml_agg.labels_, sk_agg.labels_) == 1.0)
