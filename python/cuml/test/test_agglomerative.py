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

import random

import cuml
import cuml.common.logger as logger
import numpy as np
import pytest

from cuml.cluster import AgglomerativeClustering
from cuml.datasets import make_blobs

from cuml.test.utils import get_pattern, unit_param, \
    quality_param, stress_param, array_equal

from cuml.metrics import adjusted_rand_score

from sklearn import cluster

import cupy as cp


@pytest.mark.parametrize('nrows', [1000, 10000])
@pytest.mark.parametrize('ncols', [25])
@pytest.mark.parametrize('nclusters', [2, 5, 10])
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
        n_neighbors=k)

    cuml_agg.fit(X)

    sk_agg = cluster.AgglomerativeClustering(
        n_clusters=nclusters, affinity='euclidean', linkage='single')

    sk_agg.fit(cp.asnumpy(X))

    # Cluster assignments should be exact, even though the actual
    # labels may differ
    assert(adjusted_rand_score(cuml_agg.labels_, sk_agg.labels_) == 1.0)
