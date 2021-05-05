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
from cuml.datasets import make_blobs

from cuml.metrics import adjusted_rand_score

from cuml.common import logger

import hdbscan

import cupy as cp


@pytest.mark.parametrize('nrows', [100, 1000])
@pytest.mark.parametrize('ncols', [25, 50])
@pytest.mark.parametrize('nclusters', [2, 10])
@pytest.mark.parametrize('k', [3, 5, 15])
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
    cuml_agg = HDBSCAN(verbose=logger.level_info, min_samples=k)

    try:
        cuml_agg.fit(X)
    except Exception:
        cuml_agg.fit(X)

    sk_agg = hdbscan.HDBSCAN(min_samples=k)
    sk_agg.fit(cp.asnumpy(X))

    # Cluster assignments should be exact, even though the actual
    # labels may differ
    assert(adjusted_rand_score(cuml_agg.labels_, sk_agg.labels_) == 1.0)
