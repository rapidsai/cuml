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

import pytest
import numpy as np
import cuml
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from cuml.test.utils import fit_predict, get_pattern, clusters_equal

dataset_names = ['blobs', 'noisy_circles'] + \
                [pytest.param(ds, marks=pytest.mark.xfail)
                 for ds in ['noisy_moons', 'varied', 'aniso']]


@pytest.mark.parametrize('name', dataset_names)
@pytest.mark.parametrize('nrows', [pytest.param(20, marks=pytest.mark.unit),
                                   pytest.param(500000,
                                                marks=pytest.mark.stress),
                                   pytest.param(5000,
                                                marks=pytest.mark.quality)])
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

    kmeans = cluster.KMeans(n_clusters=params['n_clusters'])
    cuml_kmeans = cuml.KMeans(n_clusters=params['n_clusters'])

    X, y = pat[0]

    X = StandardScaler().fit_transform(X)

    clustering_algorithms = (
        ('sk_Kmeans', kmeans),
        ('cuml_Kmeans', cuml_kmeans),
    )

    sk_y_pred, _ = fit_predict(clustering_algorithms[0][1],
                               clustering_algorithms[0][0], X)

    cu_y_pred, _ = fit_predict(clustering_algorithms[1][1],
                               clustering_algorithms[1][0], X)

    # Noisy circles clusters are rotated in the results,
    # since we are comparing 2 we just need to compare that both clusters
    # have approximately the same number of points.
    if name == 'noisy_circles':
        assert (np.sum(sk_y_pred) - np.sum(cu_y_pred))/len(sk_y_pred) < 2e-3

    else:
        assert clusters_equal(sk_y_pred, cu_y_pred, params['n_clusters'])
