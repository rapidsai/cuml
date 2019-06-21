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


def unit_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.unit)


def quality_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.quality)


def stress_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.stress)


@pytest.mark.parametrize('name', dataset_names)
@pytest.mark.parametrize('nrows', [unit_param(20), quality_param(5000),
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

    cu_y_pred, _ = fit_predict(cuml_kmeans,
                               'cuml_Kmeans', X)

    if nrows < 500000:
        kmeans = cluster.KMeans(n_clusters=params['n_clusters'])
        sk_y_pred, _ = fit_predict(kmeans,
                                   'sk_Kmeans', X)

        # Noisy circles clusters are rotated in the results,
        # since we are comparing 2 we just need to compare that both clusters
        # have approximately the same number of points.
        calculation = (np.sum(sk_y_pred) - np.sum(cu_y_pred))/len(sk_y_pred)
        print(cuml_kmeans.score(X), kmeans.score(X))
        score_test = (cuml_kmeans.score(X) + kmeans.score(X)) < 2e-3
        if name == 'noisy_circles':
            assert (calculation < 2e-3) and score_test

        else:
            assert (clusters_equal(sk_y_pred, cu_y_pred, params['n_clusters'])) and score_test
