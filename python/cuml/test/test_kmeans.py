# Copyright (c) 2018, NVIDIA CORPORATION.
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

dataset_names = ['noisy_moons', 'varied', 'aniso', 'noisy_circles', 'blobs']


@pytest.mark.parametrize('name', dataset_names)
def test_kmeans_sklearn_comparison(name, run_stress, run_correctness_test):

    default_base = {'quantile': .3,
                    'eps': .3,
                    'damping': .9,
                    'preference': -200,
                    'n_neighbors': 10,
                    'n_clusters': 20}
    n_samples = 10000
    if run_stress == True:
        pat = get_pattern(name, n_samples*50)
        params = default_base.copy()
        params.update(pat[1])
        X, y = pat[0]

    elif run_correctness_test==True:
        pat = get_pattern(name, n_samples)
        params = default_base.copy()
        params.update(pat[1])
        X, y = pat[0] 

    else:
        pat = get_pattern(name, np.int32(n_samples/2))
        params = default_base.copy()
        params.update(pat[1])
        X, y = pat[0]

    X = StandardScaler().fit_transform(X)
    kmeans = cluster.KMeans(n_clusters=params['n_clusters'])
    cuml_kmeans = cuml.KMeans(n_clusters=params['n_clusters'])

    clustering_algorithms = (
        ('sk_Kmeans', kmeans),
        ('cuml_Kmeans', cuml_kmeans),
    )

    sk_y_pred, _ = fit_predict(clustering_algorithms[0][1],
                               clustering_algorithms[0][0], X)

    kmeans.fit(X)
    cu_y_pred, _ = fit_predict(clustering_algorithms[1][1],
                               clustering_algorithms[1][0], X)

    if name == 'noisy_circles':
        assert (np.sum(sk_y_pred) - np.sum(cu_y_pred))/len(sk_y_pred) < 1e-10

    else:
        clusters_equal(sk_y_pred, cu_y_pred, params['n_clusters'])
