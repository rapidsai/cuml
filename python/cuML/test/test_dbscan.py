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
from cuml import DBSCAN as cuDBSCAN
from sklearn.cluster import DBSCAN as skDBSCAN
import cudf
import numpy as np

from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler

from utils import fit_predict, get_pattern, clusters_equal

dataset_names = ['noisy_moons', 'varied', 'aniso', 'blobs', 'noisy_circles',
                 'no_structure']


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
def test_dbscan_predict(datatype, input_type):

    X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]],
                 dtype=datatype)
    skdbscan = skDBSCAN(eps=3, min_samples=2)
    sk_labels = skdbscan.fit_predict(X)

    cudbscan = cuDBSCAN(eps=3, min_samples=2)

    if input_type == 'dataframe':
        gdf = cudf.DataFrame()
        gdf['0'] = np.asarray([1, 2, 2, 8, 8, 25], dtype=datatype)
        gdf['1'] = np.asarray([2, 2, 3, 7, 8, 80], dtype=datatype)
        cu_labels = cudbscan.fit_predict(gdf)
    else:
        cu_labels = cudbscan.fit_predict(X)

    for i in range(X.shape[0]):
        assert cu_labels[i] == sk_labels[i]


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
def test_dbscan_predict_numpy(datatype):
    gdf = cudf.DataFrame()
    gdf['0'] = np.asarray([1, 2, 2, 8, 8, 25], dtype=datatype)
    gdf['1'] = np.asarray([2, 2, 3, 7, 8, 80], dtype=datatype)

    X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]],
                 dtype=datatype)

    print("Calling fit_predict")
    cudbscan = cuDBSCAN(eps=3, min_samples=2)
    cu_labels = cudbscan.fit_predict(gdf)
    skdbscan = skDBSCAN(eps=3, min_samples=2)
    sk_labels = skdbscan.fit_predict(X)
    print(X.shape[0])
    for i in range(X.shape[0]):
        assert cu_labels[i] == sk_labels[i]


@pytest.mark.parametrize("name", [
                                 'noisy_moons',
                                 'blobs',
                                 'no_structure'])
def test_dbscan_sklearn_comparison(name):

    # Skipping datasets of known discrepancies in PR83 while they are corrected
    default_base = {'quantile': .3,
                    'eps': .3,
                    'damping': .9,
                    'preference': -200,
                    'n_neighbors': 10,
                    'n_clusters': 3}

    pat = get_pattern(name, 1500)

    params = default_base.copy()
    params.update(pat[1])

    dbscan = skDBSCAN(eps=params['eps'], min_samples=5)
    cuml_dbscan = cuDBSCAN(eps=params['eps'], min_samples=5)

    X, y = pat[0]

    X = StandardScaler().fit_transform(X)

    clustering_algorithms = (
        ('sk_DBSCAN', dbscan),
        ('cuml_DBSCAN', cuml_dbscan)
    )

    sk_y_pred, sk_n_clusters = fit_predict(clustering_algorithms[0][1],
                                           clustering_algorithms[0][0], X)

    cu_y_pred, cu_n_clusters = fit_predict(clustering_algorithms[1][1],
                                           clustering_algorithms[1][0], X)

    assert(sk_n_clusters == cu_n_clusters)

    clusters_equal(sk_y_pred, cu_y_pred, sk_n_clusters)
