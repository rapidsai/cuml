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
from cuml.test.utils import get_handle
from cuml import DBSCAN as cuDBSCAN
from sklearn.cluster import DBSCAN as skDBSCAN
from sklearn.datasets.samples_generator import make_blobs
import pandas as pd
import cudf
import numpy as np
from sklearn.preprocessing import StandardScaler
from cuml.test.utils import fit_predict, get_pattern, clusters_equal


dataset_names = ['noisy_moons', 'varied', 'aniso', 'blobs',
                 'noisy_circles', 'no_structure']


@pytest.mark.parametrize('max_bytes_per_batch', [10, 200, 2e6])
@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
@pytest.mark.parametrize('use_handle', [True, False])


def test_dbscan_predict(datatype, input_type, use_handle, max_bytes_per_batch,
                        run_stress, run_quality):
    # max_bytes_per_batch sizes: 10=6 batches, 200=2 batches, 2e6=1 batch
    n_samples = 10000
    n_feats = 50
    if run_stress:
        X, y = make_blobs(n_samples=n_samples*50,
                          n_features=n_feats, random_state=0)
    elif run_quality:
        X, y = make_blobs(n_samples=n_samples,
                          n_features=n_feats, random_state=0)

    else:
        X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]],
                     dtype=datatype)
    skdbscan = skDBSCAN(eps=3, min_samples=10)
    sk_labels = skdbscan.fit_predict(X)
    handle, stream = get_handle(use_handle)
    cudbscan = cuDBSCAN(handle=handle, eps=3, min_samples=2,
                        max_bytes_per_batch=max_bytes_per_batch)

    if input_type == 'dataframe':
        X = pd.DataFrame(
            {'fea%d' % i: X[0:, i] for i in range(X.shape[1])})
        X_cudf = cudf.DataFrame.from_pandas(X)
        cu_labels = cudbscan.fit_predict(X_cudf)
    else:
        cu_labels = cudbscan.fit_predict(X)

    for i in range(X.shape[0]):
        assert cu_labels[i] == sk_labels[i]


@pytest.mark.parametrize("name", [
                                 'noisy_moons',
                                 'blobs',
                                 'no_structure'])
def test_dbscan_sklearn_comparison(name, run_stress, run_quality):
    default_base = {'quantile': .3,
                    'eps': .3,
                    'damping': .9,
                    'preference': -200,
                    'n_neighbors': 10,
                    'n_clusters': 20}
    n_samples = 10000
    if run_stress:
        pat = get_pattern(name, n_samples*50)
        params = default_base.copy()
        params.update(pat[1])
        X, y = pat[0]

    elif run_quality:
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

    dbscan = skDBSCAN(eps=params['eps'], min_samples=5)
    cuml_dbscan = cuDBSCAN(eps=params['eps'], min_samples=5)

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
