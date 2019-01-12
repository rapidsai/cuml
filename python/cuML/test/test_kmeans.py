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

import cuml

import cudf
import numpy as np

from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler


def get_patterns():
    np.random.seed(0)

    n_samples = 1500
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                          noise=.05)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None

    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = datasets.make_blobs(n_samples=n_samples,
                                 cluster_std=[1.0, 2.5, 0.5],
                                 random_state=random_state)

    patterns = [
        (noisy_circles, {'damping': .77, 'preference': -240,
                         'quantile': .2, 'n_clusters': 2}),
        (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
        (varied, {'eps': .18, 'n_neighbors': 2}),
        (aniso, {'eps': .15, 'n_neighbors': 2}),
        (blobs, {}),
        (no_structure, {})]

    return patterns


def np2cudf(X):
    df = cudf.DataFrame()
    for i in range(X.shape[1]):
        df['fea%d' % i] = np.ascontiguousarray(X[:, i])
    return df


def fit_predict(algorithm, name, X):
    if name.startswith('sk'):
        algorithm.fit(X)
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)
    else:
        df = np2cudf(X)
        algorithm.fit(df)
        y_pred = algorithm.labels_.to_pandas().values.astype(np.int)
    return y_pred


@pytest.mark.parametrize('patterns', get_patterns())
def test_kmeans_sklearn(pattern):

    default_base = {'quantile': .3,
                    'eps': .3,
                    'damping': .9,
                    'preference': -200,
                    'n_neighbors': 10,
                    'n_clusters': 3}

    params = default_base.copy()
    params.update(pattern[1])

    kmeans = cluster.KMeans(n_clusters=params['n_clusters'])
    cuml_kmeans = cuml.KMeans(n_clusters=params['n_clusters'])

    X, y = pattern[0]

    X = StandardScaler().fit_transform(X)

    clustering_algorithms = (
        ('sk_Kmeans', kmeans),
        ('cuml_Kmeans', cuml_kmeans),
    )

    sk_y_pred = fit_predict(clustering_algorithms[0][1],
                            clustering_algorithms[0][0], X)

    cu_y_pred = fit_predict(clustering_algorithms[1][1],
                            clustering_algorithms[1][0], X)

    print(sk_y_pred[0:10])
    print(cu_y_pred[0:10])

    assert False
