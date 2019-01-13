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


import numpy as np
import pandas as pd
from copy import deepcopy

from numbers import Number

from sklearn import datasets

import cudf


def array_equal(a, b, tol=1e-4, with_sign=True):
    a = to_nparray(a)
    b = to_nparray(b)
    if not with_sign:
        a, b = np.abs(a), np.abs(b)
    res = np.max(np.abs(a-b)) < tol
    return res


def to_nparray(x):
    if isinstance(x, Number):
        return np.array([x])
    elif isinstance(x, pd.DataFrame):
        return x.values
    elif isinstance(x, cudf.DataFrame):
        return x.to_pandas().values
    elif isinstance(x, cudf.Series):
        return x.to_pandas().values
    return np.array(x)


def get_pattern(name, n_samples):
    np.random.seed(0)
    random_state = 170

    if name == 'noisy_circles':
        data = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
        params = {'damping': .77, 'preference': -240,
                  'quantile': .2, 'n_clusters': 2}

    elif name == 'noisy_moons':
        data = datasets.make_moons(n_samples=n_samples, noise=.05)
        params = {'damping': .75, 'preference': -220, 'n_clusters': 2}

    elif name == 'varied':
        data = datasets.make_blobs(n_samples=n_samples,
                                   cluster_std=[1.0, 2.5, 0.5],
                                   random_state=random_state)
        params = {'eps': .18, 'n_neighbors': 2}

    elif name == 'blobs':
        data = datasets.make_blobs(n_samples=n_samples, random_state=8)
        params = {}

    elif name == 'aniso':
        X, y = datasets.make_blobs(n_samples=n_samples,
                                   random_state=random_state)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X_aniso = np.dot(X, transformation)
        data = (X_aniso, y)
        params = {'eps': .15, 'n_neighbors': 2}

    elif name == 'no_structure':
        data = np.random.rand(n_samples, 2), None
        params = {}

    return [data, params]


def np_to_cudf(X):
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
        df = np_to_cudf(X)
        algorithm.fit(df)
        y_pred = algorithm.labels_.to_pandas().values.astype(np.int)

    n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
    return y_pred, n_clusters


def normalize_clusters(a0, b0, n_clusters):
    a = to_nparray(a0)
    b = to_nparray(b0)

    c = deepcopy(b)

    for i in range(n_clusters):
        idx, = np.where(a == i)
        a_to_b = c[idx[0]]
        b[c == a_to_b] = i

    return a, b


def clusters_equal(a0, b0, n_clusters):
    a, b = normalize_clusters(a0, b0, n_clusters)
    return array_equal(a, b)
