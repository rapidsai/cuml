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
from numba.cuda.cudadrv.devicearray import DeviceNDArray

from sklearn import datasets
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

import cudf
import cuml
import pytest


def array_equal(a, b, unit_tol=1e-4, total_tol=1e-4, with_sign=True):
    """
    Utility function to compare 2 numpy arrays. Two individual elements
    are assumed equal if they are within `unit_tol` of each other, and two
    arrays are considered equal if less than `total_tol` percentage of
    elements are different.

    """

    a = to_nparray(a)
    b = to_nparray(b)

    if len(a) == 0 and len(b) == 0:
        return True

    if not with_sign:
        a, b = np.abs(a), np.abs(b)
    res = (np.sum(np.abs(a-b) > unit_tol)) / len(a) < total_tol
    return res


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


def normalize_clusters(a0, b0, n_clusters):
    a = to_nparray(a0)
    b = to_nparray(b0)

    c = deepcopy(b)

    for i in range(n_clusters):
        idx, = np.where(a == i)
        a_to_b = c[idx[0]]
        b[c == a_to_b] = i

    return a, b


def to_nparray(x):
    if isinstance(x, Number):
        return np.array([x])
    elif isinstance(x, pd.DataFrame):
        return x.values
    elif isinstance(x, cudf.DataFrame):
        return x.to_pandas().values
    elif isinstance(x, cudf.Series):
        return x.to_pandas().values
    elif isinstance(x, DeviceNDArray):
        return x.copy_to_host()
    return np.array(x)


def clusters_equal(a0, b0, n_clusters, tol=1e-4):
    a, b = normalize_clusters(a0, b0, n_clusters)
    return array_equal(a, b, total_tol=tol)


def get_handle(use_handle, n_streams=0):
    if not use_handle:
        return None, None
    h = cuml.Handle(n_streams)
    s = cuml.cuda.Stream()
    h.setStream(s)
    return h, s


def small_regression_dataset(datatype):
    X, y = make_regression(n_samples=500, n_features=20,
                           n_informative=10, random_state=10)
    X = X.astype(datatype)
    y = y.astype(datatype)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=0)

    return X_train, X_test, y_train, y_test


def small_classification_dataset(datatype):
    X, y = make_classification(n_samples=500, n_features=20,
                               n_informative=10, n_classes=2,
                               random_state=10)
    X = X.astype(datatype)
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=0)

    return X_train, X_test, y_train, y_test


def unit_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.unit)


def quality_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.quality)


def stress_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.stress)
