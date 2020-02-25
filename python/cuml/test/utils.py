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
import inspect

import cupy as cp
import numpy as np
import pandas as pd
from copy import deepcopy

from numba import cuda
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
        return np.asarray([x])
    elif isinstance(x, pd.DataFrame):
        return x.values
    elif isinstance(x, cudf.DataFrame):
        return x.to_pandas().values
    elif isinstance(x, cudf.Series):
        return x.to_pandas().values
    elif isinstance(x, DeviceNDArray):
        return x.copy_to_host()
    elif isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


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


class ClassEnumerator:
    """Helper class to automatically pick up every models classes in a module.
    Filters out classes not inheriting from cuml.Base.

    Parameters
    ----------
    module: python module (ex: cuml.linear_regression)
        The module for which to retrieve models.
    exclude_classes: list of classes (optional)
        Those classes will be filtered out from the retrieved models.
    custom_constructors: dictionary of {class_name: lambda}
        Custom constructors to use instead of the default one.
        ex: {'LogisticRegression': lambda: cuml.LogisticRegression(handle=1)}
    """
    def __init__(self, module, exclude_classes=None, custom_constructors=None):
        self.module = module
        self.exclude_classes = exclude_classes or []
        self.custom_constructors = custom_constructors or []

    def _get_classes(self):
        return inspect.getmembers(self.module, inspect.isclass)

    def get_models(self):
        """Picks up every models classes from self.module.
        Filters out classes not inheriting from cuml.Base.

        Returns
        -------
        models: dictionary of {class_name: class|class_constructor}
            Dictionary of models in the module, except when a
            custom_constructor is specified, in that case the value is the
            specified custom_constructor.
        """
        classes = self._get_classes()
        models = {name: cls for name, cls in classes
                  if cls not in self.exclude_classes and
                  issubclass(cls, cuml.Base)}
        models.update(self.custom_constructors)
        return models


def get_classes_from_package(package):
    modules = [m for name, m in inspect.getmembers(package, inspect.ismodule)]
    classes = [ClassEnumerator(module).get_models() for module in modules]
    return {k: v for dictionary in classes for k, v in dictionary.items()}


def generate_random_labels(random_generation_lambda, seed=1234):
    rng = np.random.RandomState(seed)  # makes it reproducible
    a = random_generation_lambda(rng)
    b = random_generation_lambda(rng)

    return cuda.to_device(a), cuda.to_device(b)
