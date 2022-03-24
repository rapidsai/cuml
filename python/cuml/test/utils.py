# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

from raft.common.cuda import Stream

from sklearn import datasets
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

import cudf
import cuml
from cuml.common.input_utils import input_to_cuml_array
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
    res = (np.sum(np.abs(a - b) > unit_tol)) / a.size < total_tol
    return res


def get_pattern(name, n_samples):
    np.random.seed(0)
    random_state = 170

    if name == 'noisy_circles':
        data = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
        params = {
            'damping': .77,
            'preference': -240,
            'quantile': .2,
            'n_clusters': 2
        }

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


def as_type(type, *args):
    # Convert array args to type supported by
    # CumlArray.to_output ('numpy','cudf','cupy'...)
    # Ensure 2 dimensional inputs are not converted to 1 dimension
    # None remains as None
    # Scalar remains a scalar
    result = []
    for arg in args:
        if arg is None or np.isscalar(arg):
            result.append(arg)
        else:
            # make sure X with a single feature remains 2 dimensional
            if type == 'cudf' and len(arg.shape) > 1:
                result.append(input_to_cuml_array(
                    arg).array.to_output('dataframe'))
            else:
                result.append(input_to_cuml_array(arg).array.to_output(type))
    if len(result) == 1:
        return result[0]
    return tuple(result)


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


def assert_dbscan_equal(ref, actual, X, core_indices, eps):
    """
    Utility function to compare two numpy label arrays.
    The labels of core/noise points are expected to be equal, and the labels
    of border points are verified by finding a neighboring core point with the
    same label.
    """
    core_set = set(core_indices)
    N, _ = X.shape
    eps2 = eps**2

    def sqnorm(x):
        return np.inner(x, x)

    for i in range(N):
        la, lb = ref[i], actual[i]

        if i in core_set:  # core point
            assert la == lb, ("Core point mismatch at #{}: "
                              "{} (expected {})".format(i, lb, la))
        elif la == -1:  # noise point
            assert lb == -1, "Noise mislabelled at #{}: {}".format(i, lb)
        else:  # border point
            found = False
            for j in range(N):
                # Check if j is a core point with the same label
                if j in core_set and lb == actual[j]:
                    # Check if j is a neighbor of i
                    if sqnorm(X[i] - X[j]) <= eps2:
                        found = True
                        break
            assert found, ("Border point not connected to cluster at #{}: "
                           "{} (reference: {})".format(i, lb, la))

    # Note: we can also do it in a rand score fashion by checking that pairs
    # correspond in both label arrays for core points, if we need to drop the
    # requirement of minimality for core points


def get_handle(use_handle, n_streams=0):
    if not use_handle:
        return None, None
    s = Stream()
    h = cuml.Handle(stream=s, n_streams=n_streams)
    return h, s


def small_regression_dataset(datatype):
    X, y = make_regression(n_samples=1000, n_features=20,
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
    recursive: bool, default=False
        Instructs the class to recursively search submodules when True,
        otherwise only classes in the specified model will be enumerated
    """
    def __init__(self,
                 module,
                 exclude_classes=None,
                 custom_constructors=None,
                 recursive=False):
        self.module = module
        self.exclude_classes = exclude_classes or []
        self.custom_constructors = custom_constructors or []
        self.recursive = recursive

    def _get_classes(self):
        def recurse_module(module):
            classes = {}

            modules = []

            if (self.recursive):
                modules = inspect.getmembers(module, inspect.ismodule)

            # Enumerate child modules only if they are a submodule of the
            # current one. i.e. `{parent_module}.{submodule}`
            for _, m in modules:
                if (module.__name__ + "." in m.__name__):
                    classes.update(recurse_module(m))

            # Ensure we only get classes that are part of this module
            classes.update({
                (".".join((klass.__module__, klass.__qualname__))): klass
                for name,
                klass in inspect.getmembers(module, inspect.isclass)
                if module.__name__ + "." in ".".join((klass.__module__,
                                                      klass.__qualname__))
            })

            return classes

        return [(val.__name__, val) for key,
                val in recurse_module(self.module).items()]

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
        models = {
            name: cls
            for name,
            cls in classes
            if cls not in self.exclude_classes and issubclass(cls, cuml.Base)
        }
        models.update(self.custom_constructors)
        return models


def get_classes_from_package(package, import_sub_packages=False):
    """
    Gets all modules imported in the specified package and returns a dictionary
    of any classes that derive from `cuml.Base`

    Parameters
    ----------
    package : python module The python module to search import_sub_packages :
        bool, default=False When set to True, will try to import sub packages
        by searching the directory tree for __init__.py files and importing
        them accordingly. By default this is set to False

    Returns
    -------
    ClassEnumerator Class enumerator for the specified package
    """

    if (import_sub_packages):
        import os
        import importlib

        # First, find all __init__.py files in subdirectories of this package
        root_dir = os.path.dirname(package.__file__)

        root_relative = os.path.dirname(root_dir)

        # Now loop
        for root, _, files in os.walk(root_dir):

            if "__init__.py" in files:

                module_name = os.path.relpath(root, root_relative).replace(
                    os.sep, ".")

                importlib.import_module(module_name)

    return ClassEnumerator(module=package, recursive=True).get_models()


def generate_random_labels(random_generation_lambda, seed=1234, as_cupy=False):
    """
    Generates random labels to act as ground_truth and predictions for tests.

    Parameters
    ----------
    random_generation_lambda : lambda function [numpy.random] -> ndarray
        A lambda function used to generate labels for either y_true or y_pred
        using a seeded numpy.random object.
    seed : int
        Seed for the numpy.random object.
    as_cupy : bool
        Choose return type of y_true and y_pred.
        True: returns Cupy ndarray
        False: returns Numba cuda DeviceNDArray

    Returns
    -------
    y_true, y_pred, np_y_true, np_y_pred : tuple
        y_true : Numba cuda DeviceNDArray or Cupy ndarray
            Random target values.
        y_pred : Numba cuda DeviceNDArray or Cupy ndarray
            Random predictions.
        np_y_true : Numpy ndarray
            Same as y_true but as a numpy ndarray.
        np_y_pred : Numpy ndarray
            Same as y_pred but as a numpy ndarray.
    """
    rng = np.random.RandomState(seed)  # makes it reproducible
    a = random_generation_lambda(rng)
    b = random_generation_lambda(rng)

    if as_cupy:
        return cp.array(a), cp.array(b), a, b
    else:
        return cuda.to_device(a), cuda.to_device(b), a, b


def score_labeling_with_handle(func,
                               ground_truth,
                               predictions,
                               use_handle,
                               dtype=np.int32):
    """Test helper to standardize inputs between sklearn and our prims metrics.

    Using this function we can pass python lists as input of a test just like
    with sklearn as well as an option to use handle with our metrics.
    """
    a = cp.array(ground_truth, dtype=dtype)
    b = cp.array(predictions, dtype=dtype)

    handle, stream = get_handle(use_handle)

    return func(a, b, handle=handle)


def get_number_positional_args(func, default=2):
    # function to return number of positional arguments in func
    if hasattr(func, "__code__"):
        all_args = func.__code__.co_argcount
        if func.__defaults__ is not None:
            kwargs = len(func.__defaults__)
        else:
            kwargs = 0
        return all_args - kwargs
    return default


def get_shap_values(model,
                    explainer,
                    background_dataset,
                    explained_dataset,
                    api_type='shap_values'):
    # function to get shap values from an explainer using SHAP style API.
    # This function allows isolating all calls in test suite for the case of
    # API changes.
    explainer = explainer(
        model=model,
        data=background_dataset
    )
    if api_type == 'shap_values':
        shap_values = explainer.shap_values(explained_dataset)
    elif api_type == '__call__':
        shap_values = explainer(explained_dataset)

    return explainer, shap_values
