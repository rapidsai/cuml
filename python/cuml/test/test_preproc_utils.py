# Copyright (c) 2020, NVIDIA CORPORATION.
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

from cuml.common import input_to_cuml_array
from sklearn.datasets import make_classification
from cuml.thirdparty_adapters import to_output_type
from numpy.testing import assert_allclose as np_assert_allclose

import numpy as np
from cupy.sparse import csr_matrix as gpu_csr_matrix
from cupy.sparse import csc_matrix as gpu_csc_matrix
from scipy.sparse import csr_matrix as cpu_csr_matrix
from scipy.sparse import csc_matrix as cpu_csc_matrix


def create_rand_clf():
    clf, _ = make_classification(n_samples=500,
                                 n_features=20,
                                 n_clusters_per_class=1,
                                 n_informative=12,
                                 n_classes=5)
    clf = np.asfortranarray(clf)
    return clf


def create_rand_integers():
    randint = np.random.randint(30, size=(500, 20)).astype(np.float64)
    randint = np.asfortranarray(randint)
    return randint


def sparsify(dataset):
    random_loc = np.random.choice(dataset.size,
                                  int(dataset.size * 0.3),
                                  replace=False)
    dataset.ravel()[random_loc] = 0
    return cpu_csr_matrix(dataset)


@pytest.fixture(scope="session",
                params=["numpy", "dataframe", "cupy", "cudf", "numba"])
def clf_dataset(request):
    clf = create_rand_clf()
    cuml_array = input_to_cuml_array(clf)[0]
    converted_clf = cuml_array.to_output(request.param)
    return clf, converted_clf


@pytest.fixture(scope="session",
                params=["numpy", "dataframe", "cupy", "cudf", "numba"])
def int_dataset(request):
    randint = create_rand_integers()
    cuml_array = input_to_cuml_array(randint)[0]
    converted_randint = cuml_array.to_output(request.param)
    return randint, converted_randint


@pytest.fixture(scope="session",
                params=["numpy-csr", "numpy-csc", "cupy-csr", "cupy-csc"])
def sparse_clf_dataset(request):
    clf = create_rand_clf()
    clf = sparsify(clf)

    if request.param == "numpy-csr":
        converted_clf = cpu_csr_matrix(clf)
    elif request.param == "numpy-csc":
        converted_clf = cpu_csc_matrix(clf)
    elif request.param == "cupy-csr":
        converted_clf = gpu_csr_matrix(clf)
    elif request.param == "cupy-csc":
        converted_clf = gpu_csc_matrix(clf)
    return clf, converted_clf


@pytest.fixture(scope="session",
                params=["numpy-csr", "numpy-csc", "cupy-csr", "cupy-csc"])
def sparse_int_dataset(request):
    clf = create_rand_integers()
    clf = sparsify(clf)

    if request.param == "numpy-csr":
        converted_clf = cpu_csr_matrix(clf)
    elif request.param == "numpy-csc":
        converted_clf = cpu_csc_matrix(clf)
    elif request.param == "cupy-csr":
        converted_clf = gpu_csr_matrix(clf)
    elif request.param == "cupy-csc":
        converted_clf = gpu_csc_matrix(clf)
    return clf, converted_clf


def assert_allclose(actual, desired, rtol=1e-07, atol=0):
    if not isinstance(actual, np.ndarray):
        actual = to_output_type(actual, 'numpy')
    if not isinstance(desired, np.ndarray):
        desired = to_output_type(desired, 'numpy')
    return np_assert_allclose(actual, desired, rtol=rtol, atol=atol)
