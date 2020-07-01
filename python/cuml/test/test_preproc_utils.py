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
import numpy as np
from cupy.sparse import csr_matrix as gpu_csr_matrix
from cupy.sparse import csc_matrix as gpu_csc_matrix
from scipy.sparse import csr_matrix as cpu_csr_matrix
from scipy.sparse import csc_matrix as cpu_csc_matrix


clf_np, _ = make_classification(n_samples=500,
                                n_features=20,
                                n_clusters_per_class=1,
                                n_informative=12,
                                random_state=123, n_classes=5)


clf_sp_np = np.array(clf_np, copy=True)
clf_sp_np.ravel()[np.random.choice(clf_sp_np.size,
                                   int(clf_sp_np.size*0.1),
                                   replace=False)] = 0.

randint_sp_np = np.random.randint(100, size=(500, 20)).astype(np.float64)
randint_sp_np.ravel()[np.random.choice(randint_sp_np.size,
                                       int(randint_sp_np.size*0.1),
                                       replace=False)] = np.nan


@pytest.fixture(scope="session",
                params=["numpy", "dataframe", "cupy", "cudf", "numba"])
def small_clf_dataset(request):
    clf_conv = input_to_cuml_array(clf_np)[0]
    clf_conv = clf_conv.to_output(request.param)
    return clf_np, clf_conv


@pytest.fixture(scope="session",
                params=["numpy-csr", "numpy-csc", "cupy-csr", "cupy-csc"])
def small_sparse_dataset(request):
    if request.param == "numpy-csr":
        clf_sp_conv = cpu_csr_matrix(clf_sp_np)
    elif request.param == "numpy-csc":
        clf_sp_conv = cpu_csc_matrix(clf_sp_np)
    elif request.param == "cupy-csr":
        clf_sp_conv = cpu_csr_matrix(clf_sp_np)
        clf_sp_conv = gpu_csr_matrix(clf_sp_conv)
    elif request.param == "cupy-csc":
        clf_sp_conv = cpu_csc_matrix(clf_sp_np)
        clf_sp_conv = gpu_csc_matrix(clf_sp_conv)
    return clf_sp_np, clf_sp_conv


@pytest.fixture(scope="session",
                params=["numpy", "dataframe", "cupy", "cudf", "numba"])
def small_int_dataset(request):
    randint_sp_conv = input_to_cuml_array(randint_sp_np)[0]
    randint_sp_conv = randint_sp_conv.to_output(request.param)
    return randint_sp_np, randint_sp_conv
