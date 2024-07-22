#
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

from cuml.internals.import_utils import has_scipy
from cuml.internals.safe_imports import gpu_only_import

cupyx = gpu_only_import("cupyx")

if has_scipy():
    import scipy.sparse


def is_sparse(X):
    """
    Return true if X is sparse, false otherwise.
    Parameters
    ----------
    X : array-like, sparse-matrix

    Returns
    -------

    is_sparse : boolean
        is the input sparse?
    """
    is_scipy_sparse = has_scipy() and scipy.sparse.isspmatrix(X)
    return cupyx.scipy.sparse.isspmatrix(X) or is_scipy_sparse


def is_dense(X):
    return not is_sparse(X)
