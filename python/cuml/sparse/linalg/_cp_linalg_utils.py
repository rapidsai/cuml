#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

import numpy as np
import cupy as cp
import cupyx.scipy.sparse

"""
A Set of utility methods that is called by the
`robust_lobpcg.py` implementation.
"""


def is_sparse(A):
    """Check if matrix A is a sparse matrix"""
    if isinstance(A, cupyx.scipy.sparse):
        return True
    return False


def get_floating_dtype(A):
    """Return the floating point dtype of tensor A.
    Integer types map to float32.
    """
    dtype = A.dtype
    if dtype in (np.float16, np.float32, np.float64):
        return dtype
    return cp.float32


def matmul(A, B):
    """Multiply two matrices.
    If A is None, return B. A can be sparse or dense. B is always
    dense.
    """
    if A is None:
        return B
    if cupyx.scipy.sparse.issparse(A):
        return A.multiply(B)
    return cp.matmul(A, B)


def transpose(A):
    ndim = len(A.shape)
    return A.transpose(ndim - 1, ndim - 2)


def qform(A, S):
    return matmul(transpose(S), matmul(A, S))


def basis(A):
    Q, _ = cp.linalg.qr(A)
    return Q


def symeig(A, largest=False):
    stride = 1
    if largest is None:
        largest = False
    assert len(A.shape) == 2, "only accepts 2D matrix!"
    E, Z = cp.linalg.eigh(A)
    idx = cp.argsort(E)
    if largest:
        stride = -1
    idx = idx[::stride]
    E = E[idx]
    Z = Z[:, idx]
    return E, Z
