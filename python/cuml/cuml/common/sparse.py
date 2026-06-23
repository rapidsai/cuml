#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import math

import cupy as cp
import cupyx.scipy.sparse as cp_sp
import scipy.sparse as sp
from packaging.version import Version

from cuml.common.kernel_utils import cuda_kernel_factory

CUPY_VERSION = Version(cp.__version__)


__all__ = (
    "is_sparse",
    "csr_row_normalize_l1",
    "csr_row_normalize_l2",
    "csr_diag_mul",
    "sparse_cov_and_mean",
)


def is_sparse(X):
    """Returns whether X is sparse matrix.

    Parameters
    ----------
    X : array-like, sparse-matrix

    Returns
    -------
    is_sparse : bool
        Whether the input is sparse.
    """
    return sp.issparse(X) or cp_sp.issparse(X)


def _map_l1_norm_kernel(dtype):
    """Creates cupy RawKernel for csr_raw_normalize_l1 function."""

    map_kernel_str = r"""
    ({0} *data, {1} *indices, {2} *indptr, int n_samples) {

      int tid = blockDim.x * blockIdx.x + threadIdx.x;

      if(tid >= n_samples) return;
      {0} sum = 0.0;


      for(int i = indptr[tid]; i < indptr[tid+1]; i++) {
        sum += fabs(data[i]);
      }


      if(sum == 0) return;

      for(int i = indptr[tid]; i < indptr[tid+1]; i++) {
        data[i] /= sum;
      }
    }
    """
    return cuda_kernel_factory(map_kernel_str, dtype, "map_l1_norm_kernel")


def _map_l2_norm_kernel(dtype):
    """Creates cupy RawKernel for csr_raw_normalize_l2 function."""

    map_kernel_str = r"""
    ({0} *data, {1} *indices, {2} *indptr, int n_samples) {

      int tid = blockDim.x * blockIdx.x + threadIdx.x;

      if(tid >= n_samples) return;
      {0} sum = 0.0;

      for(int i = indptr[tid]; i < indptr[tid+1]; i++) {
        sum += (data[i] * data[i]);
      }

      if(sum == 0) return;

      sum = sqrt(sum);

      for(int i = indptr[tid]; i < indptr[tid+1]; i++) {
        data[i] /= sum;
      }
    }
    """
    return cuda_kernel_factory(map_kernel_str, dtype, "map_l2_norm_kernel")


def csr_row_normalize_l1(X, inplace=True):
    """Row normalize for csr matrix using the l1 norm"""
    if not inplace:
        X = X.copy()

    kernel = _map_l1_norm_kernel((X.dtype, X.indices.dtype, X.indptr.dtype))
    kernel(
        (math.ceil(X.shape[0] / 32),),
        (32,),
        (X.data, X.indices, X.indptr, X.shape[0]),
    )

    return X


def csr_row_normalize_l2(X, inplace=True):
    """Row normalize for csr matrix using the l2 norm"""
    if not inplace:
        X = X.copy()

    kernel = _map_l2_norm_kernel((X.dtype, X.indices.dtype, X.indptr.dtype))
    kernel(
        (math.ceil(X.shape[0] / 32),),
        (32,),
        (X.data, X.indices, X.indptr, X.shape[0]),
    )

    return X


def csr_diag_mul(X, y, inplace=True):
    """Multiply a sparse X matrix with diagonal matrix y"""
    if not inplace:
        X = X.copy()
    # grab underlying dense ar from y
    y = y.data[0]
    X.data *= y[X.indices]
    return X


_cov_kernel = r"""
({0} *cov_values, {0} *gram_matrix, {0} *mean_x, int n_cols) {

    int rid = blockDim.x * blockIdx.x + threadIdx.x;
    int cid = blockDim.y * blockIdx.y + threadIdx.y;

    if (rid >= n_cols || cid >= n_cols) return;

    cov_values[rid * n_cols + cid] =
        gram_matrix[rid * n_cols + cid] - mean_x[rid] * mean_x[cid];
}
"""

_gram_kernel = r"""
(const int *indptr, const int *index, {0} *data, int nrows, int ncols, {0} *out) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row >= nrows) return;

    int start = indptr[row];
    int end = indptr[row + 1];

    for (int idx1 = start; idx1 < end; idx1++) {
        int index1 = index[idx1];
        {0} data1 = data[idx1];
        for (int idx2 = idx1 + col; idx2 < end; idx2 += blockDim.x) {
            int index2 = index[idx2];
            {0} data2 = data[idx2];
            atomicAdd(&out[index1 * ncols + index2], data1 * data2);
        }
    }
}
"""

_copy_kernel = r"""
({0} *out, int ncols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= ncols || col >= ncols) return;

    if (row > col) {
        out[row * ncols + col] = out[col * ncols + row];
    }
}
"""


def sparse_cov_and_mean(X, use_dot=CUPY_VERSION >= Version("14.0.0")):
    """Computes the covariance and mean of X.

    Parameters
    ----------
    X : cupyx.scipy.sparse, shape=(n_samples, n_features)
        The input data.
    use_dot : bool
        Whether to rely on `dot` for computing the grammian matrix.
        Defaults to True for cupy >= 14 and False otherwise.

    Returns
    -------
    covariance: cp.ndarray, shape=(n_features, n_features)
        The covariance of X, in the form of E(XX) - E(X)E(X).
    mean: cp.ndarray, shape=(n_features,)
        The mean of X, equivalent to X.mean(axis=0).
    """
    if use_dot:
        gram_matrix = X.T.dot(X).toarray()
    else:
        # Workaround for cupy #7699 (fixed in cupy 14.0).
        # Earlier versions of cupy may try to allocate large amounts for sparse
        # matrix multiplication (spGEMM).
        gram_matrix = cp.zeros((X.shape[1], X.shape[1]), dtype=X.data.dtype)

        X = X.tocsr()
        block = (128,)
        grid = (X.shape[0],)
        compute_gram = cuda_kernel_factory(
            _gram_kernel, (X.dtype,), "gram_kernel"
        )
        compute_gram(
            grid,
            block,
            (
                X.indptr,
                X.indices,
                X.data,
                X.shape[0],
                X.shape[1],
                gram_matrix,
            ),
        )

        copy_gram = cuda_kernel_factory(
            _copy_kernel, (X.dtype,), "copy_kernel"
        )
        block = (32, 32)
        grid = (math.ceil(X.shape[1] / 32),) * 2
        copy_gram(
            grid,
            block,
            (gram_matrix, X.shape[1]),
        )

    mean_x = X.sum(axis=0) * (1 / X.shape[0])
    gram_matrix *= 1 / X.shape[0]
    cov_result = gram_matrix

    compute_cov = cuda_kernel_factory(_cov_kernel, (X.dtype,), "cov_kernel")
    compute_cov(
        (math.ceil(gram_matrix.shape[0] / 32),) * 2,
        (32, 32),
        (cov_result, gram_matrix, mean_x, gram_matrix.shape[0]),
    )
    return cov_result, mean_x
