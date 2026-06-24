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


def csr_row_normalize_l1(X, inplace=True):
    """Row normalize for csr matrix using the l1 norm"""
    if not inplace:
        X = X.copy()

    kernel = cuda_kernel_factory(
        """
        ({0} *data, {1} *indices, {2} *indptr, int n_samples) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;

            if (tid >= n_samples) return;
            {0} sum = 0.0;

            for (int i = indptr[tid]; i < indptr[tid+1]; i++) {
                sum += fabs(data[i]);
            }

            if (sum == 0) return;

            for (int i = indptr[tid]; i < indptr[tid+1]; i++) {
                data[i] /= sum;
            }
        }
        """,
        (X.dtype, X.indices.dtype, X.indptr.dtype),
        "csr_row_normalize_l1",
    )
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

    kernel = cuda_kernel_factory(
        """
        ({0} *data, {1} *indices, {2} *indptr, int n_samples) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;

            if (tid >= n_samples) return;
            {0} sum = 0.0;

            for (int i = indptr[tid]; i < indptr[tid+1]; i++) {
                sum += (data[i] * data[i]);
            }

            if (sum == 0) return;

            sum = sqrt(sum);

            for (int i = indptr[tid]; i < indptr[tid+1]; i++) {
                data[i] /= sum;
            }
        }
        """,
        (X.dtype, X.indices.dtype, X.indptr.dtype),
        "csr_row_normalize_l2",
    )
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
        out = X.T.dot(X).toarray()
    else:
        # Workaround for cupy #7699 (fixed in cupy 14.0).
        # Earlier versions of cupy may try to allocate large amounts for sparse
        # matrix multiplication (spGEMM).
        out = cp.zeros((X.shape[1], X.shape[1]), dtype=X.data.dtype)

        X = X.tocsr()
        compute_gram = cuda_kernel_factory(
            """
            (const int *indptr, const int *index, {0} *data, int m, int n, {0} *out) {
                int row = blockIdx.x;
                int col = threadIdx.x;

                if (row >= m) return;

                int start = indptr[row];
                int end = indptr[row + 1];

                for (int idx1 = start; idx1 < end; idx1++) {
                    int index1 = index[idx1];
                    {0} data1 = data[idx1];
                    for (int idx2 = idx1 + col; idx2 < end; idx2 += blockDim.x) {
                        int index2 = index[idx2];
                        {0} data2 = data[idx2];
                        atomicAdd(&out[index1 * n + index2], data1 * data2);
                    }
                }
            }
            """,
            (X.dtype,),
            "compute_gram",
        )
        compute_gram(
            (X.shape[0],),
            (128,),
            (
                X.indptr,
                X.indices,
                X.data,
                X.shape[0],
                X.shape[1],
                out,
            ),
        )

        reflect_diagonal = cuda_kernel_factory(
            """
            ({0} *out, int ncols) {
                int row = blockIdx.y * blockDim.y + threadIdx.y;
                int col = blockIdx.x * blockDim.x + threadIdx.x;

                if (row >= ncols || col >= ncols) return;

                if (row > col) {
                    out[row * ncols + col] = out[col * ncols + row];
                }
            }
            """,
            (X.dtype,),
            "reflect_diagonal",
        )
        reflect_diagonal(
            (math.ceil(X.shape[1] / 32),) * 2,
            (32, 32),
            (out, X.shape[1]),
        )

    mean_x = X.sum(axis=0) * (1 / X.shape[0])
    out *= 1 / X.shape[0]

    compute_cov = cuda_kernel_factory(
        """
        ({0} *out, {0} *mean_x, int n_cols) {

            int rid = blockDim.x * blockIdx.x + threadIdx.x;
            int cid = blockDim.y * blockIdx.y + threadIdx.y;

            if (rid >= n_cols || cid >= n_cols) return;

            out[rid * n_cols + cid] -= (mean_x[rid] * mean_x[cid]);
        }
        """,
        (X.dtype,),
        "subtract_mean_x_x",
    )
    compute_cov(
        (math.ceil(out.shape[0] / 32),) * 2,
        (32, 32),
        (out, mean_x, out.shape[0]),
    )
    return out, mean_x
