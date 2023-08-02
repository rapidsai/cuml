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

from cuml.common.kernel_utils import cuda_kernel_factory
import cuml.internals
import math
from cuml.internals.safe_imports import gpu_only_import

cp = gpu_only_import("cupy")
cupyx = gpu_only_import("cupyx")


cov_kernel_str = r"""
({0} *cov_values, {0} *gram_matrix, {0} *mean_x, {0} *mean_y, int n_cols) {

    int rid = blockDim.x * blockIdx.x + threadIdx.x;
    int cid = blockDim.y * blockIdx.y + threadIdx.y;

    if(rid >= n_cols || cid >= n_cols) return;

    cov_values[rid * n_cols + cid] = \
        gram_matrix[rid * n_cols + cid] - mean_x[rid] * mean_y[cid];
}
"""

gramm_kernel_csr = r"""
(const int *indptr, const int *index, {0} *data, int nrows, int ncols, {0} *out) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if(row >= nrows) return;

    int start = indptr[row];
    int end = indptr[row + 1];

    for (int idx1 = start; idx1 < end; idx1++){
        int index1 = index[idx1];
        {0} data1 = data[idx1];
        for(int idx2 = idx1 + col; idx2 < end; idx2 += blockDim.x){
            int index2 = index[idx2];
            {0} data2 = data[idx2];
            atomicAdd(&out[index1 * ncols + index2], data1 * data2);
        }
    }
}
"""

gramm_kernel_coo = r"""
(const int *rows, const int *cols, {0} *data, int nnz, int ncols, int nrows, {0} * out) {
    int i = blockIdx.x;
    if (i >= nnz) return;
    int row1 = rows[i];
    int col1 = cols[i];
    {0} data1 = data[i];
    int limit = min(i + nrows, nnz);

    for(int j = i + threadIdx.x; j < limit; j += blockDim.x){
        if(row1 < rows[j]) return;

        if(col1 <= cols[j]){
            atomicAdd(&out[col1 * ncols + cols[j]], data1 * data[j]);
        }
    }
}
"""

copy_kernel = r"""
({0} *out, int ncols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= ncols || col >= ncols) return;

    if (row > col) {
        out[row * ncols + col] = out[col * ncols + row];
    }
}
"""


def _cov_kernel(dtype):
    return cuda_kernel_factory(cov_kernel_str, (dtype,), "cov_kernel")


def _gramm_kernel_csr(dtype):
    return cuda_kernel_factory(gramm_kernel_csr, (dtype,), "gramm_kernel_csr")


def _gramm_kernel_coo(dtype):
    return cuda_kernel_factory(gramm_kernel_coo, (dtype,), "gramm_kernel_coo")


def _copy_kernel(dtype):
    return cuda_kernel_factory(copy_kernel, (dtype,), "copy_kernel")


@cuml.internals.api_return_any()
def cov(x, y, mean_x=None, mean_y=None, return_gram=False, return_mean=False):
    """
    Computes a covariance between two matrices using
    the form Cov(X, Y) = E(XY) - E(X)E(Y)

    This function prevents the need to explicitly
    compute the outer product E(X)E(Y) by taking
    advantage of the symmetry of that matrix
    and computes per element in a kernel.

    When E(XY) is approximately equal to E(X)E(Y),
    this method is prone to catastrophic cancellation.
    In such cases, a rectangular solver should be
    preferred.

    Parameters
    ----------

    x : device-array or cupyx.scipy.sparse of size (m, n)
    y : device-array or cupyx.scipy.sparse of size (m, n)
    mean_x : float (default = None)
        device-array of size (n, ) which is the mean
        of x across rows
    mean_x : float (default = None)
        device-array of size (n, ) which is the mean
        of x across rows
    return_gram : boolean (default = False)
        If True, gram matrix of the form (1 / n) * X.T.dot(Y)
        will be returned.
        When True, a copy will be created
        to store the results of the covariance.
        When False, the local gram matrix result
        will be overwritten
    return_mean: boolean (default = False)
        If True, the Maximum Likelihood Estimate used to
        calculate the mean of X and Y will be returned,
        of the form (1 / n) * mean(X) and (1 / n) * mean(Y)

    Returns
    -------

    result : cov(X, Y) when return_gram and return_mean are False
             cov(X, Y), gram(X, Y) when return_gram is True,
                return_mean is False
             cov(X, Y), mean(X), mean(Y) when return_gram is False,
                return_mean is True
             cov(X, Y), gram(X, Y), mean(X), mean(Y)
                when return_gram is True and return_mean is True
    """

    if x.dtype != y.dtype:
        raise ValueError(
            "X and Y must have same dtype (%s != %s)" % (x.dtype, y.dtype)
        )

    if x.shape != y.shape:
        raise ValueError(
            "X and Y must have same shape %s != %s" % (x.shape, y.shape)
        )

    # Fix for cuml issue #5475 & cupy issue #7699
    # addressing problems with sparse matrix multiplication (spGEMM)
    if (
        x is y
        and cupyx.scipy.sparse.issparse(x)
        and mean_x is None
        and mean_y is None
    ):
        return _cov_sparse(x, return_gram=return_gram, return_mean=return_mean)

    if mean_x is not None and mean_y is not None:
        if mean_x.dtype != mean_y.dtype:
            raise ValueError(
                "Mean of X and Mean of Y must have same dtype"
                "(%s != %s)" % (mean_x.dtype, mean_y.dtype)
            )

        if mean_x.shape != mean_y.shape:
            raise ValueError(
                "Mean of X and Mean of Y must have same shape"
                "%s != %s" % (mean_x.shape, mean_y.shape)
            )

    gram_matrix = x.T.dot(y) * (1 / x.shape[0])

    if cupyx.scipy.sparse.issparse(gram_matrix):
        gram_matrix = gram_matrix.todense()

    if mean_x is None:
        mean_x = x.sum(axis=0) * (1 / x.shape[0])

    if mean_y is None:
        mean_y = y.sum(axis=0) * (1 / y.shape[0])

    if return_gram:
        cov_result = cp.zeros(
            (gram_matrix.shape[0], gram_matrix.shape[1]),
            dtype=gram_matrix.dtype,
        )
    else:
        cov_result = gram_matrix

    compute_cov = _cov_kernel(x.dtype)

    block_size = (32, 32)
    grid_size = (
        math.ceil(gram_matrix.shape[0] / 32),
        math.ceil(gram_matrix.shape[1] / 32),
    )

    compute_cov(
        grid_size,
        block_size,
        (cov_result, gram_matrix, mean_x, mean_y, gram_matrix.shape[0]),
    )

    if not return_gram and not return_mean:
        return cov_result
    elif return_gram and not return_mean:
        return cov_result, gram_matrix
    elif not return_gram and return_mean:
        return cov_result, mean_x, mean_y
    elif return_gram and return_mean:
        return cov_result, gram_matrix, mean_x, mean_y


@cuml.internals.api_return_any()
def _cov_sparse(x, return_gram=False, return_mean=False):
    """
    Computes the mean and the covariance of matrix X of
    the form Cov(X, X) = E(XX) - E(X)E(X)

    This is a temporary fix for
    cuml issue #5475 and cupy issue #7699,
    where the operation `x.T.dot(x)` did not work for
    larger sparse matrices.

    Parameters
    ----------

    x : cupyx.scipy.sparse of size (m, n)
    return_gram : boolean (default = False)
        If True, gram matrix of the form (1 / n) * X.T.dot(X)
        will be returned.
        When True, a copy will be created
        to store the results of the covariance.
        When False, the local gram matrix result
        will be overwritten
    return_mean: boolean (default = False)
        If True, the Maximum Likelihood Estimate used to
        calculate the mean of X and X will be returned,
        of the form (1 / n) * mean(X) and (1 / n) * mean(X)

    Returns
    -------

    result : cov(X, X) when return_gram and return_mean are False
            cov(X, X), gram(X, X) when return_gram is True,
            return_mean is False
            cov(X, X), mean(X), mean(X) when return_gram is False,
            return_mean is True
            cov(X, X), gram(X, X), mean(X), mean(X)
            when return_gram is True and return_mean is True
    """

    gram_matrix = cp.zeros((x.shape[1], x.shape[1]), dtype=x.data.dtype)
    if cupyx.scipy.sparse.isspmatrix_csr(x):
        block = (128,)
        grid = (x.shape[0],)
        compute_mean_cov = _gramm_kernel_csr(x.data.dtype)
        compute_mean_cov(
            grid,
            block,
            (
                x.indptr,
                x.indices,
                x.data,
                x.shape[0],
                x.shape[1],
                gram_matrix,
            ),
        )

    elif cupyx.scipy.sparse.isspmatrix_coo(x):
        x.sum_duplicates()
        nnz = len(x.row)
        block = (128,)
        grid = (nnz,)
        compute_gram_coo = _gramm_kernel_coo(x.data.dtype)
        compute_gram_coo(
            grid,
            block,
            (x.row, x.col, x.data, nnz, x.shape[1], x.shape[0], gram_matrix),
        )

    else:
        x = x.tocsr()
        block = (128,)
        grid = (math.ceil(x.shape[0] / block[0]),)
        compute_mean_cov = _gramm_kernel_csr(x.data.dtype)
        compute_mean_cov(
            grid,
            block,
            (
                x.indptr,
                x.indices,
                x.data,
                x.shape[0],
                x.shape[1],
                gram_matrix,
            ),
        )

    copy_gram = _copy_kernel(x.data.dtype)
    block = (32, 32)
    grid = (math.ceil(x.shape[1] / block[0]), math.ceil(x.shape[1] / block[1]))
    copy_gram(
        grid,
        block,
        (gram_matrix, x.shape[1]),
    )

    mean_x = x.sum(axis=0) * (1 / x.shape[0])
    gram_matrix *= 1 / x.shape[0]

    if return_gram:
        cov_result = cp.zeros(
            (gram_matrix.shape[0], gram_matrix.shape[0]),
            dtype=gram_matrix.dtype,
        )
    else:
        cov_result = gram_matrix

    compute_cov = _cov_kernel(x.dtype)

    block_size = (32, 32)
    grid_size = (math.ceil(gram_matrix.shape[0] / 8),) * 2
    compute_cov(
        grid_size,
        block_size,
        (cov_result, gram_matrix, mean_x, mean_x, gram_matrix.shape[0]),
    )

    if not return_gram and not return_mean:
        return cov_result
    elif return_gram and not return_mean:
        return cov_result, gram_matrix
    elif not return_gram and return_mean:
        return cov_result, mean_x, mean_x
    elif return_gram and return_mean:
        return cov_result, gram_matrix, mean_x, mean_x
