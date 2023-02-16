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


from math import ceil
from cuml.internals.safe_imports import gpu_only_import_from
from cuml.internals.safe_imports import gpu_only_import

cp = gpu_only_import("cupy")
cpx = gpu_only_import("cupyx")
cuda = gpu_only_import_from("numba", "cuda")


def csr_mean_variance_axis0(X):
    """Compute mean and variance on the axis 0 of a CSR matrix

    Parameters
    ----------
    X : sparse CSR matrix
        Input array

    Returns
    -------
    mean and variance
    """
    X = X.tocsc()
    means, variances, _ = _csc_mean_variance_axis0(X)
    return means, variances


def csc_mean_variance_axis0(X):
    """Compute mean and variance on the axis 0 of a CSC matrix

    Parameters
    ----------
    X : sparse CSC matrix
        Input array

    Returns
    -------
    mean and variance
    """
    means, variances, _ = _csc_mean_variance_axis0(X)
    return means, variances


def _csc_mean_variance_axis0(X):
    """Compute mean, variance and nans count on the axis 0 of a CSC matrix

    Parameters
    ----------
    X : sparse CSC matrix
        Input array

    Returns
    -------
    mean, variance, nans count
    """
    n_samples, n_features = X.shape

    means = cp.empty(n_features)
    variances = cp.empty(n_features)
    counts_nan = cp.empty(n_features)

    start = X.indptr[0]
    for i, end in enumerate(X.indptr[1:]):
        col = X.data[start:end]

        _count_zeros = n_samples - col.size
        _count_nans = (col != col).sum()

        _mean = cp.nansum(col) / (n_samples - _count_nans)
        _variance = cp.nansum((col - _mean) ** 2)
        _variance += _count_zeros * (_mean**2)
        _variance /= n_samples - _count_nans

        means[i] = _mean
        variances[i] = _variance
        counts_nan[i] = _count_nans

        start = end
    return means, variances, counts_nan


@cuda.jit
def norm_step2_k(indptr, data, norm):
    """Apply normalization

    Parameters
    ----------
    indptr : array
        indptr of sparse matrix
    data : array
        data of sparse matrix
    norm: array
        norm by which to divide columns
    """
    row_i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    inrow_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if row_i >= indptr.shape[0] - 1:
        return

    start = indptr[row_i]
    end = indptr[row_i + 1]
    if inrow_idx >= (end - start):
        return

    data[start + inrow_idx] /= norm[row_i]


@cuda.jit
def l1_step1_k(indptr, data, norm):
    """Compute norm for L1 normalization"""
    row_i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    inrow_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if row_i >= indptr.shape[0] - 1:
        return

    start = indptr[row_i]
    end = indptr[row_i + 1]
    if inrow_idx >= (end - start):
        return

    val = abs(data[start + inrow_idx])
    cuda.atomic.add(norm, row_i, val)


def inplace_csr_row_normalize_l1(X):
    """Normalize CSR matrix inplace with L1 norm

    Parameters
    ----------
    X : sparse CSR matrix
        Input array

    Returns
    -------
    Normalized matrix
    """
    n_rows = X.indptr.shape[0]
    max_nnz = cp.diff(X.indptr).max()
    tpb = (32, 32)
    bpg_x = ceil(n_rows / tpb[0])
    bpg_y = ceil(max_nnz / tpb[1])
    bpg = (bpg_x, bpg_y)

    norm = cp.zeros(n_rows - 1, dtype=X.dtype)
    l1_step1_k[bpg, tpb](X.indptr, X.data, norm)
    norm_step2_k[bpg, tpb](X.indptr, X.data, norm)


@cuda.jit
def l2_step1_k(indptr, data, norm):
    """Compute norm for L2 normalization"""
    row_i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    inrow_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if row_i >= indptr.shape[0] - 1:
        return

    start = indptr[row_i]
    end = indptr[row_i + 1]
    if inrow_idx >= (end - start):
        return

    val = data[start + inrow_idx]
    val *= val
    cuda.atomic.add(norm, row_i, val)


def inplace_csr_row_normalize_l2(X):
    """Normalize CSR matrix inplace with L2 norm

    Parameters
    ----------
    X : sparse CSR matrix
        Input array

    Returns
    -------
    Normalized matrix
    """
    n_rows = X.indptr.shape[0]
    max_nnz = cp.diff(X.indptr).max()
    tpb = (32, 32)
    bpg_x = ceil(n_rows / tpb[0])
    bpg_y = ceil(max_nnz / tpb[1])
    bpg = (bpg_x, bpg_y)

    norm = cp.zeros(n_rows - 1, dtype=X.dtype)
    l2_step1_k[bpg, tpb](X.indptr, X.data, norm)
    norm = cp.sqrt(norm)
    norm_step2_k[bpg, tpb](X.indptr, X.data, norm)


@cuda.jit(device=True, inline=True)
def _deg2_column(d, i, j, interaction_only):
    """Compute the index of the column for a degree 2 expansion

    d is the dimensionality of the input data, i and j are the indices
    for the columns involved in the expansion.
    """
    if interaction_only:
        return int(d * i - (i**2 + 3 * i) / 2 - 1 + j)
    else:
        return int(d * i - (i**2 + i) / 2 + j)


@cuda.jit(device=True, inline=True)
def _deg3_column(d, i, j, k, interaction_only):
    """Compute the index of the column for a degree 3 expansion

    d is the dimensionality of the input data, i, j and k are the indices
    for the columns involved in the expansion.
    """
    if interaction_only:
        return int(
            (
                3 * d**2 * i
                - 3 * d * i**2
                + i**3
                + 11 * i
                - 3 * j**2
                - 9 * j
            )
            / 6
            + i**2
            - 2 * d * i
            + d * j
            - d
            + k
        )
    else:
        return int(
            (3 * d**2 * i - 3 * d * i**2 + i**3 - i - 3 * j**2 - 3 * j)
            / 6
            + d * j
            + k
        )


@cuda.jit
def perform_expansion(
    indptr,
    indices,
    data,
    expanded_data,
    expanded_indices,
    d,
    interaction_only,
    degree,
    expanded_indptr,
):
    """Kernel applying polynomial expansion on CSR matrix"""
    row_i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    inrow_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if row_i >= indptr.shape[0] - 1:
        return

    expanded_index = expanded_indptr[row_i] + inrow_idx
    if expanded_index >= expanded_indptr[row_i + 1]:
        return

    row_starts = indptr[row_i]
    row_ends = indptr[row_i + 1]

    i_ptr = row_starts
    j_ptr = -1
    k_ptr = inrow_idx

    if degree == 2:
        j_ptr = inrow_idx
        for i in range(row_starts, row_ends):
            diff = row_ends - i - interaction_only
            if j_ptr >= diff:
                j_ptr -= diff
            else:
                i_ptr = i
                break
        j_ptr += i_ptr + interaction_only
    else:
        # degree == 3
        diff = 0
        for i in range(row_starts, row_ends):
            for j in range(i + interaction_only, row_ends):
                diff = row_ends - j - interaction_only
                if k_ptr >= diff:
                    k_ptr -= diff
                else:
                    j_ptr = j
                    i_ptr = i
                    break
            if j_ptr != -1:
                break

        k_ptr += j_ptr + interaction_only

    i = indices[i_ptr]
    j = indices[j_ptr]

    if degree == 2:
        col = _deg2_column(d, i, j, interaction_only)
        expanded_indices[expanded_index] = col
        expanded_data[expanded_index] = data[i_ptr] * data[j_ptr]
    else:
        # degree == 3
        k = indices[k_ptr]
        col = _deg3_column(d, i, j, k, interaction_only)
        expanded_indices[expanded_index] = col
        expanded_data[expanded_index] = data[i_ptr] * data[j_ptr] * data[k_ptr]


def csr_polynomial_expansion(X, interaction_only, degree):
    """Apply polynomial expansion on CSR matrix

    Parameters
    ----------
    X : sparse CSR matrix
        Input array

    Returns
    -------
    New expansed matrix
    """
    assert degree in (2, 3)

    interaction_only = 1 if interaction_only else 0

    d = X.shape[1]
    if degree == 2:
        expanded_dimensionality = int((d**2 + d) / 2 - interaction_only * d)
    else:
        expanded_dimensionality = int(
            (d**3 + 3 * d**2 + 2 * d) / 6 - interaction_only * d**2
        )
    if expanded_dimensionality == 0:
        return None
    assert expanded_dimensionality > 0

    nnz = cp.diff(X.indptr)
    if degree == 2:
        total_nnz = (nnz**2 + nnz) / 2 - interaction_only * nnz
    else:
        total_nnz = (
            nnz**3 + 3 * nnz**2 + 2 * nnz
        ) / 6 - interaction_only * nnz**2
    del nnz
    nnz_cumsum = total_nnz.cumsum(dtype=cp.int64)
    total_nnz_max = int(total_nnz.max())
    total_nnz = int(total_nnz.sum())

    num_rows = X.indptr.shape[0] - 1

    expanded_data = cp.empty(shape=total_nnz, dtype=X.data.dtype)
    expanded_indices = cp.empty(shape=total_nnz, dtype=X.indices.dtype)
    expanded_indptr = cp.empty(shape=num_rows + 1, dtype=X.indptr.dtype)
    expanded_indptr[0] = X.indptr[0]
    expanded_indptr[1:] = nnz_cumsum

    tpb = (32, 32)
    bpg_x = ceil(X.indptr.shape[0] / tpb[0])
    bpg_y = ceil(total_nnz_max / tpb[1])
    bpg = (bpg_x, bpg_y)
    perform_expansion[bpg, tpb](
        X.indptr,
        X.indices,
        X.data,
        expanded_data,
        expanded_indices,
        d,
        interaction_only,
        degree,
        expanded_indptr,
    )

    return cpx.scipy.sparse.csr_matrix(
        (expanded_data, expanded_indices, expanded_indptr),
        shape=(num_rows, expanded_dimensionality),
    )
