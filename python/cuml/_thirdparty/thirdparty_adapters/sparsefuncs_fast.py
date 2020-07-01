
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


import cupy as cp
from numba import cuda
from math import ceil


def csr_mean_variance_axis0(X):
    X = X.tocsc()
    return csc_mean_variance_axis0(X)


def csc_mean_variance_axis0(X):
    n_features = X.shape[1]

    means = cp.zeros(n_features)
    variances = cp.zeros(n_features)
    counts_nan = cp.zeros(n_features)

    start = X.indptr[0]
    for i, end in enumerate(X.indptr[1:]):
        col = X.data[start:end]
        means[i] = col.mean()
        variances[i] = col.var()
        counts_nan[i] = X.nnz - cp.count_nonzero(cp.isnan(col))
        start = end
    return means, variances, counts_nan


def incr_mean_variance_axis0(X, last_mean, last_var, last_n):
    if isinstance(X, cp.sparse.csr_matrix):
        new_mean, new_var, counts_nan = csr_mean_variance_axis0(X)
    elif isinstance(X, cp.sparse.csc_matrix):
        new_mean, new_var, counts_nan = csc_mean_variance_axis0(X)

    new_n = cp.diff(X.indptr) - counts_nan

    n_features = X.shape[1]

    # First pass
    is_first_pass = True
    for i in range(n_features):
        if last_n[i] > 0:
            is_first_pass = False
            break
    if is_first_pass:
        return new_mean, new_var, new_n

    # Next passes
    for i in range(n_features):
        if new_n[i] > 0:
            updated_n[i] = last_n[i] + new_n[i]
            last_over_new_n[i] = dtype(last_n[i]) / dtype(new_n[i])
            # Unnormalized stats
            last_mean[i] *= last_n[i]
            last_var[i] *= last_n[i]
            new_mean[i] *= new_n[i]
            new_var[i] *= new_n[i]
            # Update stats
            updated_var[i] = (
                last_var[i] + new_var[i] +
                last_over_new_n[i] / updated_n[i] *
                (last_mean[i] / last_over_new_n[i] - new_mean[i])**2
            )
            updated_mean[i] = (last_mean[i] + new_mean[i]) / updated_n[i]
            updated_var[i] /= updated_n[i]
        else:
            updated_var[i] = last_var[i]
            updated_mean[i] = last_mean[i]
            updated_n[i] = last_n[i]

    return updated_mean, updated_var, updated_n


def inplace_csr_row_normalize_l1(X):
    start = X.indptr[0]
    for end in X.indptr[1:]:
        col = X.data[start:end]
        col = abs(col)
        sum_ = col.sum()
        X.data[start:end] /= sum_
        start = end


def inplace_csr_row_normalize_l2(X):
    start = X.indptr[0]
    for end in X.indptr[1:]:
        col = X.data[start:end]
        col = cp.square(col)
        sum_ = col.sum()
        X.data[start:end] /= cp.sqrt(sum_)
        start = end


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
        return int((3 * d**2 * i - 3 * d * i**2 + i**3
                   + 11 * i - 3 * j**2 - 9 * j) / 6
                   + i**2 - 2 * d * i + d * j - d + k)
    else:
        return int((3 * d**2 * i - 3 * d * i**2 + i ** 3 - i
                   - 3 * j**2 - 3 * j) / 6
                   + d * j + k)


@cuda.jit
def perform_expansion(indptr, indices, data, expanded_data,
                      expanded_indices, expanded_indptr,
                      nnz_cumsum, d, interaction_only, degree):
    row_i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    inrow_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if row_i >= indptr.shape[0]-1:
        return

    if inrow_idx == 0:
        expanded_indptr[row_i+1] = nnz_cumsum[row_i]

    row_starts = indptr[row_i]
    row_ends = indptr[row_i + 1]

    expanded_index = row_starts + inrow_idx

    if expanded_index > row_ends:
        return

    i_ptr = row_starts + interaction_only
    j_ptr = inrow_idx
    for i in range(i_ptr, row_ends):
        if degree == 2:
            diff = row_ends - i
        else:
            diff = 0
            for j in range(i + interaction_only, row_ends):
                diff += row_ends - j
        if j_ptr >= diff:
            j_ptr -= diff
            i_ptr += 1
        else:
            break

    j_ptr += row_starts

    i = indices[i_ptr]
    j = indices[j_ptr]

    if degree == 2:
        col = _deg2_column(d, i, j, interaction_only)
        expanded_indices[expanded_index] = col
        expanded_data[expanded_index] = data[i_ptr] * data[j_ptr]
    else:
        # degree == 3
        for k_ptr in range(j_ptr + interaction_only,
                           row_ends):
            k = indices[k_ptr]
            col = _deg3_column(d, i, j, k, interaction_only)
            expanded_indices[expanded_index] = col
            expanded_data[expanded_index] = data[i_ptr] * data[j_ptr] \
                * data[k_ptr]


def csr_polynomial_expansion(X, interaction_only, degree):
    assert degree in (2, 3)

    interaction_only = 1 if interaction_only else 0

    d = X.shape[1]
    if degree == 2:
        expanded_dimensionality = int((d**2 + d) / 2 - interaction_only*d)
    else:
        expanded_dimensionality = int((d**3 + 3*d**2 + 2*d) / 6
                                      - interaction_only*d**2)
    if expanded_dimensionality == 0:
        return None
    assert expanded_dimensionality > 0

    nnz = cp.diff(X.indptr)
    if degree == 2:
        total_nnz = (nnz ** 2 + nnz) / 2 - interaction_only * nnz
    else:
        total_nnz = ((nnz ** 3 + 3 * nnz ** 2 + 2 * nnz) / 6
                     - interaction_only * nnz ** 2)
    del nnz
    nnz_cumsum = total_nnz.cumsum(dtype=cp.int64)
    total_nnz_max = int(total_nnz.max())
    total_nnz = int(total_nnz.sum())

    num_rows = X.indptr.shape[0] - 1

    expanded_data = cp.zeros(shape=total_nnz, dtype=X.data.dtype)
    expanded_indices = cp.zeros(shape=total_nnz, dtype=X.indices.dtype)
    expanded_indptr = cp.zeros(shape=num_rows + 1, dtype=X.indptr.dtype)
    expanded_indptr[0] = X.indptr[0]

    tpb = (8, 8)
    bpg_x = ceil(X.indptr.shape[0] / tpb[0])
    bpg_y = ceil(total_nnz_max / tpb[1])
    bpg = (bpg_x, bpg_y)
    perform_expansion[bpg, tpb](X.indptr, X.indices, X.data,
                                expanded_data, expanded_indices,
                                expanded_indptr, nnz_cumsum,
                                d, interaction_only, degree)

    return cp.sparse.csr_matrix((expanded_data, expanded_indices,
                                 expanded_indptr),
                                shape=(num_rows, expanded_dimensionality))
