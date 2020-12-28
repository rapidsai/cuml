/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuml/common/logger.hpp>

#include <cusparse_v2.h>
#include <raft/cudart_utils.h>
#include <raft/sparse/cusparse_wrappers.h>
#include <raft/cuda_utils.cuh>

#include <label/classlabels.cuh>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cuda_runtime.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>

#include <sparse/utils.h>

namespace MLCommon {
namespace Sparse {

static const float MIN_FLOAT = std::numeric_limits<float>::min();

template <typename value_t>
__global__ void csr_to_dense_block_per_row_kernel(int n_cols,
                                                  const value_t *csrVal,
                                                  const int *csrRowPtr,
                                                  const int *csrColInd,
                                                  value_t *a) {
  int row = blockIdx.x;
  int tid = threadIdx.x;

  int colStart = csrRowPtr[row];
  int colEnd = csrRowPtr[row + 1];
  int rowNnz = colEnd - colStart;

  for (int i = tid; i < rowNnz; i += blockDim.x) {
    int colIdx = colStart + i;
    if (colIdx < colEnd) {
      int col = csrColInd[colIdx];
      a[row * n_cols + col] = csrVal[colIdx];
    }
  }
}

/**
 * Convert CSR arrays to a dense matrix in either row-
 * or column-major format. A custom kernel is used when
 * row-major output is desired since cusparse does not
 * output row-major.
 * @tparam value_idx : data type of the CSR index arrays
 * @tparam value_t : data type of the CSR value array
 * @param[in] handle : cusparse handle for conversion
 * @param[in] nrows : number of rows in CSR
 * @param[in] ncols : number of columns in CSR
 * @param[in] csr_indptr : CSR row index pointer array
 * @param[in] csr_indices : CSR column indices array
 * @param[in] csr_data : CSR data array
 * @param[in] lda : Leading dimension (used for col-major only)
 * @param[out] out : Dense output array of size nrows * ncols
 * @param[in] stream : Cuda stream for ordering events
 * @param[in] row_major : Is row-major output desired?
 */
template <typename value_idx, typename value_t>
void csr_to_dense(cusparseHandle_t handle, value_idx nrows, value_idx ncols,
                  const value_idx *csr_indptr, const value_idx *csr_indices,
                  const value_t *csr_data, value_idx lda, value_t *out,
                  cudaStream_t stream, bool row_major = true) {
  if (!row_major) {
    /**
     * If we need col-major, use cusparse.
     */
    cusparseMatDescr_t out_mat;
    CUSPARSE_CHECK(cusparseCreateMatDescr(&out_mat));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(out_mat, CUSPARSE_INDEX_BASE_ZERO));
    CUSPARSE_CHECK(cusparseSetMatType(out_mat, CUSPARSE_MATRIX_TYPE_GENERAL));

    CUSPARSE_CHECK(raft::sparse::cusparsecsr2dense(
      handle, nrows, ncols, out_mat, csr_data, csr_indptr, csr_indices, out,
      lda, stream));

    CUSPARSE_CHECK_NO_THROW(cusparseDestroyMatDescr(out_mat));

  } else {
    int blockdim = block_dim(ncols);
    CUDA_CHECK(
      cudaMemsetAsync(out, 0, nrows * ncols * sizeof(value_t), stream));
    csr_to_dense_block_per_row_kernel<<<nrows, blockdim, 0, stream>>>(
      ncols, csr_data, csr_indptr, csr_indices, out);
  }
}

/**
 * Transpose a set of CSR arrays into a set of CSC arrays.
 * @tparam value_idx : data type of the CSR index arrays
 * @tparam value_t : data type of the CSR data array
 * @param[in] handle : used for invoking cusparse
 * @param[in] csr_indptr : CSR row index array
 * @param[in] csr_indices : CSR column indices array
 * @param[in] csr_data : CSR data array
 * @param[out] csc_indptr : CSC row index array
 * @param[out] csc_indices : CSC column indices array
 * @param[out] csc_data : CSC data array
 * @param[in] csr_nrows : Number of rows in CSR
 * @param[in] csr_ncols : Number of columns in CSR
 * @param[in] nnz : Number of nonzeros of CSR
 * @param[in] allocator : Allocator for intermediate memory
 * @param[in] stream : Cuda stream for ordering events
 */
template <typename value_idx, typename value_t>
void csr_transpose(cusparseHandle_t handle, const value_idx *csr_indptr,
                   const value_idx *csr_indices, const value_t *csr_data,
                   value_idx *csc_indptr, value_idx *csc_indices,
                   value_t *csc_data, value_idx csr_nrows, value_idx csr_ncols,
                   value_idx nnz, std::shared_ptr<deviceAllocator> allocator,
                   cudaStream_t stream) {
  size_t convert_csc_workspace_size = 0;

  CUSPARSE_CHECK(raft::sparse::cusparsecsr2csc_bufferSize(
    handle, csr_nrows, csr_ncols, nnz, csr_data, csr_indptr, csr_indices,
    csc_data, csc_indptr, csc_indices, CUSPARSE_ACTION_NUMERIC,
    CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1,
    &convert_csc_workspace_size, stream));

  CUML_LOG_DEBUG("Transpose workspace size: %d", convert_csc_workspace_size);

  device_buffer<char> convert_csc_workspace(allocator, stream,
                                            convert_csc_workspace_size);

  CUSPARSE_CHECK(raft::sparse::cusparsecsr2csc(
    handle, csr_nrows, csr_ncols, nnz, csr_data, csr_indptr, csr_indices,
    csc_data, csc_indptr, csc_indices, CUSPARSE_ACTION_NUMERIC,
    CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1,
    convert_csc_workspace.data(), stream));
}

/**
 * Slice consecutive rows from a CSR array and populate newly sliced indptr array
 * @tparam value_idx
 * @param[in] start_row : beginning row to slice
 * @param[in] stop_row : ending row to slice
 * @param[in] indptr : indptr of input CSR to slice
 * @param[out] indptr_out : output sliced indptr to populate
 * @param[in] start_offset : beginning column offset of input indptr
 * @param[in] stop_offset : ending column offset of input indptr
 * @param[in] stream : cuda stream for ordering events
 */
template <typename value_idx>
void csr_row_slice_indptr(value_idx start_row, value_idx stop_row,
                          const value_idx *indptr, value_idx *indptr_out,
                          value_idx *start_offset, value_idx *stop_offset,
                          cudaStream_t stream) {
  raft::update_host(start_offset, indptr + start_row, 1, stream);
  raft::update_host(stop_offset, indptr + stop_row + 1, 1, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  value_idx s_offset = *start_offset;

  // 0-based indexing so we need to add 1 to stop row. Because we want n_rows+1, we add another 1 to stop row.
  raft::copy_async(indptr_out, indptr + start_row, (stop_row + 2) - start_row,
                   stream);

  raft::linalg::unaryOp<value_idx>(
    indptr_out, indptr_out, (stop_row + 2) - start_row,
    [s_offset] __device__(value_idx input) { return input - s_offset; },
    stream);
}

/**
 * Slice rows from a CSR, populate column and data arrays
 * @tparam[in] value_idx : data type of CSR index arrays
 * @tparam[in] value_t : data type of CSR data array
 * @param[in] start_offset : beginning column offset to slice
 * @param[in] stop_offset : ending column offset to slice
 * @param[in] indices : column indices array from input CSR
 * @param[in] data : data array from input CSR
 * @param[out] indices_out : output column indices array
 * @param[out] data_out : output data array
 * @param[in] stream : cuda stream for ordering events
 */
template <typename value_idx, typename value_t>
void csr_row_slice_populate(value_idx start_offset, value_idx stop_offset,
                            const value_idx *indices, const value_t *data,
                            value_idx *indices_out, value_t *data_out,
                            cudaStream_t stream) {
  raft::copy(indices_out, indices + start_offset, stop_offset - start_offset,
             stream);
  raft::copy(data_out, data + start_offset, stop_offset - start_offset, stream);
}

template <int TPB_X, typename T>
__global__ void csr_row_normalize_l1_kernel(
  // @TODO: This can be done much more parallel by
  // having threads in a warp compute the sum in parallel
  // over each row and then divide the values in parallel.
  const int *ia,           // csr row ex_scan (sorted by row)
  const T *vals, int nnz,  // array of values and number of non-zeros
  int m,                   // num rows in csr
  T *result) {             // output array

  // row-based matrix 1 thread per row
  int row = (blockIdx.x * TPB_X) + threadIdx.x;

  // sum all vals_arr for row and divide each val by sum
  if (row < m) {
    int start_idx = ia[row];
    int stop_idx = 0;
    if (row < m - 1) {
      stop_idx = ia[row + 1];
    } else
      stop_idx = nnz;

    T sum = T(0.0);
    for (int j = start_idx; j < stop_idx; j++) {
      sum = sum + fabs(vals[j]);
    }

    for (int j = start_idx; j < stop_idx; j++) {
      if (sum != 0.0) {
        T val = vals[j];
        result[j] = val / sum;
      } else {
        result[j] = 0.0;
      }
    }
  }
}

/**
 * @brief Perform L1 normalization on the rows of a given CSR-formatted sparse matrix
 *
 * @param ia: row_ind array
 * @param vals: data array
 * @param nnz: size of data array
 * @param m: size of row_ind array
 * @param result: l1 normalized data array
 * @param stream: cuda stream to use
 */
template <int TPB_X = 32, typename T>
void csr_row_normalize_l1(const int *ia,  // csr row ex_scan (sorted by row)
                          const T *vals,
                          int nnz,  // array of values and number of non-zeros
                          int m,    // num rows in csr
                          T *result,
                          cudaStream_t stream) {  // output array

  dim3 grid(raft::ceildiv(m, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  csr_row_normalize_l1_kernel<TPB_X, T>
    <<<grid, blk, 0, stream>>>(ia, vals, nnz, m, result);
  CUDA_CHECK(cudaGetLastError());
}

template <int TPB_X = 32, typename T>
__global__ void csr_row_normalize_max_kernel(
  // @TODO: This can be done much more parallel by
  // having threads in a warp compute the sum in parallel
  // over each row and then divide the values in parallel.
  const int *ia,           // csr row ind array (sorted by row)
  const T *vals, int nnz,  // array of values and number of non-zeros
  int m,                   // num total rows in csr
  T *result) {             // output array

  // row-based matrix 1 thread per row
  int row = (blockIdx.x * TPB_X) + threadIdx.x;

  // find max across columns and divide
  if (row < m) {
    int start_idx = ia[row];
    int stop_idx = 0;
    if (row < m - 1) {
      stop_idx = ia[row + 1];
    } else
      stop_idx = nnz;

    T max = MIN_FLOAT;
    for (int j = start_idx; j < stop_idx; j++) {
      if (vals[j] > max) max = vals[j];
    }

    // divide nonzeros in current row by max
    for (int j = start_idx; j < stop_idx; j++) {
      if (max != 0.0 && max > MIN_FLOAT) {
        T val = vals[j];
        result[j] = val / max;
      } else {
        result[j] = 0.0;
      }
    }
  }
}

/**
 * @brief Perform L_inf normalization on a given CSR-formatted sparse matrix
 *
 * @param ia: row_ind array
 * @param vals: data array
 * @param nnz: size of data array
 * @param m: size of row_ind array
 * @param result: l1 normalized data array
 * @param stream: cuda stream to use
 */

template <int TPB_X = 32, typename T>
void csr_row_normalize_max(const int *ia,  // csr row ind array (sorted by row)
                           const T *vals,
                           int nnz,  // array of values and number of non-zeros
                           int m,    // num total rows in csr
                           T *result, cudaStream_t stream) {
  dim3 grid(raft::ceildiv(m, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  csr_row_normalize_max_kernel<TPB_X, T>
    <<<grid, blk, 0, stream>>>(ia, vals, nnz, m, result);
  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__device__ int get_stop_idx(T row, T m, T nnz, const T *ind) {
  int stop_idx = 0;
  if (row < (m - 1))
    stop_idx = ind[row + 1];
  else
    stop_idx = nnz;

  return stop_idx;
}

template <typename value_idx = int, int TPB_X = 32>
__global__ void csr_to_coo_kernel(const value_idx *row_ind, value_idx m,
                                  value_idx *coo_rows, value_idx nnz) {
  // row-based matrix 1 thread per row
  value_idx row = (blockIdx.x * TPB_X) + threadIdx.x;
  if (row < m) {
    value_idx start_idx = row_ind[row];
    value_idx stop_idx = get_stop_idx(row, m, nnz, row_ind);
    for (value_idx i = start_idx; i < stop_idx; i++) coo_rows[i] = row;
  }
}

/**
 * @brief Convert a CSR row_ind array to a COO rows array
 * @param row_ind: Input CSR row_ind array
 * @param m: size of row_ind array
 * @param coo_rows: Output COO row array
 * @param nnz: size of output COO row array
 * @param stream: cuda stream to use
 */
template <typename value_idx = int, int TPB_X = 32>
void csr_to_coo(const value_idx *row_ind, value_idx m, value_idx *coo_rows,
                value_idx nnz, cudaStream_t stream) {
  // @TODO: Use cusparse for this.
  dim3 grid(raft::ceildiv(m, (value_idx)TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  csr_to_coo_kernel<value_idx, TPB_X>
    <<<grid, blk, 0, stream>>>(row_ind, m, coo_rows, nnz);

  CUDA_CHECK(cudaGetLastError());
}

template <typename T, int TPB_X = 32>
__global__ void csr_add_calc_row_counts_kernel(
  const int *a_ind, const int *a_indptr, const T *a_val, int nnz1,
  const int *b_ind, const int *b_indptr, const T *b_val, int nnz2, int m,
  int *out_rowcounts) {
  // loop through columns in each set of rows and
  // calculate number of unique cols across both rows
  int row = (blockIdx.x * TPB_X) + threadIdx.x;

  if (row < m) {
    int a_start_idx = a_ind[row];
    int a_stop_idx = get_stop_idx(row, m, nnz1, a_ind);

    int b_start_idx = b_ind[row];
    int b_stop_idx = get_stop_idx(row, m, nnz2, b_ind);

    /**
         * Union of columns within each row of A and B so that we can scan through
         * them, adding their values together.
         */
    int max_size = (a_stop_idx - a_start_idx) + (b_stop_idx - b_start_idx);

    int *arr = new int[max_size];
    int cur_arr_idx = 0;
    for (int j = a_start_idx; j < a_stop_idx; j++) {
      arr[cur_arr_idx] = a_indptr[j];
      cur_arr_idx++;
    }

    int arr_size = cur_arr_idx;
    int final_size = arr_size;

    for (int j = b_start_idx; j < b_stop_idx; j++) {
      int cur_col = b_indptr[j];
      bool found = false;
      for (int k = 0; k < arr_size; k++) {
        if (arr[k] == cur_col) {
          found = true;
          break;
        }
      }

      if (!found) {
        final_size++;
      }
    }

    out_rowcounts[row] = final_size;
    raft::myAtomicAdd(out_rowcounts + m, final_size);

    delete arr;
  }
}

template <typename T, int TPB_X = 32>
__global__ void csr_add_kernel(const int *a_ind, const int *a_indptr,
                               const T *a_val, int nnz1, const int *b_ind,
                               const int *b_indptr, const T *b_val, int nnz2,
                               int m, int *out_ind, int *out_indptr,
                               T *out_val) {
  // 1 thread per row
  int row = (blockIdx.x * TPB_X) + threadIdx.x;

  if (row < m) {
    int a_start_idx = a_ind[row];
    int a_stop_idx = get_stop_idx(row, m, nnz1, a_ind);

    int b_start_idx = b_ind[row];
    int b_stop_idx = get_stop_idx(row, m, nnz2, b_ind);

    int o_idx = out_ind[row];

    int cur_o_idx = o_idx;
    for (int j = a_start_idx; j < a_stop_idx; j++) {
      out_indptr[cur_o_idx] = a_indptr[j];
      out_val[cur_o_idx] = a_val[j];
      cur_o_idx++;
    }

    int arr_size = cur_o_idx - o_idx;
    for (int j = b_start_idx; j < b_stop_idx; j++) {
      int cur_col = b_indptr[j];
      bool found = false;
      for (int k = o_idx; k < o_idx + arr_size; k++) {
        // If we found a match, sum the two values
        if (out_indptr[k] == cur_col) {
          out_val[k] += b_val[j];
          found = true;
          break;
        }
      }

      // if we didn't find a match, add the value for b
      if (!found) {
        out_indptr[o_idx + arr_size] = cur_col;
        out_val[o_idx + arr_size] = b_val[j];
        arr_size++;
      }
    }
  }
}

/**
 * @brief Calculate the CSR row_ind array that would result
 * from summing together two CSR matrices
 * @param a_ind: left hand row_ind array
 * @param a_indptr: left hand index_ptr array
 * @param a_val: left hand data array
 * @param nnz1: size of left hand index_ptr and val arrays
 * @param b_ind: right hand row_ind array
 * @param b_indptr: right hand index_ptr array
 * @param b_val: right hand data array
 * @param nnz2: size of right hand index_ptr and val arrays
 * @param m: size of output array (number of rows in final matrix)
 * @param out_ind: output row_ind array
 * @param d_alloc: deviceAllocator to use for temp memory
 * @param stream: cuda stream to use
 */
template <typename T, int TPB_X = 32>
size_t csr_add_calc_inds(const int *a_ind, const int *a_indptr, const T *a_val,
                         int nnz1, const int *b_ind, const int *b_indptr,
                         const T *b_val, int nnz2, int m, int *out_ind,
                         std::shared_ptr<deviceAllocator> d_alloc,
                         cudaStream_t stream) {
  dim3 grid(raft::ceildiv(m, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  device_buffer<int> row_counts(d_alloc, stream, m + 1);
  CUDA_CHECK(
    cudaMemsetAsync(row_counts.data(), 0, (m + 1) * sizeof(int), stream));

  csr_add_calc_row_counts_kernel<T, TPB_X>
    <<<grid, blk, 0, stream>>>(a_ind, a_indptr, a_val, nnz1, b_ind, b_indptr,
                               b_val, nnz2, m, row_counts.data());

  int cnnz = 0;
  raft::update_host(&cnnz, row_counts.data() + m, 1, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // create csr compressed row index from row counts
  thrust::device_ptr<int> row_counts_d =
    thrust::device_pointer_cast(row_counts.data());
  thrust::device_ptr<int> c_ind_d = thrust::device_pointer_cast(out_ind);
  exclusive_scan(thrust::cuda::par.on(stream), row_counts_d, row_counts_d + m,
                 c_ind_d);

  return cnnz;
}

/**
 * @brief Calculate the CSR row_ind array that would result
 * from summing together two CSR matrices
 * @param a_ind: left hand row_ind array
 * @param a_indptr: left hand index_ptr array
 * @param a_val: left hand data array
 * @param nnz1: size of left hand index_ptr and val arrays
 * @param b_ind: right hand row_ind array
 * @param b_indptr: right hand index_ptr array
 * @param b_val: right hand data array
 * @param nnz2: size of right hand index_ptr and val arrays
 * @param m: size of output array (number of rows in final matrix)
 * @param c_ind: output row_ind array
 * @param c_indptr: output ind_ptr array
 * @param c_val: output data array
 * @param stream: cuda stream to use
 */
template <typename T, int TPB_X = 32>
void csr_add_finalize(const int *a_ind, const int *a_indptr, const T *a_val,
                      int nnz1, const int *b_ind, const int *b_indptr,
                      const T *b_val, int nnz2, int m, int *c_ind,
                      int *c_indptr, T *c_val, cudaStream_t stream) {
  dim3 grid(raft::ceildiv(m, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  csr_add_kernel<T, TPB_X>
    <<<grid, blk, 0, stream>>>(a_ind, a_indptr, a_val, nnz1, b_ind, b_indptr,
                               b_val, nnz2, m, c_ind, c_indptr, c_val);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T, int TPB_X = 32, typename Lambda = auto(T, T, T)->void>
__global__ void csr_row_op_kernel(const T *row_ind, T n_rows, T nnz,
                                  Lambda op) {
  T row = blockIdx.x * TPB_X + threadIdx.x;
  if (row < n_rows) {
    T start_idx = row_ind[row];
    T stop_idx = row < n_rows - 1 ? row_ind[row + 1] : nnz;
    op(row, start_idx, stop_idx);
  }
}

/**
 * @brief Perform a custom row operation on a CSR matrix in batches.
 * @tparam T numerical type of row_ind array
 * @tparam TPB_X number of threads per block to use for underlying kernel
 * @tparam Lambda type of custom operation function
 * @param row_ind the CSR row_ind array to perform parallel operations over
 * @param n_rows total number vertices in graph
 * @param nnz number of non-zeros
 * @param op custom row operation functor accepting the row and beginning index.
 * @param stream cuda stream to use
 */
template <typename Index_, int TPB_X = 32,
          typename Lambda = auto(Index_, Index_, Index_)->void>
void csr_row_op(const Index_ *row_ind, Index_ n_rows, Index_ nnz, Lambda op,
                cudaStream_t stream) {
  dim3 grid(raft::ceildiv(n_rows, Index_(TPB_X)), 1, 1);
  dim3 blk(TPB_X, 1, 1);
  csr_row_op_kernel<Index_, TPB_X>
    <<<grid, blk, 0, stream>>>(row_ind, n_rows, nnz, op);

  CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @brief Constructs an adjacency graph CSR row_ind_ptr array from
 * a row_ind array and adjacency array.
 * @tparam T the numeric type of the index arrays
 * @tparam TPB_X the number of threads to use per block for kernels
 * @tparam Lambda function for fused operation in the adj_graph construction
 * @param row_ind the input CSR row_ind array
 * @param total_rows number of vertices in graph
 * @param nnz number of non-zeros
 * @param batchSize number of vertices in current batch
 * @param adj an adjacency array (size batchSize x total_rows)
 * @param row_ind_ptr output CSR row_ind_ptr for adjacency graph
 * @param stream cuda stream to use
 * @param fused_op: the fused operation
 */
template <typename Index_, int TPB_X = 32,
          typename Lambda = auto(Index_, Index_, Index_)->void>
void csr_adj_graph_batched(const Index_ *row_ind, Index_ total_rows, Index_ nnz,
                           Index_ batchSize, const bool *adj,
                           Index_ *row_ind_ptr, cudaStream_t stream,
                           Lambda fused_op) {
  csr_row_op<Index_, TPB_X>(
    row_ind, batchSize, nnz,
    [fused_op, adj, total_rows, row_ind_ptr, batchSize, nnz] __device__(
      Index_ row, Index_ start_idx, Index_ stop_idx) {
      fused_op(row, start_idx, stop_idx);
      Index_ k = 0;
      for (Index_ i = 0; i < total_rows; i++) {
        // @todo: uncoalesced mem accesses!
        if (adj[batchSize * i + row]) {
          row_ind_ptr[start_idx + k] = i;
          k += 1;
        }
      }
    },
    stream);
}

template <typename Index_, int TPB_X = 32,
          typename Lambda = auto(Index_, Index_, Index_)->void>
void csr_adj_graph_batched(const Index_ *row_ind, Index_ total_rows, Index_ nnz,
                           Index_ batchSize, const bool *adj,
                           Index_ *row_ind_ptr, cudaStream_t stream) {
  csr_adj_graph_batched(
    row_ind, total_rows, nnz, batchSize, adj, row_ind_ptr, stream,
    [] __device__(Index_ row, Index_ start_idx, Index_ stop_idx) {});
}

/**
 * @brief Constructs an adjacency graph CSR row_ind_ptr array from a
 * a row_ind array and adjacency array.
 * @tparam T the numeric type of the index arrays
 * @tparam TPB_X the number of threads to use per block for kernels
 * @param row_ind the input CSR row_ind array
 * @param total_rows number of total vertices in graph
 * @param nnz number of non-zeros
 * @param adj an adjacency array
 * @param row_ind_ptr output CSR row_ind_ptr for adjacency graph
 * @param stream cuda stream to use
 * @param fused_op the fused operation
 */
template <typename Index_, int TPB_X = 32,
          typename Lambda = auto(Index_, Index_, Index_)->void>
void csr_adj_graph(const Index_ *row_ind, Index_ total_rows, Index_ nnz,
                   const bool *adj, Index_ *row_ind_ptr, cudaStream_t stream,
                   Lambda fused_op) {
  csr_adj_graph_batched<Index_, TPB_X>(row_ind, total_rows, nnz, total_rows,
                                       adj, row_ind_ptr, stream, fused_op);
}

struct WeakCCState {
 public:
  bool *xa;
  bool *fa;
  bool *m;

  WeakCCState(bool *xa, bool *fa, bool *m) : xa(xa), fa(fa), m(m) {}
};

template <typename Index_, int TPB_X = 32, typename Lambda>
__global__ void weak_cc_label_device(Index_ *labels, const Index_ *row_ind,
                                     const Index_ *row_ind_ptr, Index_ nnz,
                                     bool *fa, bool *xa, bool *m,
                                     Index_ startVertexId, Index_ batchSize,
                                     Index_ N, Lambda filter_op) {
  Index_ tid = threadIdx.x + blockIdx.x * TPB_X;
  Index_ global_id = tid + startVertexId;
  if (tid < batchSize && global_id < N) {
    if (fa[global_id]) {
      fa[global_id] = false;
      Index_ row_ind_val = row_ind[tid];

      Index_ start = row_ind_val;
      Index_ ci, cj;
      bool ci_mod = false;
      ci = labels[global_id];
      bool ci_allow_prop = filter_op(global_id);

      Index_ degree = get_stop_idx(tid, batchSize, nnz, row_ind) - row_ind_val;
      for (Index_ j = 0; j < degree;
           j++) {  // TODO: Can't this be calculated from the ex_scan?
        Index_ j_ind = row_ind_ptr[start + j];
        cj = labels[j_ind];
        bool cj_allow_prop = filter_op(j_ind);
        if (ci < cj && ci_allow_prop) {
          if (sizeof(Index_) == 4)
            atomicMin((int *)(labels + j_ind), ci);
          else if (sizeof(Index_) == 8)
            atomicMin((long long int *)(labels + j_ind), ci);
          ///@todo see https://github.com/rapidsai/cuml/issues/2306.
          // It may be worth it to use an atomic op here such as
          // atomicLogicalOr(xa + j_ind, cj_allow_prop);
          // Same can be done for m : atomicLogicalOr(m, cj_allow_prop);
          // Both can be done below for xa[global_id] with ci_allow_prop, too.
          xa[j_ind] = true;
          m[0] = true;
        } else if (ci > cj && cj_allow_prop) {
          ci = cj;
          ci_mod = true;
        }
      }
      if (ci_mod) {
        if (sizeof(Index_) == 4)
          atomicMin((int *)(labels + global_id), ci);
        else if (sizeof(Index_) == 8)
          atomicMin((long long int *)(labels + global_id), ci);
        xa[global_id] = true;
        m[0] = true;
      }
    }
  }
}

template <typename Index_, int TPB_X = 32, typename Lambda>
__global__ void weak_cc_init_label_kernel(Index_ *labels, Index_ startVertexId,
                                          Index_ batchSize, Index_ MAX_LABEL,
                                          Lambda filter_op) {
  /** F1 and F2 in the paper correspond to fa and xa */
  /** Cd in paper corresponds to db_cluster */
  Index_ tid = threadIdx.x + blockIdx.x * TPB_X;
  if (tid < batchSize) {
    Index_ global_id = tid + startVertexId;
    if (filter_op(global_id) && labels[global_id] == MAX_LABEL)
      labels[global_id] = global_id + 1;
  }
}

template <typename Index_, int TPB_X = 32>
__global__ void weak_cc_init_all_kernel(Index_ *labels, bool *fa, bool *xa,
                                        Index_ N, Index_ MAX_LABEL) {
  Index_ tid = threadIdx.x + blockIdx.x * TPB_X;
  if (tid < N) {
    labels[tid] = MAX_LABEL;
    fa[tid] = true;
    xa[tid] = false;
  }
}

template <typename Index_, int TPB_X = 32, typename Lambda>
void weak_cc_label_batched(Index_ *labels, const Index_ *row_ind,
                           const Index_ *row_ind_ptr, Index_ nnz, Index_ N,
                           WeakCCState *state, Index_ startVertexId,
                           Index_ batchSize, cudaStream_t stream,
                           Lambda filter_op) {
  ASSERT(sizeof(Index_) == 4 || sizeof(Index_) == 8,
         "Index_ should be 4 or 8 bytes");

  bool host_m;

  dim3 blocks(raft::ceildiv(batchSize, Index_(TPB_X)));
  dim3 threads(TPB_X);
  Index_ MAX_LABEL = std::numeric_limits<Index_>::max();

  weak_cc_init_label_kernel<Index_, TPB_X><<<blocks, threads, 0, stream>>>(
    labels, startVertexId, batchSize, MAX_LABEL, filter_op);
  CUDA_CHECK(cudaPeekAtLastError());

  int n_iters = 0;
  do {
    CUDA_CHECK(cudaMemsetAsync(state->m, false, sizeof(bool), stream));

    weak_cc_label_device<Index_, TPB_X><<<blocks, threads, 0, stream>>>(
      labels, row_ind, row_ind_ptr, nnz, state->fa, state->xa, state->m,
      startVertexId, batchSize, N, filter_op);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    //** swapping F1 and F2
    std::swap(state->fa, state->xa);

    //** Updating m *
    raft::update_host(&host_m, state->m, 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    n_iters++;
  } while (host_m);
}

/**
 * @brief Compute weakly connected components. Note that the resulting labels
 * may not be taken from a monotonically increasing set (eg. numbers may be
 * skipped). The MLCommon::Label package contains a primitive `make_monotonic`,
 * which will make a monotonically increasing set of labels.
 *
 * This implementation comes from [1] and solves component labeling problem in
 * parallel on CSR-indexes based upon the vertex degree and adjacency graph.
 *
 * [1] Hawick, K.A et al, 2010. "Parallel graph component labelling with GPUs and CUDA"
 *
 * @tparam Type the numeric type of non-floating point elements
 * @tparam TPB_X the threads to use per block when configuring the kernel
 * @tparam Lambda the type of an optional filter function (int)->bool
 * @param labels an array for the output labels
 * @param row_ind the compressed row index of the CSR array
 * @param row_ind_ptr the row index pointer of the CSR array
 * @param nnz the size of row_ind_ptr array
 * @param N number of vertices
 * @param startVertexId the starting vertex index for the current batch
 * @param batchSize number of vertices for current batch
 * @param state instance of inter-batch state management
 * @param stream the cuda stream to use
 * @param filter_op an optional filtering function to determine which points
 * should get considered for labeling. It gets global indexes (not batch-wide!)
 */
template <typename Index_, int TPB_X = 32, typename Lambda = auto(Index_)->bool>
void weak_cc_batched(Index_ *labels, const Index_ *row_ind,
                     const Index_ *row_ind_ptr, Index_ nnz, Index_ N,
                     Index_ startVertexId, Index_ batchSize, WeakCCState *state,
                     cudaStream_t stream, Lambda filter_op) {
  dim3 blocks(raft::ceildiv(N, Index_(TPB_X)));
  dim3 threads(TPB_X);

  Index_ MAX_LABEL = std::numeric_limits<Index_>::max();
  if (startVertexId == 0) {
    weak_cc_init_all_kernel<Index_, TPB_X><<<blocks, threads, 0, stream>>>(
      labels, state->fa, state->xa, N, MAX_LABEL);
    CUDA_CHECK(cudaPeekAtLastError());
  }

  weak_cc_label_batched<Index_, TPB_X>(labels, row_ind, row_ind_ptr, nnz, N,
                                       state, startVertexId, batchSize, stream,
                                       filter_op);
}

/**
 * @brief Compute weakly connected components. Note that the resulting labels
 * may not be taken from a monotonically increasing set (eg. numbers may be
 * skipped). The MLCommon::Label package contains a primitive `make_monotonic`,
 * which will make a monotonically increasing set of labels.
 *
 * This implementation comes from [1] and solves component labeling problem in
 * parallel on CSR-indexes based upon the vertex degree and adjacency graph.
 *
 * [1] Hawick, K.A et al, 2010. "Parallel graph component labelling with GPUs and CUDA"
 *
 * @tparam Type the numeric type of non-floating point elements
 * @tparam TPB_X the threads to use per block when configuring the kernel
 * @tparam Lambda the type of an optional filter function (int)->bool
 * @param labels an array for the output labels
 * @param row_ind the compressed row index of the CSR array
 * @param row_ind_ptr the row index pointer of the CSR array
 * @param nnz the size of row_ind_ptr array
 * @param N number of vertices
 * @param startVertexId the starting vertex index for the current batch
 * @param batchSize number of vertices for current batch
 * @param state instance of inter-batch state management
 * @param stream the cuda stream to use
 */
template <typename Index_, int TPB_X = 32>
void weak_cc_batched(Index_ *labels, const Index_ *row_ind,
                     const Index_ *row_ind_ptr, Index_ nnz, Index_ N,
                     Index_ startVertexId, Index_ batchSize, WeakCCState *state,
                     cudaStream_t stream) {
  weak_cc_batched(labels, row_ind, row_ind_ptr, nnz, N, startVertexId,
                  batchSize, state, stream,
                  [] __device__(Index_ tid) { return true; });
}

/**
 * @brief Compute weakly connected components. Note that the resulting labels
 * may not be taken from a monotonically increasing set (eg. numbers may be
 * skipped). The MLCommon::Label package contains a primitive `make_monotonic`,
 * which will make a monotonically increasing set of labels.
 *
 * This implementation comes from [1] and solves component labeling problem in
 * parallel on CSR-indexes based upon the vertex degree and adjacency graph.
 *
 * [1] Hawick, K.A et al, 2010. "Parallel graph component labelling with GPUs and CUDA"
 *
 * @tparam Type the numeric type of non-floating point elements
 * @tparam TPB_X the threads to use per block when configuring the kernel
 * @tparam Lambda the type of an optional filter function (int)->bool
 * @param labels an array for the output labels
 * @param row_ind the compressed row index of the CSR array
 * @param row_ind_ptr the row index pointer of the CSR array
 * @param nnz the size of row_ind_ptr array
 * @param N number of vertices
 * @param d_alloc: deviceAllocator to use for temp memory
 * @param stream the cuda stream to use
 * @param filter_op an optional filtering function to determine which points
 * should get considered for labeling. It gets global indexes (not batch-wide!)
 */
template <typename Index_ = int, int TPB_X = 32,
          typename Lambda = auto(Index_)->bool>
void weak_cc(Index_ *labels, const Index_ *row_ind, const Index_ *row_ind_ptr,
             Index_ nnz, Index_ N, std::shared_ptr<deviceAllocator> d_alloc,
             cudaStream_t stream, Lambda filter_op) {
  device_buffer<bool> xa(d_alloc, stream, N);
  device_buffer<bool> fa(d_alloc, stream, N);
  device_buffer<bool> m(d_alloc, stream, 1);

  WeakCCState state(xa.data(), fa.data(), m.data());
  weak_cc_batched<Index_, TPB_X>(labels, row_ind, row_ind_ptr, nnz, N, 0, N,
                                 stream, filter_op);
}

/**
 * @brief Compute weakly connected components. Note that the resulting labels
 * may not be taken from a monotonically increasing set (eg. numbers may be
 * skipped). The MLCommon::Label package contains a primitive `make_monotonic`,
 * which will make a monotonically increasing set of labels.
 *
 * This implementation comes from [1] and solves component labeling problem in
 * parallel on CSR-indexes based upon the vertex degree and adjacency graph.
 *
 * [1] Hawick, K.A et al, 2010. "Parallel graph component labelling with GPUs and CUDA"
 *
 * @tparam Type the numeric type of non-floating point elements
 * @tparam TPB_X the threads to use per block when configuring the kernel
 * @tparam Lambda the type of an optional filter function (int)->bool
 * @param labels an array for the output labels
 * @param row_ind the compressed row index of the CSR array
 * @param row_ind_ptr the row index pointer of the CSR array
 * @param nnz the size of row_ind_ptr array
 * @param N number of vertices
 * @param d_alloc: deviceAllocator to use for temp memory
 * @param stream the cuda stream to use
 */
template <typename Index_, int TPB_X = 32>
void weak_cc(Index_ *labels, const Index_ *row_ind, const Index_ *row_ind_ptr,
             Index_ nnz, Index_ N, std::shared_ptr<deviceAllocator> d_alloc,
             cudaStream_t stream) {
  device_buffer<bool> xa(d_alloc, stream, N);
  device_buffer<bool> fa(d_alloc, stream, N);
  device_buffer<bool> m(d_alloc, stream, 1);
  WeakCCState state(xa.data(), fa.data(), m.data());
  weak_cc_batched<Index_, TPB_X>(labels, row_ind, row_ind_ptr, nnz, N, 0, N,
                                 stream, [](Index_) { return true; });
}

};  // namespace Sparse
};  // namespace MLCommon
