/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cusparse_v2.h>

#include <raft/cudart_utils.h>
#include <raft/sparse/cusparse_wrappers.h>
#include <raft/cuda_utils.cuh>
#include <raft/mr/device/allocator.hpp>
#include <raft/mr/device/buffer.hpp>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cuda_runtime.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>

#include <sparse/utils.h>

namespace raft {
namespace sparse {
namespace linalg {

template <typename T, int TPB_X = 128>
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

template <typename T, int TPB_X = 128>
__global__ void csr_add_kernel(const int *a_ind, const int *a_indptr,
                               const T *a_val, int nnz1, const int *b_ind,
                               const int *b_indptr, const T *b_val, int nnz2,
                               int m, int *out_ind, int *out_indptr,
                               T *out_val) {
  // 1 thread per row
  int row = (blockIdx.x * TPB_X) + threadIdx.x;

  if (row < m) {
    int a_start_idx = a_ind[row];

    // TODO: Shouldn't need this if rowind is proper CSR
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
 * @param d_alloc: device allocator to use for temp memory
 * @param stream: cuda stream to use
 */
template <typename T, int TPB_X = 128>
size_t csr_add_calc_inds(const int *a_ind, const int *a_indptr, const T *a_val,
                         int nnz1, const int *b_ind, const int *b_indptr,
                         const T *b_val, int nnz2, int m, int *out_ind,
                         std::shared_ptr<raft::mr::device::allocator> d_alloc,
                         cudaStream_t stream) {
  dim3 grid(raft::ceildiv(m, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  raft::mr::device::buffer<int> row_counts(d_alloc, stream, m + 1);
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
template <typename T, int TPB_X = 128>
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

};  // end NAMESPACE linalg
};  // end NAMESPACE sparse
};  // end NAMESPACE raft
