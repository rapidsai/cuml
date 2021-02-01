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

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cuda_runtime.h>
#include <stdio.h>

#include <iostream>
#include <limits>

#include <sparse/utils.h>

namespace raft {
namespace sparse {
namespace linalg {

template <int TPB_X = 64, typename T>
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
template <int TPB_X = 64, typename T>
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

template <int TPB_X = 64, typename T>
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

    T max = std::numeric_limits<float>::min();
    for (int j = start_idx; j < stop_idx; j++) {
      if (vals[j] > max) max = vals[j];
    }

    // divide nonzeros in current row by max
    for (int j = start_idx; j < stop_idx; j++) {
      if (max != 0.0 && max > std::numeric_limits<float>::min()) {
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

template <int TPB_X = 64, typename T>
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

};  // end NAMESPACE linalg
};  // end NAMESPACE sparse
};  // end NAMESPACE raft