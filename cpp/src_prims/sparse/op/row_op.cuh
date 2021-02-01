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

#include <algorithm>
#include <iostream>

#include <sparse/utils.h>

namespace raft {
namespace sparse {
namespace op {

template <typename T, int TPB_X = 256, typename Lambda = auto(T, T, T)->void>
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
template <typename Index_, int TPB_X = 256,
          typename Lambda = auto(Index_, Index_, Index_)->void>
void csr_row_op(const Index_ *row_ind, Index_ n_rows, Index_ nnz, Lambda op,
                cudaStream_t stream) {
  dim3 grid(raft::ceildiv(n_rows, Index_(TPB_X)), 1, 1);
  dim3 blk(TPB_X, 1, 1);
  csr_row_op_kernel<Index_, TPB_X>
    <<<grid, blk, 0, stream>>>(row_ind, n_rows, nnz, op);

  CUDA_CHECK(cudaPeekAtLastError());
}

};  // namespace op
};  // end NAMESPACE sparse
};  // end NAMESPACE raft
