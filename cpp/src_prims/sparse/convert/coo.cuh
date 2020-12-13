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

#include <cusparse_v2.h>
#include <raft/cudart_utils.h>
#include <raft/sparse/cusparse_wrappers.h>
#include <raft/cuda_utils.cuh>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>

#include <sparse/utils.h>
#include <sparse/coo.cuh>

namespace raft {
namespace sparse {
namespace convert {

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

template <int TPB_X, typename T>
__global__ void from_knn_graph_kernel(const long *knn_indices,
                                      const T *knn_dists, int m, int k,
                                      int *rows, int *cols, T *vals) {
  int row = (blockIdx.x * TPB_X) + threadIdx.x;
  if (row < m) {
    for (int i = 0; i < k; i++) {
      rows[row * k + i] = row;
      cols[row * k + i] = knn_indices[row * k + i];
      vals[row * k + i] = knn_dists[row * k + i];
    }
  }
}

/**
 * @brief Converts a knn graph, defined by index and distance matrices,
 * into COO format.
 *
 * @param knn_indices: knn index array
 * @param knn_dists: knn distance array
 * @param m: number of vertices in graph
 * @param k: number of nearest neighbors
 * @param rows: output COO row array
 * @param cols: output COO col array
 * @param vals: output COO val array
 */
template <typename T>
void from_knn(const long *knn_indices, const T *knn_dists, int m, int k,
              int *rows, int *cols, T *vals) {
  dim3 grid(raft::ceildiv(m, 32), 1, 1);
  dim3 blk(32, 1, 1);
  from_knn_graph_kernel<32, T>
    <<<grid, blk>>>(knn_indices, knn_dists, m, k, rows, cols, vals);
  CUDA_CHECK(cudaGetLastError());
}

/**
 * Converts a knn graph, defined by index and distance matrices,
 * into COO format.
 * @param knn_indices: KNN index array (size m * k)
 * @param knn_dists: KNN dist array (size m * k)
 * @param m: number of vertices in graph
 * @param k: number of nearest neighbors
 * @param out: The output COO graph from the KNN matrices
 * @param stream: CUDA stream to use
 */
template <typename T>
void from_knn(const long *knn_indices, const T *knn_dists, int m, int k,
              COO<T> *out, cudaStream_t stream) {
  out->allocate(m * k, m, m, true, stream);

  from_knn(knn_indices, knn_dists, m, k, out->rows(), out->cols(), out->vals());
}

};  // end NAMESPACE convert
};  // end NAMESPACE sparse
};  // end NAMESPACE raft