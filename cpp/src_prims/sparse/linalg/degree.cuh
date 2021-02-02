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

#include <sparse/utils.h>
#include <sparse/coo.cuh>

namespace raft {
namespace sparse {
namespace linalg {

/**
 * @brief Count all the rows in the coo row array and place them in the
 * results matrix, indexed by row.
 *
 * @tparam TPB_X: number of threads to use per block
 * @param rows the rows array of the coo matrix
 * @param nnz the size of the rows array
 * @param results array to place results
 */
template <int TPB_X = 64>
__global__ void coo_degree_kernel(const int *rows, int nnz, int *results) {
  int row = (blockIdx.x * TPB_X) + threadIdx.x;
  if (row < nnz) {
    raft::myAtomicAdd(results + rows[row], 1);
  }
}

/**
 * @brief Count the number of values for each row
 * @tparam TPB_X: number of threads to use per block
 * @param rows: rows array of the COO matrix
 * @param nnz: size of the rows array
 * @param results: output result array
 * @param stream: cuda stream to use
 */
template <int TPB_X = 64>
void coo_degree(const int *rows, int nnz, int *results, cudaStream_t stream) {
  dim3 grid_rc(raft::ceildiv(nnz, TPB_X), 1, 1);
  dim3 blk_rc(TPB_X, 1, 1);

  coo_degree_kernel<TPB_X><<<grid_rc, blk_rc, 0, stream>>>(rows, nnz, results);
  CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Count the number of values for each row
 * @tparam TPB_X: number of threads to use per block
 * @tparam T: type name of underlying values array
 * @param in: input COO object for counting rows
 * @param results: output array with row counts (size=in->n_rows)
 * @param stream: cuda stream to use
 */
template <int TPB_X = 64, typename T>
void coo_degree(COO<T> *in, int *results, cudaStream_t stream) {
  dim3 grid_rc(raft::ceildiv(in->nnz, TPB_X), 1, 1);
  dim3 blk_rc(TPB_X, 1, 1);

  coo_degree_kernel<TPB_X>
    <<<grid_rc, blk_rc, 0, stream>>>(in->rows(), in->nnz, results);
  CUDA_CHECK(cudaGetLastError());
}

template <int TPB_X = 64, typename T>
__global__ void coo_degree_nz_kernel(const int *rows, const T *vals, int nnz,
                                     int *results) {
  int row = (blockIdx.x * TPB_X) + threadIdx.x;
  if (row < nnz && vals[row] != 0.0) {
    raft::myAtomicAdd(results + rows[row], 1);
  }
}

template <int TPB_X = 64, typename T>
__global__ void coo_degree_scalar_kernel(const int *rows, const T *vals,
                                         int nnz, T scalar, int *results) {
  int row = (blockIdx.x * TPB_X) + threadIdx.x;
  if (row < nnz && vals[row] != scalar) {
    raft::myAtomicAdd(results + rows[row], 1);
  }
}

/**
 * @brief Count the number of values for each row matching a particular scalar
 * @tparam TPB_X: number of threads to use per block
 * @tparam T: the type name of the underlying value arrays
 * @param in: Input COO array
 * @param scalar: scalar to match for counting rows
 * @param results: output row counts
 * @param stream: cuda stream to use
 */
template <int TPB_X = 64, typename T>
void coo_degree_scalar(COO<T> *in, T scalar, int *results,
                       cudaStream_t stream) {
  dim3 grid_rc(raft::ceildiv(in->nnz, TPB_X), 1, 1);
  dim3 blk_rc(TPB_X, 1, 1);
  coo_degree_scalar_kernel<TPB_X, T><<<grid_rc, blk_rc, 0, stream>>>(
    in->rows(), in->vals(), in->nnz, scalar, results);
  CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Count the number of values for each row matching a particular scalar
 * @tparam TPB_X: number of threads to use per block
 * @tparam T: the type name of the underlying value arrays
 * @param rows: Input COO row array
 * @param vals: Input COO val arrays
 * @param nnz: size of input COO arrays
 * @param scalar: scalar to match for counting rows
 * @param results: output row counts
 * @param stream: cuda stream to use
 */
template <int TPB_X = 64, typename T>
void coo_degree_scalar(const int *rows, const T *vals, int nnz, T scalar,
                       int *results, cudaStream_t stream = 0) {
  dim3 grid_rc(raft::ceildiv(nnz, TPB_X), 1, 1);
  dim3 blk_rc(TPB_X, 1, 1);
  coo_degree_scalar_kernel<TPB_X, T>
    <<<grid_rc, blk_rc, 0, stream>>>(rows, vals, nnz, scalar, results);
}

/**
 * @brief Count the number of nonzeros for each row
 * @tparam TPB_X: number of threads to use per block
 * @tparam T: the type name of the underlying value arrays
 * @param rows: Input COO row array
 * @param vals: Input COO val arrays
 * @param nnz: size of input COO arrays
 * @param results: output row counts
 * @param stream: cuda stream to use
 */
template <int TPB_X = 64, typename T>
void coo_degree_nz(const int *rows, const T *vals, int nnz, int *results,
                   cudaStream_t stream) {
  dim3 grid_rc(raft::ceildiv(nnz, TPB_X), 1, 1);
  dim3 blk_rc(TPB_X, 1, 1);
  coo_degree_nz_kernel<TPB_X, T>
    <<<grid_rc, blk_rc, 0, stream>>>(rows, vals, nnz, results);
}

/**
 * @brief Count the number of nonzero values for each row
 * @tparam TPB_X: number of threads to use per block
 * @tparam T: the type name of the underlying value arrays
 * @param in: Input COO array
 * @param results: output row counts
 * @param stream: cuda stream to use
 */
template <int TPB_X = 64, typename T>
void coo_degree_nz(COO<T> *in, int *results, cudaStream_t stream) {
  dim3 grid_rc(raft::ceildiv(in->nnz, TPB_X), 1, 1);
  dim3 blk_rc(TPB_X, 1, 1);

  coo_degree_nz_kernel<TPB_X, T>
    <<<grid_rc, blk_rc, 0, stream>>>(in->rows(), in->vals(), in->nnz, results);
}

};  // end NAMESPACE linalg
};  // end NAMESPACE sparse
};  // end NAMESPACE raft