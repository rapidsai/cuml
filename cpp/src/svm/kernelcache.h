/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cuda_utils.h>
#include <linalg/gemm.h>
#include "cache/cache.h"
#include "common/cumlHandle.hpp"
#include "common/host_buffer.hpp"
#include "gram/grammatrix.h"
#include "ml_utils.h"

namespace ML {
namespace SVM {

/**
 * @brief Collect rows of the training data into contiguous space
 *
 * The working set is a subset of all the training examples. Here we collect
 * all the training vectors that are in the working set.
 *
 * @param [in] x training data in column major format, size [n_rows x n_cols]
 * @param [in] n_rows
 * @param [in] n_cols
 * @param [out] x_ws training vectors in the working set in column major format, size [n_ws x n_cols]
 * @param [in] n_ws the number of elements in the working set
 * @param [in] ws_idx working set indices (row indices of x), size [n_ws]
 */
template <typename math_t>
__global__ void get_rows(const math_t *x, int n_rows, int n_cols, math_t *x_ws,
                         int n_ws, const int *ws_idx) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int ws_row = tid % n_ws;  // row idx
  if (tid < n_ws * n_cols) {
    int x_row = ws_idx[ws_row];  // row in the original matrix
    int col = tid / n_ws;        // col idx
    if (x_row + col * n_rows < n_rows * n_cols) {
      x_ws[tid] = x[x_row + col * n_rows];
    }
  }
}

/**
* @brief Buffer to store a kernel tile
*
* We calculate the kernel matrix for the vectors in the working set.
* For every vector x_i in the working set, we always calculate a full row of the
* kernel matrix K(x_j, x_i), j=1..n_rows.
*
* A kernel tile stores all the kernel rows for the working set, i.e. K(x_j, x_i)
* for all i in the working set, and j in 1..n_rows.
*
* The kernel values can be cached to avoid repeated calculation of the kernel
* function.
*/
template <typename math_t>
class KernelCache {
 private:
  const math_t *x;  //!< pointer to the training vectors

  MLCommon::device_buffer<math_t>
    x_ws;  //!< feature vectors in the current working set
  MLCommon::device_buffer<int>
    ws_cache_idx;  //!< cache position of a workspace vectors
  MLCommon::device_buffer<math_t> tile;  //!< Kernel matrix  tile

  int n_rows;  //!< number of rows in x
  int n_cols;  //!< number of columns in x
  int n_ws;    //!< number of elements in the working set

  cublasHandle_t cublas_handle;

  MLCommon::GramMatrix::GramMatrixBase<math_t> *kernel;

  const cumlHandle_impl handle;

  const int TPB = 256;  //!< threads per block for kernels launched

  MLCommon::Cache::Cache<math_t> cache;

  cudaStream_t stream;

 public:
  /**
   * Construct an object to manage kernel cache
   *
   * @param handle reference to cumlHandle implementation
   * @param x device array of training vectors in column major format,
   *   size [n_rows x n_cols]
   * @param n_rows number of training vectors
   * @param n_cols number of features
   * @param n_ws size of working set
   * @param kernel pointer to kernel (default linear)
   * @param cache_size (default 200 MiB)
   */
  KernelCache(const cumlHandle_impl &handle, const math_t *x, int n_rows,
              int n_cols, int n_ws,
              MLCommon::GramMatrix::GramMatrixBase<math_t> *kernel,
              float cache_size = 200)
    : cache(handle.getDeviceAllocator(), handle.getStream(), n_rows,
            cache_size),
      kernel(kernel),
      x(x),
      n_rows(n_rows),
      n_cols(n_cols),
      n_ws(n_ws),
      cublas_handle(handle.getCublasHandle()),
      x_ws(handle.getDeviceAllocator(), handle.getStream(), n_ws * n_cols),
      tile(handle.getDeviceAllocator(), handle.getStream(), n_ws * n_rows),
      ws_cache_idx(handle.getDeviceAllocator(), handle.getStream(), n_ws) {
    ASSERT(kernel != nullptr, "Kernel pointer required for KernelCache!");

    stream = handle.getStream();
  }

  ~KernelCache(){};

  /**
 * @brief Collect rows of the training data into contiguous space
 *
 * The working set is a subset of all the training examples. Here we collect
 * all the training vectors that are in the working set.
 *
 * @param [out] x_ws training vectors in the working set in column major format,
 *     size [n x n_cols]
 * @param [in] n the number of elements in the working set
 * @param [in] ws_idx working set indices (row indices of x), size [n_ws]
 * @param [in] stream
 */
  void CollectRows(math_t *x_ws, int n, const int *ws_idx,
                   cudaStream_t stream) {
    if (n > 0) {
      get_rows<<<MLCommon::ceildiv(n * n_cols, TPB), TPB, 0, stream>>>(
        x, n_rows, n_cols, x_ws, n, ws_idx);
      CUDA_CHECK(cudaPeekAtLastError());
    }
  }

  /**
   * @brief Get all the kernel matrix rows for the working set.
   * @param ws_idx indices of the working set
   * @return pointer to the kernel tile [ n_rows x n_ws] K_j,i = K(x_j, x_q)
   * where j=1..n_rows and q = ws_idx[i], j is the contiguous dimension
   */
  math_t *GetTile(int *ws_idx) {
    if (cache.GetSize() > 0) {
      int n_cached;
      cache.GetCacheIdxPartitioned(ws_idx, n_ws, ws_cache_idx.data(), &n_cached,
                                   stream);
      // collect allready cached values
      cache.GetVecs(ws_cache_idx.data(), n_cached, tile.data(), stream);

      int non_cached = n_ws - n_cached;
      if (non_cached > 0) {
        int *ws_idx_new = ws_idx + n_cached;
        // AssignCacheIdx can permute ws_idx_new, therefore it has to come
        // before calcKernel. Could come on separate stream to do collectrows
        // while AssignCacheIdx runs
        cache.AssignCacheIdx(ws_idx_new, non_cached,
                             ws_cache_idx.data() + n_cached,
                             stream);  // cache stream

        // collect training vectors for kernel elements that needs to be calculated
        CollectRows(x_ws.data(), non_cached, ws_idx_new, stream);
        math_t *tile_new = tile.data() + n_cached * n_rows;
        (*kernel)(x, n_rows, n_cols, x_ws.data(), non_cached, tile_new, stream);
        // We need AssignCacheIdx to be finished before calling StoreCols
        cache.StoreVecs(tile_new, n_rows, non_cached,
                        ws_cache_idx.data() + n_cached, stream);
      }
    } else {
      if (n_ws > 0) {
        // collect all the feature vectors in the working set
        CollectRows(x_ws.data(), n_ws, ws_idx, stream);
        (*kernel)(x, n_rows, n_cols, x_ws.data(), n_ws, tile.data(), stream);
      }
    }
    return tile.data();
  }
};

};  // end namespace SVM
};  // end namespace ML
