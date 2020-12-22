/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <matrix/reverse.cuh>
#include <raft/matrix/matrix.cuh>

#include <selection/knn.cuh>
#include <sparse/coo.cuh>
#include <sparse/csr.cuh>
#include <sparse/distance.cuh>

#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/utils/Heap.h>
#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/Select.cuh>

#include <raft/cudart_utils.h>
#include <common/device_buffer.hpp>
#include <raft/cuda_utils.cuh>

#include <raft/sparse/cusparse_wrappers.h>

#include <cusparse_v2.h>

namespace MLCommon {
namespace Sparse {
namespace Selection {

template <typename value_idx>
__global__ void iota_fill_warp_kernel(value_idx *indices, value_idx ncols) {
  int row = blockIdx.x;
  int tid = threadIdx.x;

  for (int i = tid; i < ncols; i += blockDim.x) {
    indices[row * ncols + i] = i;
  }
}

template <typename value_idx>
void iota_fill(value_idx *indices, value_idx nrows, value_idx ncols,
               cudaStream_t stream) {
  int blockdim = block_dim(ncols);

  iota_fill_warp_kernel<<<nrows, blockdim, 0, stream>>>(indices, ncols);
}

template <typename K, typename IndexType, bool select_min, int warp_q,
          int thread_q, int tpb>
__global__ void select_k_kernel(K *inK, IndexType *inV, size_t n_rows,
                                size_t n_cols, K *outK, IndexType *outV,
                                K initK, IndexType initV, int k) {
  constexpr int kNumWarps = tpb / faiss::gpu::kWarpSize;

  __shared__ K smemK[kNumWarps * warp_q];
  __shared__ IndexType smemV[kNumWarps * warp_q];

  faiss::gpu::BlockSelect<K, IndexType, select_min, faiss::gpu::Comparator<K>,
                          warp_q, thread_q, tpb>
    heap(initK, initV, smemK, smemV, k);

  // Grid is exactly sized to rows available
  int row = blockIdx.x;
  int i = threadIdx.x;

  int idx = row * n_cols;
  K *inKStart = inK + idx + i;
  IndexType *inVStart = inV + idx + i;

  // Whole warps must participate in the selection
  int limit = faiss::gpu::utils::roundDown(n_cols, faiss::gpu::kWarpSize);

  for (; i < limit; i += tpb) {
    inKStart = inK + idx + i;
    inVStart = inV + idx + i;

    heap.add(*inKStart, *inVStart);
  }

  // Handle last remainder fraction of a warp of elements
  if (i < n_cols) {
    inKStart = inK + idx + i;
    inVStart = inV + idx + i;
    heap.addThreadQ(*inKStart, *inVStart);
  }

  heap.reduce();

  for (int i = threadIdx.x; i < k; i += tpb) {
    outK[row * k + i] = smemK[i];
    outV[row * k + i] = smemV[i];
  }
}

template <typename value_idx = int, typename value_t = float, int warp_q,
          int thread_q>
inline void select_k_impl(value_t *inK, value_idx *inV, size_t n_rows,
                          size_t n_cols, value_t *outK, value_idx *outV,
                          bool select_min, int k, cudaStream_t stream) {
  auto grid = dim3(n_rows);

  constexpr int n_threads = (warp_q <= 1024) ? 128 : 64;
  auto block = dim3(n_threads);

  auto kInit = select_min ? faiss::gpu::Limits<value_t>::getMax()
                          : faiss::gpu::Limits<value_t>::getMin();
  auto vInit = -1;
  if (select_min) {
    select_k_kernel<value_t, value_idx, false, warp_q, thread_q, n_threads>
      <<<grid, block, 0, stream>>>(inK, inV, n_rows, n_cols, outK, outV, kInit,
                                   vInit, k);
  } else {
    select_k_kernel<value_t, value_idx, true, warp_q, thread_q, n_threads>
      <<<grid, block, 0, stream>>>(inK, inV, n_rows, n_cols, outK, outV, kInit,
                                   vInit, k);
  }
  CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Select the k-nearest neighbors from dense
 * distance and index matrices.
 *
 * @param[in] inK partitioned knn distance matrix
 * @param[in] inV partitioned knn index matrix
 * @param[in] n_rows number of rows in distance and index matrices
 * @param[in] n_cols number of columns in distance and index matrices
 * @param[out] outK merged knn distance matrix
 * @param[out] outV merged knn index matrix
 * @param[in] select_min whether to select the min or the max distances
 * @param[in] k number of neighbors per partition (also number of merged neighbors)
 * @param[in] stream CUDA stream to use
 */
template <typename value_idx = int, typename value_t = float>
inline void select_k(value_t *inK, value_idx *inV, size_t n_rows, size_t n_cols,
                     value_t *outK, value_idx *outV, bool select_min, int k,
                     cudaStream_t stream) {
  if (k == 1)
    select_k_impl<value_idx, value_t, 1, 1>(inK, inV, n_rows, n_cols, outK,
                                            outV, select_min, k, stream);
  else if (k <= 32)
    select_k_impl<value_idx, value_t, 32, 2>(inK, inV, n_rows, n_cols, outK,
                                             outV, select_min, k, stream);
  else if (k <= 64)
    select_k_impl<value_idx, value_t, 64, 3>(inK, inV, n_rows, n_cols, outK,
                                             outV, select_min, k, stream);
  else if (k <= 128)
    select_k_impl<value_idx, value_t, 128, 3>(inK, inV, n_rows, n_cols, outK,
                                              outV, select_min, k, stream);
  else if (k <= 256)
    select_k_impl<value_idx, value_t, 256, 4>(inK, inV, n_rows, n_cols, outK,
                                              outV, select_min, k, stream);
  else if (k <= 512)
    select_k_impl<value_idx, value_t, 512, 8>(inK, inV, n_rows, n_cols, outK,
                                              outV, select_min, k, stream);
  else if (k <= 1024)
    select_k_impl<value_idx, value_t, 1024, 8>(inK, inV, n_rows, n_cols, outK,
                                               outV, select_min, k, stream);
}

};  // END namespace Selection
};  // END namespace Sparse
};  // END namespace MLCommon
