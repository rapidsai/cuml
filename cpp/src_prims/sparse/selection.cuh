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

#include <matrix/matrix.cuh>
#include <matrix/reverse.cuh>

#include <selection/knn.cuh>
#include <sparse/coo.cuh>
#include <sparse/csr.cuh>
#include <sparse/distances.cuh>

#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/utils/Heap.h>
#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/Select.cuh>

#include <common/cudart_utils.h>
#include <common/device_buffer.hpp>
#include <cuda_utils.cuh>

#include <raft/sparse/cusparse_wrappers.h>

#include <cusparse_v2.h>

namespace MLCommon {
namespace Sparse {
namespace Selection {

template <typename K, typename IndexType, bool select_min, int warp_q, int thread_q, int tpb>
__global__ void select_k_kernel(K *inK, IndexType *inV, size_t n_rows,
                                size_t n_cols, K *outK, IndexType *outV,
                                K initK, IndexType initV,
                                int k, IndexType translation = 0) {
  constexpr int kNumWarps = tpb / faiss::gpu::kWarpSize;

  __shared__ K smemK[kNumWarps * warp_q];
  __shared__ IndexType smemV[kNumWarps * warp_q];

  faiss::gpu::BlockSelect<K, IndexType, select_min, faiss::gpu::Comparator<K>, warp_q,
                          thread_q, tpb>
    heap(initK, initV, smemK, smemV, k);

  // Grid is exactly sized to rows available
  int row = blockIdx.x;

  int i = threadIdx.x;
  K *inKStart = inK + (row * n_cols + i);
  IndexType *inVStart = inV + (row * n_cols + i);

  // Whole warps must participate in the selection
  int limit = faiss::gpu::utils::roundDown(n_cols, faiss::gpu::kWarpSize);
  for (; i < limit; i += tpb) {
    heap.add(*inKStart, (*inVStart) + translation);
    inKStart += tpb;
    inVStart += tpb;
  }

  // Handle last remainder fraction of a warp of elements
  if (i < n_cols) {
    heap.addThreadQ(*inKStart, (*inVStart) + translation);
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
                          bool select_min, int k, cudaStream_t stream,
                          value_idx translation = 0) {
  auto grid = dim3(n_rows);

  constexpr int n_threads = (warp_q <= 1024) ? 128 : 64;
  auto block = dim3(n_threads);

  auto kInit = select_min ? faiss::gpu::Limits<float>::getMax()
                          : faiss::gpu::Limits<float>::getMin();
  auto vInit = -1;
  if(select_min) {
	  select_k_kernel<value_t, value_idx, true, warp_q, thread_q, n_threads>
	    <<<grid, block, 0, stream>>>(inK, inV, n_rows, n_cols, outK, outV, kInit,
	                                 vInit, k, translation);
  } else {
	  select_k_kernel<value_t, value_idx, false, warp_q, thread_q, n_threads>
	    <<<grid, block, 0, stream>>>(inK, inV, n_rows, n_cols, outK, outV, kInit,
	                                 vInit, k, translation);
  }
  CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @brief Merge knn distances and index matrix, which have been partitioned
 * by row, into a single matrix with only the k-nearest neighbors.
 *
 * @param inK partitioned knn distance matrix
 * @param inV partitioned knn index matrix
 * @param outK merged knn distance matrix
 * @param outV merged knn index matrix
 * @param n_samples number of samples per partition
 * @param n_parts number of partitions
 * @param k number of neighbors per partition (also number of merged neighbors)
 * @param stream CUDA stream to use
 * @param translations mapping of index offsets for each partition
 */
template <typename value_idx = int, typename value_t = float>
inline void select_k(value_t *inK, value_idx *inV, size_t n_rows, size_t n_cols,
                     value_t *outK, value_idx *outV, bool select_min, int k,
                     cudaStream_t stream, value_idx translation = 0) {
  if (k == 1)
    select_k_impl<value_idx, value_t, 1, 1>(
      inK, inV, n_rows, n_cols, outK, outV, select_min, k, stream, translation);
  else if (k <= 32)
    select_k_impl<value_idx, value_t, 32, 2>(
      inK, inV, n_rows, n_cols, outK, outV, select_min, k, stream, translation);
  else if (k <= 64)
    select_k_impl<value_idx, value_t, 64, 3>(
      inK, inV, n_rows, n_cols, outK, outV, select_min, k, stream, translation);
  else if (k <= 128)
    select_k_impl<value_idx, value_t, 128, 3>(
      inK, inV, n_rows, n_cols, outK, outV, select_min, k, stream, translation);
  else if (k <= 256)
    select_k_impl<value_idx, value_t, 256, 4>(
      inK, inV, n_rows, n_cols, outK, outV, select_min, k, stream, translation);
  else if (k <= 512)
    select_k_impl<value_idx, value_t, 512, 8>(
      inK, inV, n_rows, n_cols, outK, outV, select_min, k, stream, translation);
  else if (k <= 1024)
    select_k_impl<value_idx, value_t, 1024, 8>(
      inK, inV, n_rows, n_cols, outK, outV, select_min, k, stream, translation);
}

};  // END namespace Selection
};  // END namespace Sparse
};  // END namespace MLCommon
