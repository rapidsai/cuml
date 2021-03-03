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

#include <raft/cudart_utils.h>
#include <raft/cuda_utils.cuh>

#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/GpuIndexIVFScalarQuantizer.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/utils/Heap.h>
#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/Select.cuh>
#include <faiss/gpu/utils/Tensor.cuh>

#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>

#include <cuml/common/cuml_allocator.hpp>
#include <cuml/common/device_buffer.hpp>

#include <iostream>
#include <set>


namespace raft {
namespace selection {


/**
 * @tparam value_idx data type of indices
 * @tparam value_t data type of values and distances
 * @tparam warp_q
 * @tparam thread_q
 * @tparam tpb
 * @param out_inds
 * @param out_dists
 * @param index
 * @param query
 * @param n_index_rows
 * @param k
 */
template<typename value_idx = int, typename value_t = float,
  int warp_q = 1024, int thread_q = 8, int tpb = 128>
__global__ void haversine_knn_kernel(value_idx *out_inds, value_t *out_dists,
                      value_t *index, value_t *query,
                      size_t n_index_rows, int k) {

  value_idx query_row = blockIdx.x;

  constexpr int kNumWarps = tpb / faiss::gpu::kWarpSize;

  __shared__ value_t smemK[kNumWarps * warp_q];
  __shared__ value_idx smemV[kNumWarps * warp_q];

  faiss::gpu::BlockSelect<value_t, value_idx, false,
    faiss::gpu::Comparator<value_t>, warp_q, thread_q,
    tpb>
    heap(faiss::gpu::Limits<value_t>::getMax(), -1, smemK, smemV, k);

  // Grid is exactly sized to rows available
  int chunk_size = faiss::gpu::utils::roundDown(n_index_rows, tpb);

  value_t *query_ptr = query + (blockIdx.x * 2);
  value_t x1 = query_ptr[0];
  value_t y1 = query_ptr[1];

  int i = threadIdx.x;

  value_idx translation = 0;

  for (i = threadIdx.x; i < chunk_size; i += tpb) {

    value_t *idx_ptr = index + (i * 2);
    value_t x2 = idx_ptr[0];
    value_t y2 = idx_ptr[1];

    value_t dist = 2 * sqrt(pow(sin((x1-y1)/2), 2) +
                            cos(x1) * cos(y1) * pow(sin((x2 - y2)/2)));

    heap.add(dist, i);
  }

  heap.reduce();

  for (int i = threadIdx.x; i < k; i += tpb) {
    out_dists[blockIdx.x * k + i] = smemK[i];
    out_inds[blockIdx.x * k + i] = smemV[i];
  }
}

/**
 *
 * @tparam value_idx
 * @tparam value_t
 * @param out_inds
 * @param out_dists
 * @param index
 * @param query
 * @param n_index_rows
 * @param n_query_rows
 * @param k
 * @param stream
 */
template<typename value_idx, typename value_t>
void haversine_knn(value_idx *out_inds, value_t *out_dists,
                   value_t *index, value_t *query,
                   size_t n_index_rows, size_t n_query_rows,
                   int k, cudaStream_t stream) {

  haversine_knn_kernel<<<n_query_rows, 128, 0, stream>>>(
    out_inds, out_dists, index, query, n_index_rows, k);
}

}; // end selection
}; // end raft