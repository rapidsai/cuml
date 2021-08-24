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

#include <raft/cuda_utils.cuh>

#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/Select.cuh>

namespace raft {
namespace selection {

template <typename value_t>
DI value_t compute_haversine(value_t x1, value_t y1, value_t x2, value_t y2)
{
  value_t sin_0 = sin(0.5 * (x1 - y1));
  value_t sin_1 = sin(0.5 * (x2 - y2));
  value_t rdist = sin_0 * sin_0 + cos(x1) * cos(y1) * sin_1 * sin_1;

  return 2 * asin(sqrt(rdist));
}

/**
 * @tparam value_idx data type of indices
 * @tparam value_t data type of values and distances
 * @tparam warp_q
 * @tparam thread_q
 * @tparam tpb
 * @param[out] out_inds output indices
 * @param[out] out_dists output distances
 * @param[in] index index array
 * @param[in] query query array
 * @param[in] n_index_rows number of rows in index array
 * @param[in] k number of closest neighbors to return
 */
template <typename value_idx, typename value_t, int warp_q = 1024, int thread_q = 8, int tpb = 128>
__global__ void haversine_knn_kernel(value_idx* out_inds,
                                     value_t* out_dists,
                                     const value_t* index,
                                     const value_t* query,
                                     size_t n_index_rows,
                                     int k)
{
  constexpr int kNumWarps = tpb / faiss::gpu::kWarpSize;

  __shared__ value_t smemK[kNumWarps * warp_q];
  __shared__ value_idx smemV[kNumWarps * warp_q];

  faiss::gpu::
    BlockSelect<value_t, value_idx, false, faiss::gpu::Comparator<value_t>, warp_q, thread_q, tpb>
      heap(faiss::gpu::Limits<value_t>::getMax(), -1, smemK, smemV, k);

  // Grid is exactly sized to rows available
  int limit = faiss::gpu::utils::roundDown(n_index_rows, faiss::gpu::kWarpSize);

  const value_t* query_ptr = query + (blockIdx.x * 2);
  value_t x1               = query_ptr[0];
  value_t x2               = query_ptr[1];

  int i = threadIdx.x;

  for (; i < limit; i += tpb) {
    const value_t* idx_ptr = index + (i * 2);
    value_t y1             = idx_ptr[0];
    value_t y2             = idx_ptr[1];

    value_t dist = compute_haversine(x1, y1, x2, y2);

    heap.add(dist, i);
  }

  // Handle last remainder fraction of a warp of elements
  if (i < n_index_rows) {
    const value_t* idx_ptr = index + (i * 2);
    value_t y1             = idx_ptr[0];
    value_t y2             = idx_ptr[1];

    value_t dist = compute_haversine(x1, y1, x2, y2);

    heap.addThreadQ(dist, i);
  }

  heap.reduce();

  for (int i = threadIdx.x; i < k; i += tpb) {
    out_dists[blockIdx.x * k + i] = smemK[i];
    out_inds[blockIdx.x * k + i]  = smemV[i];
  }
}

/**
 * Conmpute the k-nearest neighbors using the Haversine
 * (great circle arc) distance. Input is assumed to have
 * 2 dimensions (latitude, longitude) in radians.

 * @tparam value_idx
 * @tparam value_t
 * @param[out] out_inds output indices array on device (size n_query_rows * k)
 * @param[out] out_dists output dists array on device (size n_query_rows * k)
 * @param[in] index input index array on device (size n_index_rows * 2)
 * @param[in] query input query array on device (size n_query_rows * 2)
 * @param[in] n_index_rows number of rows in index array
 * @param[in] n_query_rows number of rows in query array
 * @param[in] k number of closest neighbors to return
 * @param[in] stream stream to order kernel launch
 */
template <typename value_idx, typename value_t>
void haversine_knn(value_idx* out_inds,
                   value_t* out_dists,
                   const value_t* index,
                   const value_t* query,
                   size_t n_index_rows,
                   size_t n_query_rows,
                   int k,
                   cudaStream_t stream)
{
  haversine_knn_kernel<<<n_query_rows, 128, 0, stream>>>(
    out_inds, out_dists, index, query, n_index_rows, k);
}

};  // namespace selection
};  // namespace raft
