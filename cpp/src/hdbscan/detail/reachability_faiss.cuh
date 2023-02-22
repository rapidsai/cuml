/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file thirdparty/LICENSES/LICENSE.faiss
 */

/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <raft/distance/distance.cuh>
#include <raft/util/pow2_utils.cuh>
#include <raft/spatial/knn/detail/faiss_select/DistanceUtils.h>
#include <raft/spatial/knn/detail/faiss_select/Select.cuh>
#include <raft/spatial/knn/detail/selection_faiss.cuh>
#include <thrust/iterator/transform_iterator.h>

#include <raft/core/handle.hpp>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>

#include <cstddef>

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Reachability {

template <typename value_t, int NumWarpQ, int NumThreadQ, int ThreadsPerBlock>
__global__ void l2SelectMinK(const value_t* pairwise_distances,
                             const value_t* core_dists,
                             value_t* out_dists,
                             int* out_inds,
                             int n_rows,
                             int n_cols,
                             int batch_offset_i,
                             int batch_offset_j,
                             int k,
                             value_t initK,
                             value_t alpha)
{
  // Each block handles a single row of the distances (results)
  constexpr int kNumWarps = ThreadsPerBlock / 32;

  __shared__ value_t smemK[kNumWarps * NumWarpQ];
  __shared__ int smemV[kNumWarps * NumWarpQ];

  using namespace raft::spatial::knn::detail::faiss_select;

  BlockSelect<value_t, int, false, Comparator<key_t>, NumWarpQ, NumThreadQ, ThreadsPerBlock> heap(
    initK, -1, smemK, smemV, k);

  // Grid is exactly sized to rows available
  int row = blockIdx.x;

  // Whole warps must participate in the selection
  int limit = raft::Pow2<raft::WarpSize>::roundDown(n_cols);
  int i     = threadIdx.x;

  for (; i < limit; i += blockDim.x) {
    value_t v = pairwise_distances[row * n_cols + i];
    v = max(core_dists[i + batch_offset_j], max(core_dists[row + batch_offset_i], alpha * v));
    heap.add(v, i);
  }

  if (i < n_cols) {
    value_t v = pairwise_distances[row * n_cols + i];
    v = max(core_dists[i + batch_offset_j], max(core_dists[row + batch_offset_i], alpha * v));
    heap.addThreadQ(v, i);
  }

  heap.reduce();
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    out_dists[row * k + i] = smemK[i];
    out_inds[row * k + i]  = smemV[i];
  }
}

/**
 * Computes expanded L2 metric, projects points into reachability
 * space, and performs a k-select.
 * @tparam value_t
 * @param[in] productDistances Tensor (or blocked view) of inner products
 * @param[in] centroidDistances Tensor of l2 norms
 * @param[in] coreDistances Tensor of core distances
 * @param[out] outDistances Tensor of output distances
 * @param[out] outIndices Tensor of output indices
 * @param[in] batch_offset starting row (used when productDistances is a batch)
 * @param[in] k number of neighbors to select
 * @param[in] stream cuda stream for ordering gpu computations
 */
template <typename value_t>
void runL2SelectMin(const value_t* pairwise_distances,
                    const value_t* core_dists,
                    value_t* out_dists,
                    int* out_inds,
                    int n_rows,
                    int n_cols,
                    int batch_offset_i,
                    int batch_offset_j,
                    int k,
                    value_t alpha,
                    cudaStream_t stream)
{
  auto grid = dim3(n_rows);

#define RUN_L2_SELECT(BLOCK, NUM_WARP_Q, NUM_THREAD_Q)                           \
  do {                                                                           \
    l2SelectMinK<value_t, NUM_WARP_Q, NUM_THREAD_Q, BLOCK>                       \
      <<<grid, BLOCK, 0, stream>>>(pairwise_distances,                           \
                                   core_dists,                                   \
                                   out_dists,                                    \
                                   out_inds,                                     \
                                   n_rows,                                       \
                                   n_cols,                                       \
                                   batch_offset_i,                               \
                                   batch_offset_j,                               \
                                   k,                                            \
                                   std::numeric_limits<value_t>::max(),          \
                                   alpha);                                       \
  } while (0)

  // block size 128 for everything <= 1024
  if (k <= 32) {
    RUN_L2_SELECT(128, 32, 2);
  } else if (k <= 64) {
    RUN_L2_SELECT(128, 64, 3);
  } else if (k <= 128) {
    RUN_L2_SELECT(128, 128, 3);
  } else if (k <= 256) {
    RUN_L2_SELECT(128, 256, 4);
  } else if (k <= 512) {
    RUN_L2_SELECT(128, 512, 8);
  } else if (k <= 1024) {
    RUN_L2_SELECT(128, 1024, 8);
  } else if (k <= 2048) {
    // smaller block for less shared memory
    RUN_L2_SELECT(64, 2048, 8);
  } else {
    ASSERT(false, "K of %d is unsupported", k);
  }
}

/**
 * Given core distances, Fuses computations of L2 distances between all
 * points, projection into mutual reachability space, and k-selection.
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle for resource reuse
 * @param[out] out_inds  output indices array (size m * k)
 * @param[out] out_dists output distances array (size m * k)
 * @param[in] X input data points (size m * n)
 * @param[in] m number of rows in X
 * @param[in] n number of columns in X
 * @param[in] k neighborhood size (includes self-loop)
 * @param[in] core_dists array of core distances (size m)
 */
template <typename value_idx, typename value_t>
void mutual_reachability_knn_l2(const raft::handle_t& handle,
                                value_idx* out_inds,
                                value_t* out_dists,
                                const value_t* X,
                                size_t m,
                                size_t n,
                                int k,
                                value_t* core_dists,
                                value_t alpha)
{
  // Figure out the number of rows/cols to tile for
  size_t tile_rows   = 0;
  size_t tile_cols   = 0;
  auto stream        = handle.get_stream();
  auto device_memory = handle.get_workspace_resource();
  auto total_mem     = device_memory->get_mem_info(stream).second;
  raft::spatial::knn::detail::faiss_select::chooseTileSize(
    m, n, n, sizeof(value_t), total_mem, tile_rows, tile_cols);

  // tile_cols must be at least k items
  tile_cols = std::max(tile_cols, static_cast<size_t>(k));

  // stores pairwise distances for the current tile
  rmm::device_uvector<value_t> temp_distances(tile_rows * tile_cols, stream);

  // if we're tiling over columns, we need additional buffers for temporary output
  // distances/indices
  size_t num_col_tiles = raft::ceildiv(n, tile_cols);
  size_t temp_out_cols = k * num_col_tiles;

  // the final column tile could have less than 'k' items in it
  // in which case the number of columns here is too high in the temp output.
  // adjust if necessary
  auto last_col_tile_size = n % tile_cols;
  if (last_col_tile_size && (last_col_tile_size < static_cast<size_t>(k))) {
    temp_out_cols -= k - last_col_tile_size;
  }
  rmm::device_uvector<value_t> temp_out_distances(tile_rows * temp_out_cols, stream);
  rmm::device_uvector<value_idx> temp_out_indices(tile_rows * temp_out_cols, stream);

  for (size_t i = 0; i < m; i += tile_rows) {
    size_t current_query_size = std::min(tile_rows, m - i);

    for (size_t j = 0; j < n; j += tile_cols) {
      size_t current_centroid_size = std::min(tile_cols, n - j);
      size_t current_k             = std::min(current_centroid_size, static_cast<size_t>(k));

      raft::distance::pairwise_distance<value_t, int>(handle,
                                                    X + i * n,
                                                    X + j * n,
                                                    temp_distances.data(),
                                                    current_query_size,
                                                    current_centroid_size,
                                                    n,
                                                    raft::distance::DistanceType::L2Expanded);

      runL2SelectMin(temp_distances.data(),
                     core_dists,
                     out_dists + i * k,
                     out_inds + i * k,
                     (int) current_query_size,
                     (int) current_centroid_size,
                     i,
                     j,
                     current_k,
                     alpha,
                     stream);

      // if we're tiling over columns, we need to do a couple things to fix up
      // the output of select_k
      // 1. The column id's in the output are relative to the tile, so we need
      // to adjust the column ids by adding the column the tile starts at (j)
      // 2. select_k writes out output in a row-major format, which means we
      // can't just concat the output of all the tiles and do a select_k on the
      // concatenation.
      // Fix both of these problems in a single pass here
      if (tile_cols != n) {
        const value_t* in_distances = out_dists + i * k;
        const value_idx* in_indices     = out_inds + i * k;
        value_t* out_distances      = temp_out_distances.data();
        value_idx* out_indices          = temp_out_indices.data();

        auto count = thrust::make_counting_iterator<value_idx>(0);
        thrust::for_each(handle.get_thrust_policy(),
                         count,
                         count + current_query_size * current_k,
                         [=] __device__(value_idx i) {
                           value_idx row = i / current_k, col = i % current_k;
                           value_idx out_index = row * temp_out_cols + j * k / tile_cols + col;

                           out_distances[out_index] = in_distances[i];
                           out_indices[out_index]   = in_indices[i] + j;
                         });
      }
    }

    if (tile_cols != n) {
      // select the actual top-k items here from the temporary output
      raft::spatial::knn::detail::select_k<value_idx, value_t>(temp_out_distances.data(),
                                                                   temp_out_indices.data(),
                                                                   current_query_size,
                                                                   temp_out_cols,
                                                                   out_dists + i * k,
                                                                   out_inds + i * k,
                                                                   true,
                                                                   k,
                                                                   stream);
    }
  }
}

};  // end namespace Reachability
};  // end namespace detail
};  // end namespace HDBSCAN
};  // end namespace ML
