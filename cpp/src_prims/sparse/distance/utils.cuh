/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cuml/common/logger.hpp>

#include <raft/mr/device/allocator.hpp>
#include <raft/mr/device/buffer.hpp>

#include <cub/cub.cuh>

namespace raft {
namespace sparse {
namespace distance {

/**
 * @tparam value_idx
 * @param out
 * @param in
 * @param n
 */
template <typename value_idx, bool count_greater_than>
__global__ void max_kernel(value_idx *out, const value_idx *in,
                           value_idx *out_count, value_idx n,
                           value_idx greater_than) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  typedef cub::BlockReduce<value_idx, 256> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  auto degree = in[tid] - in[tid - 1];
  value_idx v = tid < n ? degree : 0;
  value_idx agg = BlockReduce(temp_storage).Reduce(v, cub::Max());

  if (count_greater_than) {
    bool predicate = tid < n && degree > greater_than;
    value_idx count = __syncthreads_count(predicate);

    if (threadIdx.x == 0) atomicAdd(out_count, count);
  }

  if (threadIdx.x == 0) atomicMax(out, agg);
}

template <typename value_idx, bool count_greater_than>
inline std::pair<value_idx, value_idx> max_degree(
  const value_idx *indptr, const value_idx n_rows,
  std::shared_ptr<raft::mr::device::allocator> allocator,
  const cudaStream_t stream, value_idx greater_than = 0) {
  raft::mr::device::buffer<value_idx> max_d(allocator, stream, 1);
  CUDA_CHECK(cudaMemsetAsync(max_d.data(), 0, sizeof(value_idx), stream));

  /**
   * A custom max reduction is performed until https://github.com/rapidsai/cuml/issues/3431
   * is fixed.
   */
  value_idx count_h;
  if (count_greater_than) {
    raft::mr::device::buffer<value_idx> count_d(allocator, stream, 1);
    CUDA_CHECK(cudaMemsetAsync(count_d.data(), 0, sizeof(value_idx), stream));

    max_kernel<value_idx, true><<<raft::ceildiv(n_rows, 256), 256, 0, stream>>>(
      max_d.data(), indptr + 1, count_d.data(), n_rows, greater_than);

    raft::update_host(&count_h, count_d.data(), 1, stream);
  } else {
    max_kernel<value_idx, false>
      <<<raft::ceildiv(n_rows, 256), 256, 0, stream>>>(
        max_d.data(), indptr + 1, nullptr, n_rows, greater_than);
  }

  value_idx max_h;
  raft::update_host(&max_h, max_d.data(), 1, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUML_LOG_DEBUG("max nnz: %d", max_h);

  return std::make_pair(std::move(max_h), std::move(count_h));
}

}  // namespace distance
}  // namespace sparse
}  // namespace raft
