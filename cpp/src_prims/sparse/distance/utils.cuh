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

#include <rmm/device_vector.hpp>

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

  value_idx v;
  if (tid < n) {
    v = in[tid] - in[tid - 1];
  } else {
    v = 0;
  }
  value_idx agg = BlockReduce(temp_storage).Reduce(v, cub::Max());

  if (count_greater_than) {
    bool predicate = tid < n && v > greater_than;
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
    CUML_LOG_DEBUG("count nnz: %d", count_h);
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

template <typename T>
void printv(rmm::device_vector<T> &vec, const std::string &name = "",
            const size_t displ = 5) {
  std::cout.precision(15);
  std::cout << name << " size = " << vec.size() << std::endl;
  thrust::copy(vec.begin(), vec.end(),
               std::ostream_iterator<T>(std::cout, " "));
  std::cout << std::endl << std::endl;
}

/**
 * Computes the maximum number of columns that can be stored
 * in shared memory in dense form with the given block size
 * and precision.
 * @return the maximum number of columns that can be stored in smem
 */
template <typename value_idx, typename value_t, int tpb = 1024>
inline int max_cols_per_block() {
  // max cols = (total smem available - cub reduction smem)
  return (raft::getSharedMemPerBlock() -
          ((tpb / raft::warp_size()) * sizeof(value_t))) /
         sizeof(value_t);
}

}  // namespace distance
}  // namespace sparse
}  // namespace raft
