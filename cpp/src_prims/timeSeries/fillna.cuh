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

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <cuml/tsa/arima_common.h>
#include <raft/cudart_utils.h>
#include <cuml/common/device_buffer.hpp>
#include <linalg/batched/matrix.cuh>
#include <raft/cuda_utils.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/mr/device/allocator.hpp>
#include "jones_transform.cuh"

// Auxiliary functions in anonymous namespace
namespace {

struct FillnaTemp {
  /** After the scan, this index refers to the position of the last valid value */
  int index;
  /** This indicates whether a value is valid, i.e != NaN */
  bool is_valid;
  /** This indicates that this position is the first of a series and values from the previous series
   * in the batch cannot be used to fill missing observations */
  bool is_first;
};

// Functor for making the temp object from an index
template <bool forward, typename T>
struct FillnaTempMaker {
  const T* data;
  int batch_size;
  int n_obs;

  __host__ __device__ FillnaTempMaker(const T* data_, int batch_size_, int n_obs_)
    : data(data_), batch_size(batch_size_), n_obs(n_obs_)
  {
  }

  __host__ __device__ __forceinline__ FillnaTemp operator()(const int& index) const
  {
    if (forward)
      return {index, !isnan(data[index]), index % n_obs == 0};
    else {
      int index_bwd = batch_size * n_obs - 1 - index;
      return {index_bwd, !isnan(data[index_bwd]), index % n_obs == 0};
    }
  }
};

struct FillnaOp {
  __host__ __device__ __forceinline__ FillnaTemp operator()(const FillnaTemp& lhs,
                                                            const FillnaTemp& rhs) const
  {
    return (rhs.is_first || rhs.is_valid) ? rhs : lhs;
  }
};

template <bool forward, typename T>
__global__ void fillna_broadcast_kernel(T* data, int n_elem, FillnaTemp* d_indices)
{
  for (int index0 = blockIdx.x * blockDim.x + threadIdx.x; index0 < n_elem;
       index0 += gridDim.x * blockDim.x) {
    int index1     = forward ? index0 : n_elem - 1 - index0;
    int from_index = d_indices[index0].index;
    if (from_index != index1) data[index1] = data[from_index];
  }
}

}  // namespace

namespace MLCommon {
namespace TimeSeries {

/**
 * Fill NaN values naively with the last known value in each pass,
 * with first a forward pass followed by a backward pass.
 *
 * @param[inout] data       Data which will be processed in-place
 * @param[in]    batch_size Number of series in the batch
 * @param[in]    n_obs      Number of observations per series
 * @param[in]    allocator  Device memory allocator
 * @param[in]    stream     CUDA stream
 */
template <typename T>
void fillna(T* data,
            int batch_size,
            int n_obs,
            std::shared_ptr<raft::mr::device::allocator> allocator,
            cudaStream_t stream)
{
  MLCommon::device_buffer<FillnaTemp> indices(allocator, stream, batch_size * n_obs);
  FillnaTempMaker<true, T> transform_op_fwd(data, batch_size, n_obs);
  FillnaTempMaker<false, T> transform_op_bwd(data, batch_size, n_obs);
  cub::CountingInputIterator<int> counting(0);
  FillnaOp scan_op;

  // Iterators wrapping the data with metadata (valid, first of its series)
  cub::TransformInputIterator<FillnaTemp, FillnaTempMaker<true, T>, cub::CountingInputIterator<int>>
    itr_fwd(counting, transform_op_fwd);
  cub::
    TransformInputIterator<FillnaTemp, FillnaTempMaker<false, T>, cub::CountingInputIterator<int>>
      itr_bwd(counting, transform_op_bwd);

  // Allocate temporary storage
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveScan(
    nullptr, temp_storage_bytes, itr_fwd, indices.data(), scan_op, batch_size * n_obs, stream);
  MLCommon::device_buffer<char> temp_storage(allocator, stream, temp_storage_bytes);
  void* d_temp_storage = (void*)temp_storage.data();

  // Execute scan (forward)
  cub::DeviceScan::InclusiveScan(d_temp_storage,
                                 temp_storage_bytes,
                                 itr_fwd,
                                 indices.data(),
                                 scan_op,
                                 batch_size * n_obs,
                                 stream);

  const int TPB      = 256;
  const int n_blocks = raft::ceildiv<int>(n_obs * batch_size, TPB);

  // Broadcast last valid values to missing values (forward)
  fillna_broadcast_kernel<true>
    <<<n_blocks, TPB, 0, stream>>>(data, batch_size * n_obs, indices.data());
  CUDA_CHECK(cudaPeekAtLastError());

  // Execute scan (backward)
  cub::DeviceScan::InclusiveScan(
    d_temp_storage, temp_storage_bytes, itr_bwd, indices.data(), scan_op, batch_size * n_obs);

  // Broadcast last valid values to missing values (backward)
  fillna_broadcast_kernel<false>
    <<<n_blocks, TPB, 0, stream>>>(data, batch_size * n_obs, indices.data());
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace TimeSeries
}  // namespace MLCommon