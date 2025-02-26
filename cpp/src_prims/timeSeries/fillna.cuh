/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include "jones_transform.cuh"

#include <cuml/tsa/arima_common.h>

#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <linalg/batched/matrix.cuh>

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
CUML_KERNEL void fillna_interpolate_kernel(T* data,
                                           int n_elem,
                                           FillnaTemp* d_indices_fwd,
                                           FillnaTemp* d_indices_bwd)
{
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n_elem;
       index += gridDim.x * blockDim.x) {
    if (isnan(data[index])) {
      FillnaTemp fwd = d_indices_fwd[index];
      FillnaTemp bwd = d_indices_bwd[n_elem - 1 - index];
      T value_fwd    = data[fwd.index];
      T value_bwd    = data[bwd.index];

      if (!fwd.is_valid) {
        data[index] = value_bwd;
      } else if (!bwd.is_valid) {
        data[index] = value_fwd;
      } else {
        T coef      = (T)(index - fwd.index) / (T)(bwd.index - fwd.index);
        data[index] = ((T)1 - coef) * value_fwd + coef * value_bwd;
      }
    }
  }
}

}  // namespace

namespace MLCommon {
namespace TimeSeries {

/**
 * Fill NaN values by interpolating between the last and next valid values
 *
 * @param[inout] data       Data which will be processed in-place
 * @param[in]    batch_size Number of series in the batch
 * @param[in]    n_obs      Number of observations per series
 * @param[in]    stream     CUDA stream
 */
template <typename T>
void fillna(T* data, int batch_size, int n_obs, cudaStream_t stream)
{
  rmm::device_uvector<FillnaTemp> indices_fwd(batch_size * n_obs, stream);
  rmm::device_uvector<FillnaTemp> indices_bwd(batch_size * n_obs, stream);
  FillnaTempMaker<true, T> transform_op_fwd(data, batch_size, n_obs);
  FillnaTempMaker<false, T> transform_op_bwd(data, batch_size, n_obs);
  thrust::counting_iterator<int> counting(0);
  FillnaOp scan_op;

  // Iterators wrapping the data with metadata (valid, first of its series)
  thrust::transform_iterator<FillnaTempMaker<true, T>,
                             thrust::counting_iterator<int>,
                             thrust::use_default,
                             FillnaTemp>
    itr_fwd(counting, transform_op_fwd);
  thrust::transform_iterator<FillnaTempMaker<false, T>,
                             thrust::counting_iterator<int>,
                             thrust::use_default,
                             FillnaTemp>
    itr_bwd(counting, transform_op_bwd);

  // Allocate temporary storage
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveScan(
    nullptr, temp_storage_bytes, itr_fwd, indices_fwd.data(), scan_op, batch_size * n_obs, stream);
  rmm::device_uvector<char> temp_storage(temp_storage_bytes, stream);
  void* d_temp_storage = (void*)temp_storage.data();

  // Execute scan (forward)
  cub::DeviceScan::InclusiveScan(d_temp_storage,
                                 temp_storage_bytes,
                                 itr_fwd,
                                 indices_fwd.data(),
                                 scan_op,
                                 batch_size * n_obs,
                                 stream);

  // Execute scan (backward)
  cub::DeviceScan::InclusiveScan(d_temp_storage,
                                 temp_storage_bytes,
                                 itr_bwd,
                                 indices_bwd.data(),
                                 scan_op,
                                 batch_size * n_obs,
                                 stream);

  const int TPB      = 256;
  const int n_blocks = raft::ceildiv<int>(n_obs * batch_size, TPB);

  // Interpolate valid values
  fillna_interpolate_kernel<false><<<n_blocks, TPB, 0, stream>>>(
    data, batch_size * n_obs, indices_fwd.data(), indices_bwd.data());
  RAFT_CUDA_TRY(cudaGetLastError());
}

}  // namespace TimeSeries
}  // namespace MLCommon
