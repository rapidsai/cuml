/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <raft/util/cuda_utils.cuh>

namespace MLCommon {

template <typename DataT, typename IdxT>
CUML_KERNEL void iotaKernel(DataT* out, DataT start, DataT step, IdxT len)
{
  auto tid = (IdxT)blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < len) { out[tid] = start + DataT(tid) * step; }
}

/**
 * @brief GPU version of std::iota
 * @tparam DataT data type
 * @tparam IdxT indexing arithmetic type
 * @param out the output array
 * @param start start value in the array
 * @param step step size for each successive locations in the array
 * @param len the array length
 * @param stream cuda stream
 */
template <typename DataT, typename IdxT>
void iota(DataT* out, DataT start, DataT step, IdxT len, cudaStream_t stream)
{
  static const int TPB = 512;
  IdxT nblks           = raft::ceildiv<IdxT>(len, TPB);
  iotaKernel<DataT, IdxT><<<nblks, TPB, 0, stream>>>(out, start, step, len);
  RAFT_CUDA_TRY(cudaGetLastError());
}

}  // namespace MLCommon
