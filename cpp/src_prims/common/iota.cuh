/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
