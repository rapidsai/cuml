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

#include <cuml/common/utils.hpp>

#include <raft/util/cuda_utils.cuh>

namespace MLCommon {
namespace LinAlg {
namespace Batched {

static constexpr int TileDim   = 32;
static constexpr int BlockRows = 8;

// Ref: https://devblogs.nvidia.com/efficient-matrix-transpose-cuda-cc/
///@todo: special-case for blockIdx.x == blockIdx.y to reduce gmem traffic
template <typename DataT, typename IdxT, typename EpilogueOp>
CUML_KERNEL void symmKernel(DataT* out, const DataT* in, IdxT batchSize, IdxT n, EpilogueOp op)
{
  __shared__ DataT smem[TileDim][TileDim + 1];  // +1 to avoid bank conflicts
  IdxT batchOffset = blockIdx.z * n * n;
  IdxT myRowStart  = blockIdx.y * TileDim + threadIdx.y;
  IdxT myColStart  = blockIdx.x * TileDim + threadIdx.x;
  IdxT myIdx       = batchOffset + myRowStart * n + myColStart;
  // load the transpose part
  IdxT otherRowStart = blockIdx.x * TileDim + threadIdx.y;
  IdxT otherColStart = blockIdx.y * TileDim + threadIdx.x;
  IdxT otherIdx      = batchOffset + otherRowStart * n + otherColStart;
  if (otherColStart < n) {
#pragma unroll
    for (int i = 0; i < TileDim; i += BlockRows) {
      if (otherRowStart + i < n) { smem[threadIdx.y + i][threadIdx.x] = in[otherIdx + i * n]; }
    }
  }
  __syncthreads();
  if (myColStart < n) {
#pragma unroll
    for (int i = 0; i < TileDim; i += BlockRows) {
      auto offset = myIdx + i * n;
      if (myRowStart + i < n) {
        auto sum    = smem[threadIdx.x][threadIdx.y + i] + in[offset];
        out[offset] = op(sum * DataT(0.5), offset);
      }
    }
  }
}

/**
 * @brief An out-of-place batched matrix symmetrizer. In other words, given
 *        a bunch of square matrices Ai, it computes (Ai + Ai') / 2.
 * @tparam DataT data type
 * @tparam IdxT index type
 * @tparam EpilogueOp any custom operation before storing the elements
 * @param out the output symmetric matrices (dim = batchSize x n x n, row-major)
 * @param in the input square matrices (dim = batchSize x n x n, row-major)
 * @param batchSize number of such matrices
 * @param n dimension of each square matrix
 * @param stream cuda stream
 * @param op custom epilogue functor
 */
template <typename DataT, typename IdxT, typename EpilogueOp = raft::identity_op>
void make_symm(DataT* out,
               const DataT* in,
               IdxT batchSize,
               IdxT n,
               cudaStream_t stream,
               EpilogueOp op = raft::identity_op())
{
  dim3 blk(TileDim, BlockRows);
  auto nblks = raft::ceildiv<int>(n, TileDim);
  dim3 grid(nblks, nblks, batchSize);
  symmKernel<DataT, IdxT, EpilogueOp><<<grid, blk, 0, stream>>>(out, in, batchSize, n, op);
  RAFT_CUDA_TRY(cudaGetLastError());
}

}  // end namespace Batched
}  // end namespace LinAlg
}  // end namespace MLCommon
