/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "cuda_utils.h"

namespace MLCommon {
namespace LinAlg {
namespace Batched {

static constexpr int TileDim = 32;
static constexpr int BlockRows = 8;

// Ref: https://devblogs.nvidia.com/efficient-matrix-transpose-cuda-cc/
template <typename DataT, typename IdxT, typename EpilogueOp>
__global__ void symmKernel(DataT* out, const DataT* in, IdxT batchSize, IdxT n,
                           EpilogueOp op) {
  __shared__ DataT smem[TileDim][TileDim + 1];  // +1 to avoid bank conflicts
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
template <typename DataT, typename IdxT, typename EpilogueOp = Nop<DataT, IdxT>>
void make_symm(DataT* out, const DataT* in, IdxT batchSize, IdxT n,
               cudaStream_t stream, EpilogueOp op = Nop<DataT, IdxT>()) {
  dim3 blk(TileDim, BlockRows);
  auto nblks = ceildiv<int>(n, TileDim);
  dim3 grid(nblks, nblks, batchSize);
  symmKernel<DataT, IdxT, EpilogueOp><<<grid, blk, 0, stream>>>(
    out, in, batchSize, n, op);
  CUDA_CHECK(cudaGetLastError());
}

}  // end namespace Batched
}  // end namespace LinAlg
}  // end namespace MLCommon
