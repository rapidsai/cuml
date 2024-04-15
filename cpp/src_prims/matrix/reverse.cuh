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
#include <raft/util/vectorized.cuh>

namespace MLCommon {
namespace Matrix {

template <typename math_t, int veclen_, typename Lambda>
CUML_KERNEL void reverseKernel(math_t* out,
                               const math_t* in,
                               int nrows,
                               int ncols,
                               bool rowMajor,
                               bool alongRows,
                               int len,
                               Lambda op)
{
  typedef raft::TxN_t<math_t, veclen_> VecType;
  int idx = (threadIdx.x + (blockIdx.x * blockDim.x)) * VecType::Ratio;
  if (idx >= len) return;
  int srcIdx, dstIdx;
  if (!rowMajor && !alongRows) {
    int srcRow = idx % nrows;
    int srcCol = idx / nrows;
    int dstRow = srcRow;
    int dstCol = ncols - srcCol - 1;
    srcIdx     = idx;
    dstIdx     = dstCol * nrows + dstRow;
  } else if (!rowMajor && alongRows) {
    int mod    = raft::ceildiv(nrows, 2);
    int srcRow = idx % mod;
    int srcCol = idx / mod;
    int dstRow = nrows - srcRow - VecType::Ratio;
    int dstCol = srcCol;
    srcIdx     = srcCol * nrows + srcRow;
    dstIdx     = dstCol * nrows + dstRow;
  } else if (rowMajor && !alongRows) {
    int mod    = raft::ceildiv(ncols, 2);
    int srcRow = idx / mod;
    int srcCol = idx % mod;
    int dstRow = srcRow;
    int dstCol = ncols - srcCol - VecType::Ratio;
    srcIdx     = srcCol + srcRow * ncols;
    dstIdx     = dstCol + dstRow * ncols;
  } else {
    int srcRow = idx / ncols;
    int srcCol = idx % ncols;
    int dstRow = nrows - srcRow - 1;
    int dstCol = srcCol;
    srcIdx     = idx;
    dstIdx     = dstCol + dstRow * ncols;
  }
  VecType a, b;
  a.load(in, srcIdx);
  b.load(in, dstIdx);
  // while reversing along coalesced dimension, also reverse the elements
  if ((rowMajor && !alongRows) || (!rowMajor && alongRows)) {
#pragma unroll
    for (int i = 0; i < VecType::Ratio; ++i) {
      raft::swapVals(a.val.data[i], a.val.data[VecType::Ratio - i - 1]);
      raft::swapVals(b.val.data[i], b.val.data[VecType::Ratio - i - 1]);
    }
  }
#pragma unroll
  for (int i = 0; i < VecType::Ratio; ++i) {
    a.val.data[i] = op(a.val.data[i]);
    b.val.data[i] = op(b.val.data[i]);
  }
  a.store(out, dstIdx);
  b.store(out, srcIdx);
}

template <typename math_t, int veclen_, typename Lambda, int TPB>
void reverseImpl(math_t* out,
                 const math_t* in,
                 int nrows,
                 int ncols,
                 bool rowMajor,
                 bool alongRows,
                 Lambda op,
                 cudaStream_t stream)
{
  int len         = alongRows ? raft::ceildiv(nrows, 2) * ncols : nrows * raft::ceildiv(ncols, 2);
  const int nblks = raft::ceildiv(veclen_ ? len / veclen_ : len, TPB);
  reverseKernel<math_t, veclen_, Lambda>
    <<<nblks, TPB, 0, stream>>>(out, in, nrows, ncols, rowMajor, alongRows, len, op);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * @brief Reversal of the input matrix along the specified dimension
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam Lambda the device-lambda performing the actual operation
 * @tparam TPB threads-per-block in the final kernel launched
 * @param out the output matrix (supports inplace operation)
 * @param in the input matrix
 * @param nrows number of rows in the input matrix
 * @param ncols number of cols in the input matrix
 * @param rowMajor input matrix is row major or not
 * @param alongRows whether to reverse along rows or not
 * @param stream cuda stream where to launch work
 * @param op the device-lambda to perform an optional final unary operation on
 *  each element after the reverse
 */
template <typename math_t, typename Lambda = raft::identity_op<math_t>, int TPB = 256>
void reverse(math_t* out,
             const math_t* in,
             int nrows,
             int ncols,
             bool rowMajor,
             bool alongRows,
             cudaStream_t stream,
             Lambda op = raft::identity_op<math_t>())
{
  size_t bytes = (rowMajor ? ncols : nrows) * sizeof(math_t);
  if (16 / sizeof(math_t) && bytes % 16 == 0) {
    reverseImpl<math_t, 16 / sizeof(math_t), Lambda, TPB>(
      out, in, nrows, ncols, rowMajor, alongRows, op, stream);
  } else if (8 / sizeof(math_t) && bytes % 8 == 0) {
    reverseImpl<math_t, 8 / sizeof(math_t), Lambda, TPB>(
      out, in, nrows, ncols, rowMajor, alongRows, op, stream);
  } else if (4 / sizeof(math_t) && bytes % 4 == 0) {
    reverseImpl<math_t, 4 / sizeof(math_t), Lambda, TPB>(
      out, in, nrows, ncols, rowMajor, alongRows, op, stream);
  } else if (2 / sizeof(math_t) && bytes % 2 == 0) {
    reverseImpl<math_t, 2 / sizeof(math_t), Lambda, TPB>(
      out, in, nrows, ncols, rowMajor, alongRows, op, stream);
  } else if (1 / sizeof(math_t)) {
    reverseImpl<math_t, 1 / sizeof(math_t), Lambda, TPB>(
      out, in, nrows, ncols, rowMajor, alongRows, op, stream);
  } else {
    reverseImpl<math_t, 1, Lambda, TPB>(out, in, nrows, ncols, rowMajor, alongRows, op, stream);
  }
}

};  // end namespace Matrix
};  // end namespace MLCommon
