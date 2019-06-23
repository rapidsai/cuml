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
#include "vectorized.h"

namespace MLCommon {

template <typename DataT, int VecLen, typename Lambda, typename IdxT>
__global__ void scatterKernel(DataT *out, const DataT *in, const IdxT *idx,
                              IdxT len, Lambda op) {
  typedef TxN_t<DataT, VecLen> DataVec;
  typedef TxN_t<IdxT, VecLen> IdxVec;
  IdxT tid = threadIdx.x + ((IdxT)blockIdx.x * blockDim.x);
  tid *= VecLen;
  if (tid >= len) return;
  IdxVec idxIn;
  idxIn.load(idx, tid);
  DataVec dataIn;
#pragma unroll
  for (int i = 0; i < VecLen; ++i) {
    auto inPos = idxIn.val.data[i];
    dataIn.val.data[i] = op(in[inPos], tid + i);
  }
  dataIn.store(out, tid);
}

template <typename DataT, int VecLen, typename Lambda, typename IdxT, int TPB>
void scatterImpl(DataT *out, const DataT *in, const IdxT *idx, IdxT len,
                 Lambda op, cudaStream_t stream) {
  const IdxT nblks = ceildiv(VecLen ? len / VecLen : len, (IdxT)TPB);
  scatterKernel<DataT, VecLen, Lambda, IdxT>
    <<<nblks, TPB, 0, stream>>>(out, in, idx, len, op);
  CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Performs scatter operation based on the input indexing array
 * @tparam DataT data type whose array gets scattered
 * @tparam IdxT indexing type
 * @tparam Lambda the device-lambda performing a unary operation on the loaded
 * data before it gets scattered
 * @tparam TPB threads-per-block in the final kernel launched
 * @param out the output array
 * @param in the input array
 * @param idx the indexing array
 * @param len number of elements in the input array
 * @param op the device-lambda
 * @param stream cuda stream where to launch work
 */
template <typename DataT, typename IdxT, typename Lambda, int TPB = 256>
void scatter(DataT *out, const DataT *in, const IdxT *idx, IdxT len, Lambda op,
             cudaStream_t stream) {
  if (len <= 0) return;
  size_t maxPerElem = max(sizeof(DataT), sizeof(IdxT));
  size_t bytes = len * maxPerElem;
  uint64_t dataAddr = uint64_t(in);
  uint64_t idxAddr = uint64_t(idx);
  if (16 / maxPerElem && bytes % 16 == 0) {
    scatterImpl<DataT, 16 / maxPerElem, Lambda, IdxT, TPB>(out, in, idx, len,
                                                           op, stream);
  } else if (8 / maxPerElem && bytes % 8 == 0) {
    unaryOpImpl<DataT, 8 / maxPerElem, Lambda, IdxT, TPB>(out, in, idx, len,
                                                          op, stream);
  } else if (4 / maxPerElem && bytes % 4 == 0) {
    unaryOpImpl<DataT, 4 / maxPerElem, Lambda, IdxT, TPB>(out, in, idx, len,
                                                          op, stream);
  } else if (2 / maxPerElem && bytes % 2 == 0) {
    unaryOpImpl<DataT, 2 / maxPerElem, Lambda, IdxT, TPB>(out, in, idx, len,
                                                          op, stream);
  } else if (1 / maxPerElem) {
    unaryOpImpl<DataT, 1 / maxPerElem, Lambda, IdxT, TPB>(out, in, idx, len,
                                                          op, stream);
  } else {
    unaryOpImpl<DataT, 1, Lambda, IdxT, TPB>(out, in, idx, len, op, stream);
  }
}

} // end namespace MLCommon
