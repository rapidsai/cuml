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

#include <cub/cub.cuh>
#include "cuda_utils.h"
#include "vectorized.h"

namespace MLCommon {
namespace LinAlg {
namespace Batched {

/**
 * @brief Computes dot-product between 2 vectors each of which is stored in the
 *        registers of all participating threads
 * @tparam DataT data type
 * @tparam IdxT idx type
 * @tparam TPB threads per block
 * @tparam VecLen number of elements
 * @param x x vector
 * @param y y vector
 * @param len len of both vectors
 * @param smem dynamic shared memory needed for reduction. It must be atleast of
 *             size: `sizeof(BlockReduce::TempStorage)` if `broadcast` is false
 *             else, this size + sizeof(DataT).
 * @param broadcast only thread 0 will contain the final dot product if false,
 *                  else every thread will contain this value
 */
template <typename DataT, typename IdxT, int TPB, int VecLen>
DI DataT dotProduct(const DataT (&x)[VecLen], const DataT (&y)[VecLen],
                    IdxT len, char* smem, bool broadcast = false) {
  auto tid = threadIdx.x;
  auto val = DataT(0.0);
  if (tid < len) {
    #pragma unroll
    for (int i = 0; i < VecLen; ++i) {
      val += x[i] * y[i];
    }
  }
  typedef cub::BlockReduce<DataT, TPB> BlockReduce;
  typedef typename BlockReduce::TempStorage BlockReduceSmem;
  auto temp = *reinterpret_cast<BlockReduceSmem*>(smem);
  auto* sDot = reinterpret_cast<DataT>(smem + sizeof(BlockReduceSmem));
  auto dot = BlockReduce(temp).Sum(val);
  if (broadcast) {
    if (tid == 0) {
      sDot[0] = dot;
    }
    __syncthreads();
    dot = sDot[0];
  }
  return dot;
}

template <typename DataT, typename IdxT, int VecLenAx, int VecLenY, int TPB>
__global__ void gemvKernel(DataT* y, const DataT* A, const DataT* x, IdxT m,
                           IdxT n) {
  typedef TxN_t<DataT, VecLenAx> VecTypeAx;
  typedef TxN_t<DataT, VecLenY> VecTypeY;
  extern __shared__ char sdot[];
  VecTypeAx _x, _a;
  VecTypeY _y;
  IdxT idx = threadIdx.x * VecTypeAx::Ratio;
  IdxT batchOffset = blockIdx.x * m * n;
  _x.fill(DataT(0.0));
  if (idx < n) {
    _x.load(x, blockIdx.x * n + idx);
  }
  for (IdxT i = 0; i < m; i += VecTypeY::Ratio) {
    #pragma unroll
    for (IdxT j = 0; j < VecTypeY::Ratio; ++j) {
      _a.fill(DataT(0.0));
      if (idx < n) {
        _a.load(A, batchOffset + (i + j) * m + idx);
      }
      _y.data[j] = dotProduct<DataT, IdxT, TPB>(_a.data, _x.data, n, sdot,
                                                false);
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      _y.store(y, i * VecTypeY::Ratio);
    }
  }
}

template <typename DataT, typename IdxT, int VecLenAx, int VecLenY, int TPB>
void gemvImplY(DataT* y, const DataT* A, const DataT* x, IdxT m, IdxT n,
               IdxT batchSize, cudaStream_t stream) {
  typedef cub::BlockReduce<DataT, TPB> BlockReduce;
  typedef typename BlockReduce::TempStorage BlockReduceSmem;
  size_t smemSize = sizeof(BlockReduceSmem);
  gemvKernel<DataT, IdxT, VecLenAx, VecLenY, TPB>
    <<<batchSize, TPB, smemSize, stream>>>(y, A, x, m, n);
  CUDA_CHECK(cudaGetLastError());
}

template <typename DataT, typename IdxT, int VecLenAx, int TPB>
void gemvImplAx(DataT* y, const DataT* A, const DataT* x, IdxT m, IdxT n,
                IdxT batchSize, cudaStream_t stream) {
  size_t bytes = m * sizeof(DataT);
  if (16 / sizeof(DataT) && bytes % 16 == 0) {
    gemvImplY<DataT, IdxT, VecLenAx, 16 / sizeof(DataT), TPB>(
      y, A, x, m, n, batchSize, stream);
  } else if (8 / sizeof(DataT) && bytes % 8 == 0) {
    gemvImplY<DataT, IdxT, VecLenAx, 8 / sizeof(DataT), TPB>(
      y, A, x, m, n, batchSize, stream);
  } else if (4 / sizeof(DataT) && bytes % 4 == 0) {
    gemvImplY<DataT, IdxT, VecLenAx, 4 / sizeof(DataT), TPB>(
      y, A, x, m, n, batchSize, stream);
  } else if (2 / sizeof(DataT) && bytes % 2 == 0) {
    gemvImplY<DataT, IdxT, VecLenAx, 2 / sizeof(DataT), TPB>(
      y, A, x, m, n, batchSize, stream);
  } else if (1 / sizeof(DataT)) {
    gemvImplY<DataT, IdxT, VecLenAx, 1 / sizeof(DataT), TPB>(
      y, A, x, m, n, batchSize, stream);
  } else {
    gemvImplY<DataT, IdxT, VecLenAx, 1, TPB>(y, A, x, m, n, batchSize, stream);
  }
}

/**
 * @brief Per threadblock batched gemv. This works well when each of the input
 *        matrices in the batch are of same dimensions and small enough to fit
 *        in a single threadblock.
 * @tparam DataT data type
 * @tparam IdxT idx type
 * @tparam TPB threads per block
 * @param y the output vectors (dim = batchSize x m, row-major)
 * @param A input matrices (dim = batchSize x m x n, row-major)
 * @param x the input vectors (dim = batchSize x n, row-major)
 * @param m number of rows in A
 * @param n number of columns in A
 * @param batchSize batch size
 * @param stream cuda stream
 */
template <typename DataT, typename IdxT, int TPB>
void gemv(DataT* y, const DataT* A, const DataT* x, IdxT m, IdxT n,
          IdxT batchSize, cudaStream_t stream) {
  size_t bytes = n * sizeof(DataT);
  if (16 / sizeof(DataT) && bytes % 16 == 0) {
    gemvImplAx<DataT, IdxT, 16 / sizeof(DataT), TPB>(y, A, x, m, n, batchSize,
                                                     stream);
  } else if (8 / sizeof(DataT) && bytes % 8 == 0) {
    gemvImplAx<DataT, IdxT, 8 / sizeof(DataT), TPB>(y, A, x, m, n, batchSize,
                                                    stream);
  } else if (4 / sizeof(DataT) && bytes % 4 == 0) {
    gemvImplAx<DataT, IdxT, 4 / sizeof(DataT), TPB>(y, A, x, m, n, batchSize,
                                                    stream);
  } else if (2 / sizeof(DataT) && bytes % 2 == 0) {
    gemvImplAx<DataT, IdxT, 2 / sizeof(DataT), TPB>(y, A, x, m, n, batchSize,
                                                    stream);
  } else if (1 / sizeof(DataT)) {
    gemvImplAx<DataT, IdxT, 1 / sizeof(DataT), TPB>(y, A, x, m, n, batchSize,
                                                    stream);
  } else {
    gemvImplAx<DataT, IdxT, 1, TPB>(y, A, x, m, n, batchSize, stream);
  }
}

};  // end namespace Batched
};  // end namespace LinAlg
};  // end namespace MLCommon
