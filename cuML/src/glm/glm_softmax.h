/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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
#include "glm/glm_base.h"
#include "linalg/binary_op.h"
#include <glm/glm_vectors.h>

namespace ML {
namespace GLM {
using MLCommon::ceildiv;
using MLCommon::myExp;
using MLCommon::myLog;
using MLCommon::myMax;

//Input: matrix Z (dims: CxN)
//Computes softmax cross entropy loss across columns, i.e. normalization column-wise.
//
//This kernel performs best for small number of classes C.
//It's much faster than implementation based on ml-prims (up to ~2x - ~10x for small C <= BX).
//More importantly, it does not require another CxN scratch space.
//In that case the block covers the whole column and warp reduce is fast
//TODO for very large C, there should be maybe rather something along the lines of
//     coalesced reduce, i.e. blocks should take care of columns
//TODO split into two kernels for small and large case?
template <typename T, int BX = 32, int BY = 8>
__global__ void logSoftmaxKernel(T *out, T *dZ, const T *in, const T *labels,
                                 int C, int N, bool getDerivative = true) {
  typedef cub::WarpReduce<T, BX> WarpRed;
  typedef cub::BlockReduce<T, BX, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BY>
      BlockRed;

  __shared__ union {
    typename WarpRed::TempStorage warpStore[BY];
    typename BlockRed::TempStorage blockStore;
    T sh_val[BY];
  } shm;

  int y = threadIdx.y + blockIdx.x * BY;
  int len = C * N;

  bool delta = false;
  // TODO is there a better way to read this?
  if (getDerivative && threadIdx.x == 0) {
    shm.sh_val[threadIdx.y] = labels[y];
  }
  __syncthreads();
  T label = shm.sh_val[threadIdx.y];
  __syncthreads();
  T eta_y = 0;
  T myEta = 0;
  T etaMax = -1e9;
  T lse = 0;
  /*
   * Phase 1: Find Maximum m over column
   */
  for (int x = threadIdx.x; x < C; x += BX) {
    int idx = x + y * C;
    if (x < C && idx < len) {
      myEta = in[idx];
      if (x == label) {
        delta = true;
        eta_y = myEta;
      }
      etaMax = myMax<T>(myEta, etaMax);
    }
  }
  T tmpMax = WarpRed(shm.warpStore[threadIdx.y]).Reduce(etaMax, cub::Max());
  if (threadIdx.x == 0) {
    shm.sh_val[threadIdx.y] = tmpMax;
  }
  __syncthreads();
  etaMax = shm.sh_val[threadIdx.y];
  __syncthreads();

  /*
   * Phase 2: Compute stabilized log-sum-exp over column
   * lse = m + log(sum(exp(eta - m)))
   */
  // TODO there must be a better way to do this...
  if (C <= BX) { // this means one block covers a column and myEta is valid
    int idx = threadIdx.x + y * C;
    if (threadIdx.x < C && idx < len) {
      lse = myExp<T>(myEta - etaMax);
    }
  } else {
    for (int x = threadIdx.x; x < C; x += BX) {
      int idx = x + y * C;
      if (x < C && idx < len) {
        lse += myExp<T>(in[idx] - etaMax);
      }
    }
  }
  T tmpLse = WarpRed(shm.warpStore[threadIdx.y]).Sum(lse);
  if (threadIdx.x == 0) {
    shm.sh_val[threadIdx.y] = etaMax + myLog<T>(tmpLse);
  }
  __syncthreads();
  lse = shm.sh_val[threadIdx.y];
  __syncthreads();

  /*
   * Phase 3: Compute derivatives dL/dZ = P - delta_y
   * P is the softmax distribution, delta_y the kronecker delta for the class of
   * label y If we getDerivative=false, dZ will just contain P, which might be
   * useful
   */

  if (C <= BX) { // this means one block covers a column and myEta is valid
    int idx = threadIdx.x + y * C;
    if (threadIdx.x < C && idx < len) {
      dZ[idx] = (myExp<T>(myEta - lse) -
                 (getDerivative ? (threadIdx.x == label) : T(0)));
    }
  } else {
    for (int x = threadIdx.x; x < C; x += BX) {
      int idx = x + y * C;
      if (x < C && idx < len) {
        T logP = in[idx] - lse;
        dZ[idx] = (myExp<T>(logP) - (getDerivative ? (x == label) : T(0)));
      }
    }
  }

  if (!getDerivative) // no need to continue, lossval will be undefined
    return;

  T lossVal = 0;
  if (delta) {
    lossVal = (lse - eta_y) / N;
  }

  /*
   * Phase 4: accumulate loss value
   */
  T blockSum = BlockRed(shm.blockStore).Sum(lossVal);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    atomicAdd(out, blockSum);
  }
}

template <typename T>
void launchLogsoftmax(T *loss_val, T *dldZ, const T *Z, const T *labels, int C,
                      int N, cudaStream_t stream = 0) {

  CUDA_CHECK(cudaMemset(loss_val, 0, sizeof(T)));
  if (C <= 4) {
    dim3 bs(4, 64);
    dim3 gs(ceildiv(N, 64));
    logSoftmaxKernel<T, 4, 64>
        <<<gs, bs, 0, stream>>>(loss_val, dldZ, Z, labels, C, N);
  } else if (C <= 8) {
    dim3 bs(8, 32);
    dim3 gs(ceildiv(N, 32));
    logSoftmaxKernel<T, 8, 32>
        <<<gs, bs, 0, stream>>>(loss_val, dldZ, Z, labels, C, N);
  } else if (C <= 16) {
    dim3 bs(16, 16);
    dim3 gs(ceildiv(N, 16));
    logSoftmaxKernel<T, 16, 16>
        <<<gs, bs, 0, stream>>>(loss_val, dldZ, Z, labels, C, N);
  } else {
    dim3 bs(32, 8);
    dim3 gs(ceildiv(N, 8));
    logSoftmaxKernel<T, 32, 8>
        <<<gs, bs, 0, stream>>>(loss_val, dldZ, Z, labels, C, N);
  }
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T> struct Softmax : GLMBase<T, Softmax<T>> {
  typedef GLMBase<T, Softmax<T>> Super;

  Softmax(int D, int C, bool has_bias, const cublasHandle_t & cublas)
      : Super(D, C, has_bias, cublas) {}

  inline void getLossAndDZ(T *loss_val, SimpleMat<T> &Z,
                           const SimpleVec<T> &y, cudaStream_t stream =0) {

    launchLogsoftmax(loss_val, Z.data, Z.data, y.data, Z.m, Z.n, stream);
  }
};

}; // namespace GLM
}; // namespace ML
