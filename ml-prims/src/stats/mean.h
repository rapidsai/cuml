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

#include <cub/cub.cuh>
#include "cuda_utils.h"
#include "linalg/eltwise.h"

namespace MLCommon {
namespace Stats {

///@todo: ColsPerBlk has been tested only for 32!
template <typename Type, int TPB, int ColsPerBlk = 32>
__global__ void meanKernelRowMajor(Type *mu, const Type *data, int D, int N) {
  const int RowsPerBlkPerIter = TPB / ColsPerBlk;
  int thisColId = threadIdx.x % ColsPerBlk;
  int thisRowId = threadIdx.x / ColsPerBlk;
  int colId = thisColId + (blockIdx.y * ColsPerBlk);
  int rowId = thisRowId + (blockIdx.x * RowsPerBlkPerIter);
  Type thread_data = Type(0);
  const int stride = RowsPerBlkPerIter * gridDim.x;
  for (int i = rowId; i < N; i += stride)
    thread_data += (colId < D) ? data[i * D + colId] : Type(0);
  __shared__ Type smu[ColsPerBlk];
  if (threadIdx.x < ColsPerBlk)
    smu[threadIdx.x] = Type(0);
  __syncthreads();
  myAtomicAdd(smu + thisColId, thread_data);
  __syncthreads();
  if (threadIdx.x < ColsPerBlk)
    myAtomicAdd(mu + colId, smu[thisColId]);
}

template <typename Type, int TPB>
__global__ void meanKernelColMajor(Type *mu, const Type *data, int D, int N) {
  typedef cub::BlockReduce<Type, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  Type thread_data = Type(0);
  int colStart = blockIdx.x * N;
  for (int i = threadIdx.x; i < N; i += TPB) {
    int idx = colStart + i;
    thread_data += data[idx];
  }
  Type acc = BlockReduce(temp_storage).Sum(thread_data);
  if (threadIdx.x == 0) {
    mu[blockIdx.x] = acc / N;
  }
}

/**
 * @brief Compute mean of the input matrix
 *
 * Mean operation is assumed to be performed on a given column.
 *
 * @tparam Type: the data type
 * @param mu: the output mean vector
 * @param data: the input matrix
 * @param D: number of columns of data
 * @param N: number of rows of data
 * @param sample: whether to evaluate sample mean or not. In other words,
 * whether
 *  to normalize the output using N-1 or N, for true or false, respectively
 * @param rowMajor: whether the input data is row or col major
 * @param stream: cuda stream
 */
template <typename Type>
void mean(Type *mu, const Type *data, int D, int N, bool sample, bool rowMajor,
          cudaStream_t stream = 0) {
  static const int TPB = 256;
  if (rowMajor) {
    static const int RowsPerThread = 4;
    static const int ColsPerBlk = 32;
    static const int RowsPerBlk = (TPB / ColsPerBlk) * RowsPerThread;
    dim3 grid(ceildiv(N, RowsPerBlk), ceildiv(D, ColsPerBlk));
    CUDA_CHECK(cudaMemset(mu, 0, sizeof(Type) * D));
    meanKernelRowMajor<Type, TPB, ColsPerBlk><<<grid, TPB, 0, stream>>>(
      mu, data, D, N);
    CUDA_CHECK(cudaPeekAtLastError());
    Type ratio = Type(1) / (sample ? Type(N - 1) : Type(N));
    LinAlg::scalarMultiply(mu, mu, ratio, D);
  } else {
    meanKernelColMajor<Type, TPB><<<D, TPB, 0, stream>>>(mu, data, D, N);
  }
  CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @brief Compute mean of the input matrix using multiple GPUs.
 *
 * Mean operation is assumed to be performed on a given column.
 *
 * @param Type the data type
 * @param mu: the output mean vector
 * @param data: the input matrix
 * @param D: number of columns of data
 * @param N: number of rows of data
 * @param n_gpu: number of gpus.
 * @param sample whether to evaluate sample mean or not. In other words, whether
 *  to normalize the output using N-1 or N, for true or false, respectively
 * @param rowMajor whether the input data is row or col major
 * @param row_split: true if the data is broken by row
 * @param sync: synch the streams if it's true
 */
template <typename Type>
void meanMG(TypeMG<Type> *mu, const TypeMG<Type> *data, int D, int N,
            int n_gpus, bool sample, bool rowMajor, bool row_split = false,
            bool sync = false) {
  // TODO: row split is not tested yet.
  if (row_split) {
    TypeMG<Type> mu_temp[n_gpus];
    for (int i = 0; i < n_gpus; i++) {
      mu_temp[i].gpu_id = data[i].gpu_id;
      mu_temp[i].n_cols = data[i].n_cols;
      mu_temp[i].n_rows = 1;
    }

    allocateMG(mu_temp, n_gpus, n_gpus, mu_temp[0].n_cols, true, false, true);

    for (int i = 0; i < n_gpus; i++) {
      CUDA_CHECK(cudaSetDevice(data[i].gpu_id));

      mean(mu_temp[i].d_data, data[i].d_data, data[i].n_cols, data[i].n_rows,
           sample, rowMajor, data[i].stream);
    }

    int len = mu_temp[0].n_cols * n_gpus;
    Type *h_mu_temp = (Type *)malloc(len * sizeof(Type));
    updateHostMG(h_mu_temp, mu_temp, n_gpus, rowMajor);

    Type *h_mu = (Type *)malloc(mu_temp[0].n_cols * sizeof(Type));
    for (int i = 0; i < mu_temp[i].n_cols; i++) {
      for (int j = 0; j < n_gpus; j++) {
        h_mu[i] = h_mu[i] + h_mu_temp[j * mu_temp[i].n_cols + i];
      }
    }

    updateDeviceMG(mu, h_mu, n_gpus, false);
  } else {
    for (int i = 0; i < n_gpus; i++) {
      CUDA_CHECK(cudaSetDevice(data[i].gpu_id));

      mean(mu[i].d_data, data[i].d_data, data[i].n_cols, data[i].n_rows, sample,
           rowMajor, data[i].stream);
    }
  }

  if (sync)
    streamSyncMG(data, n_gpus);
}
};
// end namespace Stats
};
// end namespace MLCommon
