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
#include "linalg/binary_op.h"


namespace MLCommon {
namespace Stats {


///@todo: ColPerBlk has been tested only for 32!
template <typename Type, int TPB, int ColsPerBlk = 32>
__global__ void stddevKernelRowMajor(Type *std, const Type *data, int D,
                                     int N) {
  const int RowsPerBlkPerIter = TPB / ColsPerBlk;
  int thisColId = threadIdx.x % ColsPerBlk;
  int thisRowId = threadIdx.x / ColsPerBlk;
  int colId = thisColId + (blockIdx.y * ColsPerBlk);
  int rowId = thisRowId + (blockIdx.x * RowsPerBlkPerIter);
  Type thread_data = Type(0);
  const int stride = RowsPerBlkPerIter * gridDim.x;
  for (int i = rowId; i < N; i += stride) {
    Type val = (colId < D) ? data[i * D + colId] : Type(0);
    thread_data += val * val;
  }
  __shared__ Type sstd[ColsPerBlk];
  if (threadIdx.x < ColsPerBlk)
    sstd[threadIdx.x] = Type(0);
  __syncthreads();
  myAtomicAdd(sstd + thisColId, thread_data);
  __syncthreads();
  if (threadIdx.x < ColsPerBlk)
    myAtomicAdd(std + colId, sstd[thisColId]);
}

template <typename Type, int TPB>
__global__ void stddevKernelColMajor(Type *std, const Type *data,
                                     const Type *mu, int D, int N) {
  typedef cub::BlockReduce<Type, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  Type thread_data = Type(0);
  int colStart = blockIdx.x * N;
  Type m = mu[blockIdx.x];
  for (int i = threadIdx.x; i < N; i += TPB) {
    int idx = colStart + i;
    Type diff = data[idx] - m;
    thread_data += diff * diff;
  }
  Type acc = BlockReduce(temp_storage).Sum(thread_data);
  if (threadIdx.x == 0) {
    std[blockIdx.x] = mySqrt(acc / N);
  }
}


template <typename Type, int TPB>
__global__ void varsKernelColMajor(Type *var, const Type *data, const Type *mu,
                                   int D, int N) {
  typedef cub::BlockReduce<Type, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  Type thread_data = Type(0);
  int colStart = blockIdx.x * N;
  Type m = mu[blockIdx.x];
  for (int i = threadIdx.x; i < N; i += TPB) {
    int idx = colStart + i;
    Type diff = data[idx] - m;
    thread_data += diff * diff;
  }
  Type acc = BlockReduce(temp_storage).Sum(thread_data);
  if (threadIdx.x == 0) {
    var[blockIdx.x] = acc / N;
  }
}

/**
 * @brief Compute stddev of the input matrix
 *
 * Stddev operation is assumed to be performed on a given column.
 *
 * @tparam Type the data type
 * @param std the output stddev vector
 * @param data the input matrix
 * @param mu the mean vector
 * @param D number of columns of data
 * @param N number of rows of data
 * @param sample whether to evaluate sample stddev or not. In other words,
 * whether
 *  to normalize the output using N-1 or N, for true or false, respectively
 * @param rowMajor whether the input data is row or col major
 * @param stream cuda stream where to launch work
 */
template <typename Type>
void stddev(Type *std, const Type *data, const Type *mu, int D, int N,
            bool sample, bool rowMajor, cudaStream_t stream = 0) {
  static const int TPB = 256;
  if (rowMajor) {
    static const int RowsPerThread = 4;
    static const int ColsPerBlk = 32;
    static const int RowsPerBlk = (TPB / ColsPerBlk) * RowsPerThread;
    dim3 grid(ceildiv(N, RowsPerBlk), ceildiv(D, ColsPerBlk));
    CUDA_CHECK(cudaMemset(std, 0, sizeof(Type) * D));
    stddevKernelRowMajor<Type, TPB, ColsPerBlk><<<grid, TPB, 0, stream>>>(
      std, data, D, N);
    Type ratio = Type(1) / (sample ? Type(N - 1) : Type(N));
    LinAlg::binaryOp(std, std, mu, D, [ratio] __device__(Type a, Type b) {
      return mySqrt(a * ratio - b * b);
    });
  } else {
    stddevKernelColMajor<Type, TPB><<<D, TPB, 0, stream>>>(std, data, mu, D, N);
  }
  CUDA_CHECK(cudaPeekAtLastError());
}


/**
 * @brief Compute standard deviation of the input matrix using multiple GPUs.
 *
 * Variance operation is assumed to be performed on a given column
 *
 * @tparam Type the data type
 * @param std the output stddev vector
 * @param data the input matrix
 * @param mu the mean vector
 * @param D number of columns of data
 * @param N number of rows of data
 * @param n_gpu: number of gpus.
 * @param sample whether to evaluate sample stddev or not. In other words,
 * whether
 *  to normalize the output using N-1 or N, for true or false, respectively
 * @param rowMajor whether the input data is row or col major
 * @param row_split: true if the data is broken by row
 * @param sync: synch the streams if it's true
 */
template <typename Type>
void stddevMG(TypeMG<Type> *std, const TypeMG<Type> *data,
              const TypeMG<Type> *mu, int D, int N, int n_gpus, bool sample,
              bool rowMajor, bool row_split = false, bool sync = false) {
  if (row_split) {
    ASSERT(false, "varsMG: row split is not supported");
  } else {
    for (int i = 0; i < n_gpus; i++) {
      CUDA_CHECK(cudaSetDevice(data[i].gpu_id));

      stddev(std[i].d_data, data[i].d_data, mu[i].d_data, data[i].n_cols,
             data[i].n_rows, sample, rowMajor, data[i].stream);
    }
  }

  if (sync)
    streamSyncMG(data, n_gpus);
}

/**
 * @brief Compute variance of the input matrix
 *
 * Variance operation is assumed to be performed on a given column.
 *
 * @tparam Type the data type
 * @param var the output stddev vector
 * @param data the input matrix
 * @param mu the mean vector
 * @param D number of columns of data
 * @param N number of rows of data
 * @param sample whether to evaluate sample stddev or not. In other words,
 * whether
 *  to normalize the output using N-1 or N, for true or false, respectively
 * @param rowMajor whether the input data is row or col major
 * @param stream cuda stream where to launch work
 */
template <typename Type>
void vars(Type *var, const Type *data, const Type *mu, int D, int N,
          bool sample, bool rowMajor, cudaStream_t stream = 0) {
  static const int TPB = 256;
  if (rowMajor) {
    static const int RowsPerThread = 4;
    static const int ColsPerBlk = 32;
    static const int RowsPerBlk = (TPB / ColsPerBlk) * RowsPerThread;
    dim3 grid(ceildiv(N, RowsPerBlk), ceildiv(D, ColsPerBlk));
    CUDA_CHECK(cudaMemset(var, 0, sizeof(Type) * D));
    stddevKernelRowMajor<Type, TPB, ColsPerBlk><<<grid, TPB, 0, stream>>>(
      var, data, D, N);
    Type ratio = Type(1) / (sample ? Type(N - 1) : Type(N));
    LinAlg::binaryOp(var, var, mu, D, [ratio] __device__(Type a, Type b) {
      return a * ratio - b * b;
    });
  } else {
    varsKernelColMajor<Type, TPB><<<D, TPB, 0, stream>>>(var, data, mu, D, N);
  }
  CUDA_CHECK(cudaPeekAtLastError());
}


/**
 * @brief Compute variance of the input matrix using multiple GPUs.
 *
 * Variance operation is assumed to be performed on a given column
 *
 * @tparam Type the data type
 * @param var the output stddev vector
 * @param data the input matrix
 * @param mu the mean vector
 * @param D number of columns of data
 * @param N number of rows of data
 * @param n_gpu: number of gpus.
 * @param sample whether to evaluate sample stddev or not. In other words,
 * whether
 *  to normalize the output using N-1 or N, for true or false, respectively
 * @param rowMajor whether the input data is row or col major
 * @param row_split: true if the data is broken by row
 * @param sync: synch the streams if it's true
 */
template <typename Type>
void varsMG(TypeMG<Type> *var, const TypeMG<Type> *data, const TypeMG<Type> *mu,
            int D, int N, int n_gpus, bool sample, bool rowMajor,
            bool row_split = false, bool sync = false) {
  if (row_split) {
    ASSERT(false, "varsMG: row split is not supported");
  } else {
    for (int i = 0; i < n_gpus; i++) {
      CUDA_CHECK(cudaSetDevice(data[i].gpu_id));

      vars(var[i].d_data, data[i].d_data, mu[i].d_data, data[i].n_cols,
           data[i].n_rows, sample, rowMajor, data[i].stream);
    }
  }

  if (sync)
    streamSyncMG(data, n_gpus);
}

}; // end namespace Stats
}; // end namespace MLCommon
