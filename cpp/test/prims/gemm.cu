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

#include <gtest/gtest.h>
#include "linalg/gemm.h"

namespace MLCommon {
namespace LinAlg {

template <typename T>
__global__ void fillKernel(T *arr, T val, int N) {
  const int stride = blockDim.x * gridDim.x;
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  for (int i = tid; i < N; i += stride) arr[i] = val;
}

template <typename T, int NTHREADS = 256, int NITEMS = 4>
void fill(T *arr, T val, int N) {
  const int nblks = ceildiv<int>(N, NTHREADS * NITEMS);
  fillKernel<T><<<nblks, NTHREADS>>>(arr, val, N);
  CUDA_CHECK(cudaPeekAtLastError());
}

TEST(Gemm, Gemm_128x128x8) {
  float *A, *B, *C, *D;
  int M = 128, N = 128, K = 64;
  CUDA_CHECK(cudaMalloc((void **)&A, sizeof(float) * M * K));
  fill(A, 1.f, M * K);
  CUDA_CHECK(cudaMalloc((void **)&B, sizeof(float) * K * N));
  fill(B, 0.5f, K * N);
  CUDA_CHECK(cudaMalloc((void **)&C, sizeof(float) * M * N));
  fill(C, 2.f, M * N);
  CUDA_CHECK(cudaMalloc((void **)&D, sizeof(float) * M * N));
  CUDA_CHECK(cudaMemset(D, 0, sizeof(float) * M * N));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  gemm<float, float, float, cutlass::Shape<8, 128, 128>>(
    CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, 1.f, B, N, A, K, 1.f, C, N, D, stream);
  float *hD = new float[M * N];
  updateHost<float>(hD, D, M * N, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (int i = 0; i < M * N; ++i) {
    ASSERT_FLOAT_EQ(0.5f * K + 2.f, hD[i]) << " @hD[" << i << "]";
  }
  delete[] hD;
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
  CUDA_CHECK(cudaFree(D));
}

}  // end namespace LinAlg
}  // end namespace MLCommon
