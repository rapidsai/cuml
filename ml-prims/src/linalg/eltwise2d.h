#pragma once
#include <cuda_utils.h>

namespace MLCommon {
namespace LinAlg {

template <typename Type, typename Lambda>
__global__ void eltwise2DKernel(int rows, // m
                                int cols, // n
                                const Type* dotA,
                                const Type* dotB,
                                const Type* pC,
                                Type* pD,
                                Type alpha,
                                Type beta,
                                Lambda op)
{
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < cols * rows) {
    const auto x = tid % cols;
    const auto y = tid / cols;
    const auto ab = pD[tid];
    const auto a = dotA[y];
    const auto b = dotB[x];
    Type accm = alpha * op(a, b, ab);

    if (beta) {
      accm += beta * pC[tid];
    }
    pD[tid] = accm;
  }
}

template <typename Type, typename Lambda>
void eltwise2D(int rows, // m
               int cols, // n
               const Type* dotA,
               const Type* dotB,
               const Type* pC,
               Type* pD,
               Type alpha,
               Type beta,
               Lambda op,
               cudaStream_t stream=0)
{
  size_t threads = 256;
  size_t blocks = ((cols * rows) + threads - 1) / threads;
  eltwise2DKernel<Type><<<blocks, threads, 0, stream>>>(rows, cols,
    dotA, dotB, pC, pD, alpha, beta, op);
  CUDA_CHECK(cudaPeekAtLastError());
}

} // namespace LinAlg
} // namespace MLCommon
