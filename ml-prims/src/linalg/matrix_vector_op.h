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
#include "vectorized.h"


namespace MLCommon {
namespace LinAlg {


template <typename Type, int veclen_, typename Lambda>
__global__ void
  matrixVectorOpKernel(Type *out, const Type *matrix, const Type *vector, int D,
                       int N, bool rowMajor, bool bcastAlongRows, Lambda op) {
  typedef TxN_t<Type, veclen_> VecType;
  int len = N * D;
  int idx = (threadIdx.x + (blockIdx.x * blockDim.x)) * VecType::Ratio;
  if (idx >= len)
    return;
  int vIdx;
  VecType mat, vec;
  ///@todo: yikes! use fast-int-div here.
  ///@todo: shared mem for vector could help with perf
  if (rowMajor && bcastAlongRows) {
    vIdx = idx % D;
    vec.load(vector, vIdx);
  } else if (!rowMajor && !bcastAlongRows) {
    vIdx = idx % N;
    vec.load(vector, vIdx);
  } else if (rowMajor && !bcastAlongRows) {
    vIdx = idx / D;
    Type b = vector[vIdx];
    vec.fill(b);
  } else {
    vIdx = idx / N;
    Type b = vector[vIdx];
    vec.fill(b);
  }
  mat.load(matrix, idx);
#pragma unroll
  for (int i = 0; i < VecType::Ratio; ++i)
    mat.val.data[i] = op(mat.val.data[i], vec.val.data[i]);
  mat.store(out, idx);
}

template <typename Type, int veclen_, typename Lambda, int TPB>
void matrixVectorOpImpl(Type *out, const Type *matrix, const Type *vec, int D,
                        int N, bool rowMajor, bool bcastAlongRows, Lambda op,
                        cudaStream_t stream = 0) {
  int nblks = ceildiv(N * D, TPB);
  matrixVectorOpKernel<Type, veclen_, Lambda><<<nblks, TPB, 0, stream>>>(
    out, matrix, vec, D, N, rowMajor, bcastAlongRows, op);
  CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @brief Operations for all the columns or rows with a given vector.
 * @tparam Type the matrix/vector type
 * @tparam Lambda a device function which represents a binary operator
 * @tparam TPB threads per block of the cuda kernel launched
 * @param out the output matrix (passing out = matrix makes it in-place)
 * @param matrix the input matrix
 * @param vec the vector
 * @param D number of columns of matrix
 * @param N number of rows of matrix
 * @param rowMajor whether input is row or col major
 * @param bcastAlongRows whether the broadcast of vector needs to happen along
 * the rows of the matrix or columns
 * @param op the mathematical operation
 * @param stream cuda stream where to launch work
 */
template <typename Type, typename Lambda, int TPB = 256>
void matrixVectorOp(Type *out, const Type *matrix, const Type *vec, int D,
                    int N, bool rowMajor, bool bcastAlongRows, Lambda op,
                    cudaStream_t stream = 0) {
  int stride = rowMajor ? D : N;
  size_t bytes = stride * sizeof(Type);
  if (16 / sizeof(Type) && bytes % 16 == 0) {
    matrixVectorOpImpl<Type, 16 / sizeof(Type), Lambda, TPB>(
      out, matrix, vec, D, N, rowMajor, bcastAlongRows, op, stream);
  } else if (8 / sizeof(Type) && bytes % 8 == 0) {
    matrixVectorOpImpl<Type, 8 / sizeof(Type), Lambda, TPB>(
      out, matrix, vec, D, N, rowMajor, bcastAlongRows, op, stream);
  } else if (4 / sizeof(Type) && bytes % 4 == 0) {
    matrixVectorOpImpl<Type, 4 / sizeof(Type), Lambda, TPB>(
      out, matrix, vec, D, N, rowMajor, bcastAlongRows, op, stream);
  } else if (2 / sizeof(Type) && bytes % 2 == 0) {
    matrixVectorOpImpl<Type, 2 / sizeof(Type), Lambda, TPB>(
      out, matrix, vec, D, N, rowMajor, bcastAlongRows, op, stream);
  } else if (1 / sizeof(Type)) {
    matrixVectorOpImpl<Type, 1 / sizeof(Type), Lambda, TPB>(
      out, matrix, vec, D, N, rowMajor, bcastAlongRows, op, stream);
  } else {
    matrixVectorOpImpl<Type, 1, Lambda, TPB>(out, matrix, vec, D, N, rowMajor,
                                             bcastAlongRows, op, stream);
  }
}

}; // end namespace LinAlg
}; // end namespace MLCommon
