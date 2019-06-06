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

#include "cuda_utils.h"
#include "linalg/matrix_vector_op.h"

namespace MLCommon {
namespace LinAlg {

template <typename Type, typename IdxType = int>
__global__ void naiveMatVecKernel(Type *out, const Type *mat, const Type *vec,
                                  IdxType D, IdxType N, bool rowMajor,
                                  bool bcastAlongRows, Type scalar) {
  IdxType idx = threadIdx.x + blockIdx.x * blockDim.x;
  IdxType len = N * D;
  IdxType col;
  if (rowMajor && bcastAlongRows) {
    col = idx % D;
  } else if (!rowMajor && !bcastAlongRows) {
    col = idx % N;
  } else if (rowMajor && !bcastAlongRows) {
    col = idx / D;
  } else {
    col = idx / N;
  }
  if (idx < len) {
    out[idx] = mat[idx] + scalar * vec[col];
  }
}

template <typename Type, typename IdxType = int>
void naiveMatVec(Type *out, const Type *mat, const Type *vec, IdxType D,
                 IdxType N, bool rowMajor, bool bcastAlongRows, Type scalar) {
  static const IdxType TPB = 64;
  IdxType len = N * D;
  IdxType nblks = ceildiv(len, TPB);
  naiveMatVecKernel<Type>
    <<<nblks, TPB>>>(out, mat, vec, D, N, rowMajor, bcastAlongRows, scalar);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename Type, typename IdxType = int>
__global__ void naiveMatVecKernel(Type *out, const Type *mat, const Type *vec1,
                                  const Type *vec2, IdxType D, IdxType N,
                                  bool rowMajor, bool bcastAlongRows,
                                  Type scalar) {
  IdxType idx = threadIdx.x + blockIdx.x * blockDim.x;
  IdxType len = N * D;
  IdxType col;
  if (rowMajor && bcastAlongRows) {
    col = idx % D;
  } else if (!rowMajor && !bcastAlongRows) {
    col = idx % N;
  } else if (rowMajor && !bcastAlongRows) {
    col = idx / D;
  } else {
    col = idx / N;
  }
  if (idx < len) {
    out[idx] = mat[idx] + scalar * vec1[col] + vec2[col];
  }
}

template <typename Type, typename IdxType = int>
void naiveMatVec(Type *out, const Type *mat, const Type *vec1, const Type *vec2,
                 IdxType D, IdxType N, bool rowMajor, bool bcastAlongRows,
                 Type scalar) {
  static const IdxType TPB = 64;
  IdxType len = N * D;
  IdxType nblks = ceildiv(len, TPB);
  naiveMatVecKernel<Type><<<nblks, TPB>>>(out, mat, vec1, vec2, D, N, rowMajor,
                                          bcastAlongRows, scalar);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // end namespace LinAlg
}  // end namespace MLCommon
