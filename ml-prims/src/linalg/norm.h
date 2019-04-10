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

#include "linalg/reduce.h"


namespace MLCommon {
namespace LinAlg {

/** different types of norms supported on the input buffers */
enum NormType { L1Norm = 0, L2Norm };


/**
 * @brief Compute row-wise norm of the input matrix and perform fin_op lambda
 *
 * Row-wise norm is useful while computing pairwise distance matrix, for
 * example.
 * This is used in many clustering algos like knn, kmeans, dbscan, etc... The
 * current implementation is optimized only for bigger values of 'D'.
 *
 * @tparam Type the data type
 * @tparam Lambda Final op lambda
 * @param dots the output vector of row-wise dot products
 * @param data the input matrix (currently assumed to be row-major)
 * @param D number of columns of data
 * @param N number of rows of data
 * @param type the type of norm to be applied
 * @param rowMajor whether the input is row-major or not
 * @param fin_op the final lambda op
 * @param stream cuda stream where to launch work
 */
template <typename Type, typename Lambda = Nop<Type>>
void rowNorm(Type *dots, const Type *data, int D, int N, NormType type,
             bool rowMajor, Lambda fin_op = Nop<Type>(),
             cudaStream_t stream = 0) {
  switch (type) {
    case L1Norm:
      LinAlg::reduce(dots, data, D, N, (Type)0, rowMajor, true, false, stream,
                     L1Op<Type>(), Sum<Type>(), fin_op);
      break;
    case L2Norm:
      LinAlg::reduce(dots, data, D, N, (Type)0, rowMajor, true, false, stream,
                     L2Op<Type>(), Sum<Type>(), fin_op);
      break;
    default:
      ASSERT(false, "Invalid norm type passed! [%d]", type);
  };
}


/**
 * @brief Compute column-wise norm of the input matrix and perform fin_op
 * @tparam Type the data type
 * @param dots the output vector of column-wise dot products
 * @param data the input matrix (currently assumed to be row-major)
 * @param D number of columns of data
 * @param N number of rows of data
 * @param type the type of norm to be applied
 * @param rowMajor whether the input is row-major or not
 * @param fin_op the final lambda op
 * @param stream cuda stream where to launch work
 */
template <typename Type, typename Lambda = Nop<Type>>
void colNorm(Type *dots, const Type *data, int D, int N, NormType type,
             bool rowMajor, Lambda fin_op = Nop<Type>(),
             cudaStream_t stream = 0) {
  switch (type) {
    case L1Norm:
      LinAlg::reduce(dots, data, D, N, (Type)0, rowMajor, false, false, stream,
                     L1Op<Type>(), Sum<Type>(), fin_op);
      break;
    case L2Norm:
      LinAlg::reduce(dots, data, D, N, (Type)0, rowMajor, false, false, stream,
                     L2Op<Type>(), Sum<Type>(), fin_op);
      break;
    default:
      ASSERT(false, "Invalid norm type passed! [%d]", type);
  };
}

}; // end namespace LinAlg
}; // end namespace MLCommon
