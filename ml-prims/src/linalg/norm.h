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

#include "linalg/coalesced_reduction.h"
#include "linalg/strided_reduction.h"


namespace MLCommon {
namespace LinAlg {

/** different types of norms supported on the input buffers */
enum NormType { L1Norm = 0, L2Norm };


/**
 * @brief Compute row-wise norm of the input matrix
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @param dots the output vector of row-wise dot products
 * @param data the input matrix (currently assumed to be row-major)
 * @param D number of columns of data
 * @param N number of rows of data
 * @param type the type of norm to be applied
 * @param stream cuda stream where to launch work
 */
template <typename Type, typename IdxType = int>
void rowNorm(Type *dots, const Type *data, IdxType D, IdxType N, NormType type,
          cudaStream_t stream = 0) {
  switch (type) {
    case L1Norm:
      LinAlg::coalescedReduction(dots, data, D, N, (Type)0,
                                 false, stream,
                                 [] __device__(Type in, IdxType i) { return myAbs(in); });
      break;
    case L2Norm:
      LinAlg::coalescedReduction(dots, data, D, N, (Type)0,
                                 false, stream,
                                 [] __device__(Type in, IdxType i) { return in * in; });
      break;
    default:
      ASSERT(false, "Invalid norm type passed! [%d]", type);
  };
}

/**
 * @brief Compute row-wise norm of the input matrix and perform fin_op lambda
 *
 * Row-wise norm is useful while computing pairwise distance matrix, for
 * example.
 * This is used in many clustering algos like knn, kmeans, dbscan, etc... The
 * current implementation is optimized only for bigger values of 'D'.
 *
 * @tparam Type the data type
 * @tparam Lambda device final lambda
 * @tparam IdxType Integer type used to for addressing
 * @param dots the output vector of row-wise dot products
 * @param data the input matrix (currently assumed to be row-major)
 * @param D number of columns of data
 * @param N number of rows of data
 * @param type the type of norm to be applied
 * @param fin_op the final lambda op
 * @param stream cuda stream where to launch work
 */
template <typename Type, typename Lambda, typename IdxType = int>
void rowNorm(Type *dots, const Type *data, IdxType D, IdxType N, NormType type,
             Lambda fin_op, cudaStream_t stream = 0) {
  switch (type) {
    case L1Norm:
      LinAlg::coalescedReduction(dots, data, D, N, (Type)0,
                                 false, stream,
                                 [] __device__(Type in, IdxType i) { return myAbs(in); }, 
                                 [] __device__(Type a, Type b) { return a+b; }, fin_op);
      break;
    case L2Norm:
      LinAlg::coalescedReduction(dots, data, D, N, (Type)0,
                                 false, stream,
                                 [] __device__(Type in, IdxType i) { return in * in; },
                                 [] __device__(Type a, Type b) { return a+b; }, fin_op);
      break;
    default:
      ASSERT(false, "Invalid norm type passed! [%d]", type);
  };
}


/**
 * @brief Compute column-wise norm of the input matrix
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @param dots the output vector of column-wise dot products
 * @param data the input matrix (currently assumed to be row-major)
 * @param D number of columns of data
 * @param N number of rows of data
 * @param type the type of norm to be applied
 * @param stream cuda stream where to launch work
 */
template <typename Type, typename IdxType = int>
void colNorm(Type *dots, const Type *data, IdxType D, IdxType N, NormType type,
             cudaStream_t stream = 0) {
  switch (type) {
    case L1Norm:
      LinAlg::stridedReduction(dots, data, D, N, (Type)0,
                               false, stream,
                               [] __device__(Type v, IdxType i) { return myAbs(v); });
      break;
    case L2Norm:
      LinAlg::stridedReduction(dots, data, D, N, (Type)0,
                               false, stream,
                               [] __device__(Type v, IdxType i) { return v * v; });
      break;
    default:
      ASSERT(false, "Invalid norm type passed! [%d]", type);
  };
}

/**
 * @brief Compute column-wise norm of the input matrix and perform fin_op
 * @tparam Type the data type
 * @tparam Lambda device final lambda
 * @tparam IdxType Integer type used to for addressing
 * @param dots the output vector of column-wise dot products
 * @param data the input matrix (currently assumed to be row-major)
 * @param D number of columns of data
 * @param N number of rows of data
 * @param type the type of norm to be applied
 * @param fin_op the final lambda op
 * @param stream cuda stream where to launch work
 */
template <typename Type, typename Lambda, typename IdxType = int>
void colNorm(Type *dots, const Type *data, IdxType D, IdxType N, NormType type,
             Lambda fin_op, cudaStream_t stream = 0) {
  switch (type) {
    case L1Norm:
      LinAlg::stridedReduction(dots, data, D, N, (Type)0,
                               false, stream,
                               [] __device__(Type v, IdxType i) { return myAbs(v); },
                               [] __device__(Type a, Type b) { return a + b; },
                               fin_op);
      break;
    case L2Norm:
      LinAlg::stridedReduction(dots, data, D, N, (Type)0,
                               false, stream,
                               [] __device__(Type v, IdxType i) { return v * v; },
                               [] __device__(Type a, Type b) { return a + b; },
                               fin_op);
      break;
    default:
      ASSERT(false, "Invalid norm type passed! [%d]", type);
  };
}

}; // end namespace LinAlg
}; // end namespace MLCommon
