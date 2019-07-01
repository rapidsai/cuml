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

#include "coalesced_reduction.h"
#include "cuda_utils.h"
#include "strided_reduction.h"

namespace MLCommon {
namespace LinAlg {

/**
 * @brief Compute reduction of the input matrix along the requested dimension
 *
 * @tparam InType the data type of the input
 * @tparam OutType the data type of the output (as well as the data type for
 *  which reduction is performed)
 * @tparam IdxType data type of the indices of the array
 * @tparam MainLambda Unary lambda applied while acculumation (eg: L1 or L2 norm)
 * It must be a 'callable' supporting the following input and output:
 * <pre>OutType (*MainLambda)(InType, IdxType);</pre>
 * @tparam ReduceLambda Binary lambda applied for reduction (eg: addition(+) for L2 norm)
 * It must be a 'callable' supporting the following input and output:
 * <pre>OutType (*ReduceLambda)(OutType);</pre>
 * @tparam FinalLambda the final lambda applied before STG (eg: Sqrt for L2 norm)
 * It must be a 'callable' supporting the following input and output:
 * <pre>OutType (*FinalLambda)(OutType);</pre>
 * @param dots the output reduction vector
 * @param data the input matrix
 * @param D number of columns
 * @param N number of rows
 * @param init initial value to use for the reduction
 * @param rowMajor input matrix is row-major or not
 * @param alongRows whether to reduce along rows or columns
 * @param stream cuda stream where to launch work
 * @param inplace reduction result added inplace or overwrites old values?
 * @param main_op elementwise operation to apply before reduction
 * @param reduce_op binary reduction operation
 * @param final_op elementwise operation to apply before storing results
 */
template <typename InType, typename OutType = InType, typename IdxType = int,
          typename MainLambda = Nop<InType, IdxType>,
          typename ReduceLambda = Sum<OutType>,
          typename FinalLambda = Nop<OutType>>
void reduce(OutType *dots, const InType *data, int D, int N, OutType init,
            bool rowMajor, bool alongRows, cudaStream_t stream,
            bool inplace = false, MainLambda main_op = Nop<InType, IdxType>(),
            ReduceLambda reduce_op = Sum<OutType>(),
            FinalLambda final_op = Nop<OutType>()) {
  if (rowMajor && alongRows) {
    coalescedReduction(dots, data, D, N, init, stream, inplace, main_op,
                       reduce_op, final_op);
  } else if (rowMajor && !alongRows) {
    stridedReduction(dots, data, D, N, init, stream, inplace, main_op,
                     reduce_op, final_op);
  } else if (!rowMajor && alongRows) {
    stridedReduction(dots, data, N, D, init, stream, inplace, main_op,
                     reduce_op, final_op);
  } else {
    coalescedReduction(dots, data, N, D, init, stream, inplace, main_op,
                       reduce_op, final_op);
  }
}

};  // end namespace LinAlg
};  // end namespace MLCommon
