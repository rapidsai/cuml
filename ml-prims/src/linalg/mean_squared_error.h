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

#include "map_then_reduce.h"

namespace MLCommon {
namespace LinAlg {

/**
 * @brief CUDA version mean squared error function mean((A-B)**2)
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam TPB threads-per-block 
 * @param out the output mean squared error value (assumed to be a device pointer)
 * @param A input array (assumed to be a device pointer)
 * @param B input array (assumed to be a device pointer)
 * @param len number of elements in the input arrays
 * @param workingBuffer temporary array at least as large as the inputs (assumed to be a device pointer)
 * @param weight weight to apply to every term in the mean squared error calculation
 * @param stream cuda-stream where to launch this kernel
 */
template<typename math_t, int TPB = 256>
    void meanSquaredError(math_t* out, const math_t * A, const math_t *B, size_t len, math_t weight = 1.0, cudaStream_t stream){
        auto sq_diff = [len, weight] __device__(const math_t a, const math_t b){
            math_t diff = a - b;
            return diff * diff * weight / len;
        };
        mapThenSumReduce<math_t, decltype(sq_diff), TPB>(out, len, sq_diff, stream, A, B);
    }

}; // end namespace LinAlg
}; // end namespace MLCommon
