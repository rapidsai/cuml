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

#include "linalg/unary_op.h"


namespace MLCommon {
namespace Functions {


template <typename T>
void f_log(T *out, T *in, T scalar, int len, cudaStream_t stream);

template <>
inline void f_log(float *out, float *in, float scalar, int len, cudaStream_t stream) {
	LinAlg::unaryOp(out, in, len, [scalar] __device__ (float in) {
		                                         return logf(in) * scalar;
		                                   },
                                       stream);

}

template <>
inline void f_log(double *out, double *in, double scalar, int len, cudaStream_t stream) {
	LinAlg::unaryOp(out, in, len, [scalar] __device__ (double in) {
		                                         return log(in) * scalar;
		                                   },
                                       stream);
}


/** @} */
}
;
}
;
// end namespace ML
