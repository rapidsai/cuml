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

#include <linalg/unary_op.h>


namespace MLCommon {
namespace Functions {

template <typename T>
void sigmoid(T *out, T *in, int len);

template <>
inline void sigmoid(float *out, float *in, int len) {
	float scalar = float(1);
	LinAlg::unaryOp(out, in, len, [scalar] __device__ (float in) {
		                                         return 1.0 / (1.0 + expf(-in));
		                                   });
}

template <>
inline void sigmoid(double *out, double *in, int len) {
	double scalar = double(1);
	LinAlg::unaryOp(out, in, len, [scalar] __device__ (double in) {
		                                         return 1.0 / (1.0 + exp(-in));
		                                   });
}


/** @} */
}
;
}
;
// end namespace ML
