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

namespace MLCommon {
namespace LinAlg {

namespace {
  /**
   * Like Python range.
   *
   * Fills the output as out[i] = i.

   * \param [out] out device array, size [n]
   * \param [in] n length of the array
   */
__global__ void range(int *out, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) {
    out[tid] = tid;
  }
}

}  // unnamed namespace
}  // namespace LinAlg
}  // namespace MLCommon
