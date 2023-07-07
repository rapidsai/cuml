/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

#include <raft/linalg/unary_op.cuh>

namespace MLCommon {
namespace Functions {

template <typename math_t, typename idx_type = int>
void sign(
  math_t* out, const math_t* in, const math_t scalar, const idx_type len, cudaStream_t stream)
{
  raft::linalg::unaryOp(
    out,
    in,
    len,
    [scalar] __device__(math_t in) {
      if (in < math_t(0))
        return (math_t(-1) * scalar);
      else if (in > math_t(0))
        return (math_t(1) * scalar);
      else
        return math_t(0);
    },
    stream);
}

template <typename math_t, typename idx_type = int>
void sign(math_t* out, const math_t* in, const idx_type n_len, cudaStream_t stream)
{
  math_t scalar = math_t(1);
  sign(out, in, scalar, n_len, stream);
}

};  // namespace Functions
};  // namespace MLCommon
// end namespace ML
