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
#include <raft/util/cuda_utils.cuh>

namespace MLCommon {
namespace Functions {

template <typename T, typename IdxType = int>
void sigmoid(T* out, T* in, IdxType len, cudaStream_t stream)
{
  T one = T(1);
  raft::linalg::unaryOp(
    out, in, len, [one] __device__(T in) { return one / (one + raft::exp(-in)); }, stream);
}

};  // end namespace Functions
};  // end namespace MLCommon
