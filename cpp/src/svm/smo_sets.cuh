/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <raft/util/cuda_utils.cuh>

namespace ML {
namespace SVM {

/** Determine whether a training instance is in the upper set */
template <typename math_t>
DI bool in_upper(math_t a, math_t y, math_t C)
{
  // (0 < a && a < C) || (y == 1  && a == 0) || (y == -1 && a == C);
  // since a is always clipped to lie in the [0 C] region, therefore this is equivalent with
  return (y < 0 && a > 0) || (y > 0 && a < C);
}

/** Determine whether a training instance is in the lower set */
template <typename math_t>
DI bool in_lower(math_t a, math_t y, math_t C)
{
  // (0 < a && a < C) || (y == -1 && a == 0) || (y == 1 && a == C);
  // since a is always clipped to lie in the [0 C] region, therefore this is equivalent with
  return (y < 0 && a < C) || (y > 0 && a > 0);
}

};  // end namespace SVM
};  // namespace ML
