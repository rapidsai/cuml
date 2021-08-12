/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

#include <algorithm>
#include <cstddef>
#include <random>

namespace ML {
namespace Solver {

template <typename math_t>
void initShuffle(std::vector<math_t>& rand_indices, std::mt19937& g, math_t random_state = 0)
{
  g.seed((int)random_state);
  for (std::size_t i = 0; i < rand_indices.size(); ++i)
    rand_indices[i] = i;
}

template <typename math_t>
void shuffle(std::vector<math_t>& rand_indices, std::mt19937& g)
{
  std::shuffle(rand_indices.begin(), rand_indices.end(), g);
}

};  // namespace Solver
};  // namespace ML
// end namespace ML
