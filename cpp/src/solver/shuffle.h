/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
