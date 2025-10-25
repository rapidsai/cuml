/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
