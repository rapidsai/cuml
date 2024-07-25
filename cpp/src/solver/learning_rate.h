/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

#include <cuml/solvers/params.hpp>

#include <math.h>

namespace ML {
namespace Solver {

template <typename math_t>
math_t max(math_t a, math_t b)
{
  return (a < b) ? b : a;
  ;
}

template <typename math_t>
math_t invScaling(math_t eta, math_t power_t, int t)
{
  return (eta / pow(t, power_t));
}

template <typename math_t>
math_t regDLoss(math_t a, math_t b)
{
  return a - b;
}

template <typename math_t>
math_t calOptimalInit(math_t alpha)
{
  math_t typw         = sqrt(math_t(1.0) / sqrt(alpha));
  math_t initial_eta0 = typw / max(math_t(1.0), regDLoss(-typw, math_t(1.0)));
  return (math_t(1.0) / (initial_eta0 * alpha));
}

template <typename math_t>
math_t optimal(math_t alpha, math_t optimal_init, int t)
{
  return math_t(1.0) / (alpha * (optimal_init + t - 1));
}

template <typename math_t>
math_t calLearningRate(ML::lr_type lr_type, math_t eta, math_t power_t, math_t alpha, math_t t)
{
  if (lr_type == ML::lr_type::CONSTANT) {
    return eta;
  } else if (lr_type == ML::lr_type::INVSCALING) {
    return invScaling(eta, power_t, t);
  } else if (lr_type == ML::lr_type::OPTIMAL) {
    return optimal(alpha, eta, t);
  } else {
    return math_t(0);
  }
}

};  // namespace Solver
};  // namespace ML
// end namespace ML
