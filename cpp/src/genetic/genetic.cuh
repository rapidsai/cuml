/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cuml/genetic/common.h>
#include <raft/cuda_utils.cuh>

namespace cuml {
namespace genetic {
namespace detail {

HDI float p_reproduce(const param& p) {
  auto sum = p.p_crossover + p.p_subtree_mutation + p.p_hoist_mutation +
             p.p_point_mutation;
  auto ret = 1.f - sum;
  return fmaxf(0.f, fminf(ret, 1.f));
}

HDI int max_programs(const param& p) {
  // in the worst case every generation's top program ends up reproducing,
  // thereby adding another program into the population
  return p.population_size + p.generations;
}

HDI int criterion(const param& p) {
  // Returns 0 if a smaller value is preferred and 1 for the opposite
  switch (p.metric) {
    case metric_t::mse:
      return 0;
    case metric_t::logloss:
      return 0;
    case metric_t::mae:
      return 0;
    case metric_t::rmse:
      return 0;
    case metric_t::pearson:
      return 1;
    case metric_t::spearman:
      return 1;
    default:
      return -1;
  }
}
}  // namespace detail
}  // namespace genetic
}  // namespace cuml
