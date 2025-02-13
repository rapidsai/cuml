/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Stability {

/**
 * Uses cluster distances, births, and sizes to compute stabilities
 * which are used for cluster selection.
 * @tparam value_idx
 * @tparam value_t
 */
template <typename value_idx, typename value_t>
struct stabilities_functor {
 public:
  stabilities_functor(value_t* stabilities_,
                      const value_t* births_,
                      const value_idx* parents_,
                      const value_t* lambdas_,
                      const value_idx* sizes_,
                      const value_idx n_leaves_)
    : stabilities(stabilities_),
      births(births_),
      parents(parents_),
      lambdas(lambdas_),
      sizes(sizes_),
      n_leaves(n_leaves_)
  {
  }

  __device__ void operator()(const int& idx)
  {
    auto parent = parents[idx] - n_leaves;

    atomicAdd(&stabilities[parent], (lambdas[idx] - births[parent]) * sizes[idx]);
  }

 private:
  value_t* stabilities;
  const value_t *births, *lambdas;
  const value_idx *parents, *sizes, n_leaves;
};

};  // namespace Stability
};  // namespace detail
};  // namespace HDBSCAN
};  // namespace ML
