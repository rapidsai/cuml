/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include "../../prims/test_utils.h"

#include <cuml/genetic/common.h>

#include <gtest/gtest.h>

namespace cuml {
namespace genetic {

TEST(Genetic, ParamTest)
{
  param p;
  ASSERT_EQ(p.population_size, 1000);
  ASSERT_EQ(p.hall_of_fame, 100);
  ASSERT_EQ(p.n_components, 10);
  ASSERT_EQ(p.generations, 20);
  ASSERT_EQ(p.tournament_size, 20);
  ASSERT_EQ(p.stopping_criteria, 0.0f);
  ASSERT_EQ(p.const_range[0], -1.0f);
  ASSERT_EQ(p.const_range[1], 1.0f);
  ASSERT_EQ(p.init_depth[0], 2);
  ASSERT_EQ(p.init_depth[1], 6);
  ASSERT_EQ(p.init_method, init_method_t::half_and_half);
  ASSERT_EQ(p.function_set.size(), 4u);
  ASSERT_EQ(p.function_set[0], node::type::add);
  ASSERT_EQ(p.function_set[1], node::type::mul);
  ASSERT_EQ(p.function_set[2], node::type::div);
  ASSERT_EQ(p.function_set[3], node::type::sub);
  ASSERT_EQ(p.transformer, transformer_t::sigmoid);
  ASSERT_EQ(p.arity_set[2][0], node::type::add);
  ASSERT_EQ(p.arity_set[2].size(), 4);
  ASSERT_EQ(p.metric, metric_t::mae);
  ASSERT_EQ(p.parsimony_coefficient, 0.001f);
  ASSERT_EQ(p.p_crossover, 0.9f);
  ASSERT_EQ(p.p_subtree_mutation, 0.01f);
  ASSERT_EQ(p.p_hoist_mutation, 0.01f);
  ASSERT_EQ(p.p_point_mutation, 0.01f);
  ASSERT_EQ(p.p_point_replace, 0.05f);
  ASSERT_EQ(p.max_samples, 1.0f);
  ASSERT_EQ(p.feature_names.size(), 0u);
  ASSERT_EQ(p.random_state, 0ull);
}

TEST(Genetic, p_reproduce)
{
  param p;
  auto ret = p.p_reproduce();
  ASSERT_TRUE(MLCommon::match(p.p_reproduce(), 0.07f, MLCommon::CompareApprox<float>(0.0001f)));
}

}  // namespace genetic
}  // namespace cuml
