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

#include "node.h"
#include "program.h"

#include <cstdint>
#include <string>
#include <vector>

namespace cuml {
namespace genetic {

/** Type of initialization of the member programs in the population */
enum class init_method_t : uint32_t {
  /** random nodes chosen, allowing shorter or asymmetrical trees */
  grow,
  /** growing till a randomly chosen depth */
  full,
  /** 50% of the population on `grow` and the rest with `full` */
  half_and_half,
};  // enum class init_method_t

/** fitness metric types */
enum class metric_t : uint32_t {
  /** mean absolute error (regression-only) */
  mae,
  /** mean squared error (regression-only) */
  mse,
  /** root mean squared error (regression-only) */
  rmse,
  /** pearson product-moment coefficient (regression and transformation) */
  pearson,
  /** spearman's rank-order coefficient (regression and transformation) */
  spearman,
  /** binary cross-entropy loss (classification-only) */
  logloss,
};  // enum class metric_t

enum class transformer_t : uint32_t {
  /** sigmoid function */
  sigmoid,
};  // enum class transformer_t

/**
 * @brief contains all the hyper-parameters for training
 *
 * @note Unless otherwise mentioned, all the parameters below are applicable to
 *       all of classification, regression and transformation.
 */
struct param {
  /** number of programs in each generation */
  int population_size = 1000;
  /**
   * number of fittest programs to compare during correlation
   * (transformation-only)
   */
  int hall_of_fame = 100;
  /**
   * number of fittest programs to return from `hall_of_fame` top programs
   * (transformation-only)
   */
  int n_components = 10;
  /** number of generations to evolve */
  int generations = 20;
  /**
   * number of programs that compete in the tournament to become part of next
   * generation
   */
  int tournament_size = 20;
  /** metric threshold used for early stopping */
  float stopping_criteria = 0.0f;
  /** minimum/maximum value for `constant` nodes */
  float const_range[2] = {-1.0f, 1.0f};
  /** minimum/maximum depth of programs after initialization */
  int init_depth[2] = {2, 6};
  /** initialization method */
  init_method_t init_method = init_method_t::half_and_half;
  /** list of functions to choose from */
  std::vector<node::type> function_set{node::type::add, node::type::mul,
                                       node::type::div, node::type::sub};
  /** transformation function to class probabilities (classification-only) */
  transformer_t transformer = transformer_t::sigmoid;
  /** fitness metric */
  metric_t metric = metric_t::mae;
  /** penalization factor for large programs */
  float parsimony_coefficient = 0.001f;
  /** crossover mutation probability of the tournament winner */
  float p_crossover = 0.9f;
  /** subtree mutation probability of the tournament winner*/
  float p_subtree_mutation = 0.01f;
  /** hoist mutation probability of the tournament winner */
  float p_hoist_mutation = 0.01f;
  /** point mutation probabiilty of the tournament winner */
  float p_point_mutation = 0.01f;
  /** point replace probabiility for point mutations */
  float p_point_replace = 0.05f;
  /** subsampling factor */
  float max_samples = 1.0f;
  /** list of feature names for generating syntax trees from the programs */
  std::vector<std::string> feature_names;
  ///@todo: feature_names
  ///@todo: verbose
  /** random seed used for RNG */
  uint64_t random_state = 0ull;

  /** Computes the probability of 'reproduction' */
  float p_reproduce() const;

  /** maximum possible number of programs */
  int max_programs() const;
};  // struct param

}  // namespace genetic
}  // namespace cuml
