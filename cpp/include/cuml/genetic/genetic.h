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

#pragma once

#include "common.h"
#include "program.h"

#include <raft/core/handle.hpp>

namespace cuml {
namespace genetic {

/**
 * @brief Visualize an AST
 *
 * @param prog  host object containing the AST
 * @return      String representation of the AST
 */
std::string stringify(const program& prog);

/**
 * @brief Fit either a regressor, classifier or a transformer to the given dataset
 *
 * @param handle          cuML handle
 * @param input           device pointer to the feature matrix
 * @param labels          device pointer to the label vector of length n_rows
 * @param sample_weights  device pointer to the sample weights of length n_rows
 * @param n_rows          number of rows of the feature matrix
 * @param n_cols          number of columns of the feature matrix
 * @param params          host struct containing hyperparameters needed for training
 * @param final_progs     device pointer to the final generation of programs(sorted by decreasing
 * fitness)
 * @param history         host vector containing the list of all programs in every generation
 * (sorted by decreasing fitness)
 *
 * @note This module allocates extra device memory for the nodes of the last generation that is
 * pointed by `final_progs[i].nodes` for each program `i` in `final_progs`. The amount of memory
 * allocated is found at runtime, and is `final_progs[i].len * sizeof(node)` for each program `i`.
 * The reason this isn't deallocated within the function is because the resulting memory is needed
 * for executing predictions in `symRegPredict`, `symClfPredict`, `symClfPredictProbs` and
 * `symTransform` functions. The above device memory is expected to be explicitly deallocated by the
 * caller AFTER calling the predict function.
 */
void symFit(const raft::handle_t& handle,
            const float* input,
            const float* labels,
            const float* sample_weights,
            const int n_rows,
            const int n_cols,
            param& params,
            program_t& final_progs,
            std::vector<std::vector<program>>& history);

/**
 * @brief Make predictions for a symbolic regressor
 *
 * @param handle      cuML handle
 * @param input       device pointer to feature matrix
 * @param n_rows      number of rows of the feature matrix
 * @param best_prog   device pointer to best AST fit during training
 * @param output      device pointer to output values
 */
void symRegPredict(const raft::handle_t& handle,
                   const float* input,
                   const int n_rows,
                   const program_t& best_prog,
                   float* output);

/**
 * @brief Probability prediction for a symbolic classifier. If a transformer(like sigmoid) is
 *        specified, then it is applied on the output before returning it.
 *
 * @param handle      cuML handle
 * @param input       device pointer to feature matrix
 * @param n_rows      number of rows of the feature matrix
 * @param params      host struct containing training hyperparameters
 * @param best_prog   The best program obtained during training. Inferences are made using this
 * @param output      device pointer to output probability(in col major format)
 */
void symClfPredictProbs(const raft::handle_t& handle,
                        const float* input,
                        const int n_rows,
                        const param& params,
                        const program_t& best_prog,
                        float* output);

/**
 * @brief Return predictions for a binary classification program defining the decision boundary
 *
 * @param handle      cuML handle
 * @param input       device pointer to feature matrix
 * @param n_rows      number of rows of the feature matrix
 * @param params      host struct containing training hyperparameters
 * @param best_prog   Best program obtained after training
 * @param output      Device pointer to output predictions
 */
void symClfPredict(const raft::handle_t& handle,
                   const float* input,
                   const int n_rows,
                   const param& params,
                   const program_t& best_prog,
                   float* output);

/**
 * @brief Transform the values in the input feature matrix according to the supplied programs
 *
 * @param handle      cuML handle
 * @param input       device pointer to feature matrix
 * @param params      Hyperparameters used during training
 * @param final_progs List of ASTs used for generating new features
 * @param n_rows      number of rows of the feature matrix
 * @param n_cols      number of columns of the feature matrix
 * @param output      device pointer to transformed input
 */
void symTransform(const raft::handle_t& handle,
                  const float* input,
                  const param& params,
                  const program_t& final_progs,
                  const int n_rows,
                  const int n_cols,
                  float* output);

}  // namespace genetic
}  // namespace cuml
