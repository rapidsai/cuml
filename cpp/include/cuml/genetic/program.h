/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <raft/core/handle.hpp>

#include <random>

namespace cuml {
namespace genetic {

/**
 * @brief The main data structure to store the AST that represents a program
 *        in the current generation
 */
struct program {
  /**
   * the AST. It is stored in the reverse of DFS-right-child-first order. In
   * other words, construct a regular AST in the form of depth-first, but
   * instead of storing the left child first, store the right child and so on.
   * Now take the resulting 1D array and reverse it.
   *
   * @note The pointed memory buffer is NOT owned by this class and further it
   *       is assumed to be a zero-copy (aka pinned memory) buffer, at least in
   *       this initial version
   */

  /**
   * Default constructor
   */
  explicit program();

  /**
   * @brief Destroy the program object
   *
   */
  ~program();

  /**
   * @brief Copy constructor for a new program object
   *
   * @param src
   */
  explicit program(const program& src);

  /**
   * @brief assignment operator
   *
   * @param[in] src source program to be copied
   *
   * @return current program reference
   */
  program& operator=(const program& src);

  node* nodes;
  /** total number of nodes in this AST */
  int len;
  /** maximum depth of this AST */
  int depth;
  /** fitness score of current AST */
  float raw_fitness_;
  /** fitness metric used for current AST*/
  metric_t metric;
  /** mutation type responsible for production */
  mutation_t mut_type;
};  // struct program

/** program_t is a shorthand for device programs */
typedef program* program_t;

/**
 * @brief Calls the execution kernel to evaluate all programs on the given dataset
 *
 * @param h          cuML handle
 * @param d_progs    Device pointer to programs
 * @param n_rows     Number of rows in the input dataset
 * @param n_progs    Total number of programs being evaluated
 * @param data       Device pointer to input dataset (in col-major format)
 * @param y_pred     Device pointer to output of program evaluation
 */
void execute(const raft::handle_t& h,
             const program_t& d_progs,
             const int n_rows,
             const int n_progs,
             const float* data,
             float* y_pred);

/**
 * @brief Compute the loss based on the metric specified in the training hyperparameters.
 *        It performs a batched computation for all programs in one shot.
 *
 * @param h         cuML handle
 * @param n_rows    The number of labels/rows in the expected output
 * @param n_progs   The number of programs being batched
 * @param y         Device pointer to the expected output (SIZE = n_samples)
 * @param y_pred    Device pointer to the predicted output (SIZE = n_samples * n_progs)
 * @param w         Device pointer to sample weights (SIZE = n_samples)
 * @param score     Device pointer to final score (SIZE = n_progs)
 * @param params    Training hyperparameters
 */
void compute_metric(const raft::handle_t& h,
                    int n_rows,
                    int n_progs,
                    const float* y,
                    const float* y_pred,
                    const float* w,
                    float* score,
                    const param& params);

/**
 * @brief Computes the fitness scores for a sngle program on the given dataset
 *
 * @param h cuML handle
 * @param d_prog          Device pointer to program
 * @param score           Device pointer to fitness vals
 * @param params          Training hyperparameters
 * @param n_rows          Number of rows in the input dataset
 * @param data            Device pointer to input dataset
 * @param y               Device pointer to input labels
 * @param sample_weights  Device pointer to sample weights
 */
void find_fitness(const raft::handle_t& h,
                  program_t& d_prog,
                  float* score,
                  const param& params,
                  const int n_rows,
                  const float* data,
                  const float* y,
                  const float* sample_weights);

/**
 * @brief Computes the fitness scores for all programs on the given dataset
 *
 * @param h cuML handle
 * @param n_progs         Batch size(Number of programs)
 * @param d_progs         Device pointer to list of programs
 * @param score           Device pointer to fitness vals computed for all programs
 * @param params          Training hyperparameters
 * @param n_rows          Number of rows in the input dataset
 * @param data            Device pointer to input dataset
 * @param y               Device pointer to input labels
 * @param sample_weights  Device pointer to sample weights
 */
void find_batched_fitness(const raft::handle_t& h,
                          int n_progs,
                          program_t& d_progs,
                          float* score,
                          const param& params,
                          const int n_rows,
                          const float* data,
                          const float* y,
                          const float* sample_weights);

/**
 * @brief Computes and sets the fitness scores for a single program on the given dataset
 *
 * @param h cuML handle
 * @param d_prog          Device pointer to program
 * @param h_prog          Host program object
 * @param params          Training hyperparameters
 * @param n_rows          Number of rows in the input dataset
 * @param data            Device pointer to input dataset
 * @param y               Device pointer to input labels
 * @param sample_weights  Device pointer to sample weights
 */
void set_fitness(const raft::handle_t& h,
                 program_t& d_prog,
                 program& h_prog,
                 const param& params,
                 const int n_rows,
                 const float* data,
                 const float* y,
                 const float* sample_weights);

/**
 * @brief Computes and sets the fitness scores for all programs on the given dataset
 *
 * @param h cuML handle
 * @param n_progs         Batch size
 * @param d_progs         Device pointer to list of programs
 * @param h_progs         Host vector of programs corresponding to d_progs
 * @param params          Training hyperparameters
 * @param n_rows          Number of rows in the input dataset
 * @param data            Device pointer to input dataset
 * @param y               Device pointer to input labels
 * @param sample_weights  Device pointer to sample weights
 */
void set_batched_fitness(const raft::handle_t& h,
                         int n_progs,
                         program_t& d_progs,
                         std::vector<program>& h_progs,
                         const param& params,
                         const int n_rows,
                         const float* data,
                         const float* y,
                         const float* sample_weights);

/**
 * @brief Returns precomputed fitness score of program on the host,
 *        after accounting for parsimony
 *
 * @param prog    The host program
 * @param params  Training hyperparameters
 * @return Fitness score corresponding to trained program
 */
float get_fitness(const program& prog, const param& params);

/**
 * @brief Evaluates and returns the depth of the current program.
 *
 * @param p_out The given program
 * @return The depth of the current program
 */
int get_depth(const program& p_out);

/**
 * @brief Build a random program with depth atmost 10
 *
 * @param p_out   The output program
 * @param params  Training hyperparameters
 * @param rng     RNG to decide nodes to add
 */
void build_program(program& p_out, const param& params, std::mt19937& rng);

/**
 * @brief Perform a point mutation on the given program(AST)
 *
 * @param prog    The input program
 * @param p_out   The result program
 * @param params  Training hyperparameters
 * @param rng     RNG to decide nodes to mutate
 */
void point_mutation(const program& prog, program& p_out, const param& params, std::mt19937& rng);

/**
 * @brief Perform a 'hoisted' crossover mutation using the parent and donor programs.
 *        The donor subtree selected is hoisted to ensure our constrains on total depth
 *
 * @param prog    The input program
 * @param donor   The donor program
 * @param p_out   The result program
 * @param params  Training hyperparameters
 * @param rng     RNG for subtree selection
 */
void crossover(const program& prog,
               const program& donor,
               program& p_out,
               const param& params,
               std::mt19937& rng);

/**
 * @brief Performs a crossover mutation with a randomly built new program.
 *        Since crossover is 'hoisted', this will ensure that depth constrains
 *        are not violated.
 *
 * @param prog    The input program
 * @param p_out   The result mutated program
 * @param params  Training hyperparameters
 * @param rng     RNG to control subtree selection and temporary program addition
 */
void subtree_mutation(const program& prog, program& p_out, const param& params, std::mt19937& rng);

/**
 * @brief Perform a hoist mutation on a random subtree of the given program
 *        (replace a subtree with a subtree of a subtree)
 *
 * @param prog    The input program
 * @param p_out   The output program
 * @param params  Training hyperparameters
 * @param rng     RNG to control subtree selection
 */
void hoist_mutation(const program& prog, program& p_out, const param& params, std::mt19937& rng);
}  // namespace genetic
}  // namespace cuml
