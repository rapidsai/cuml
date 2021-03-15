/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include "fitness.h"
#include <cuml/cuml.hpp>

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
   *       is assumed to be a zero-copy (aka pinned memory) buffer, atleast in
   *       this initial version
   */
  node* nodes;
  /** total number of nodes in this AST */
  int len;
  /** maximum depth of this AST */
  int depth;
  /** fitness score of current AST */
  float raw_fitness_;
  /** fitness metric used for current AST*/
  metric_t metric;
};  // struct program

/** program_t is the type of the program */
typedef program* program_t;

/** returns predictions for given dataset */
void execute(const raft::handle_t &h, program_t p, 
                 float** X, float* y_pred, int num_rows, float* sample_weights);
                 
/** computes fitness score for a single program */
void raw_fitness(const raft::handle_t &h, program_t p, 
                 float** X, float* y, int num_rows, float* sample_weights, float* score);

/** returns precomputed fitness score of program*/
void fitness(const raft::handle_t &h, program_t p, 
             float parsimony_coeff, float* score);

}  // namespace genetic
}  // namespace cuml
