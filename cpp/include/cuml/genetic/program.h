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
#include "common.h"
#include <random>
#include <cuml/cuml.hpp>

namespace cuml {
namespace genetic {

struct node;

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

  /** 
   * Default constructor
   */
  explicit program();
  
  /**
   * @param[in] src source program to be copied
   */
  explicit program(const program &src);
  
  /**
   * Destructor for current program
   */
  ~program(){ delete[] nodes; }
  
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

/** program_t is the type of the program */
typedef program* program_t;

/** returns predictions for given dataset on a single program */
void execute_single(const raft::handle_t &h, program_t p, 
                     float* data, float* y_pred, int n_rows);

/** returns predictions for given dataset on multiple programs program */
void execute_batched(const raft::handle_t &h, program_t p, 
                     float* data, float* y_pred, int n_rows, 
                     int n_progs);
                 
/** computes fitness score for a single program */
void raw_fitness(const raft::handle_t &h, program_t p, 
                 float* data, float* y, int num_rows, 
                 float* sample_weights, float* score);

/** returns precomputed fitness score of program */
void fitness(const raft::handle_t &h, program_t p, 
             float parsimony_coeff, float* score);

/** build a random program of max-depth */
void build_program(program_t p_out, const param &params, std::mt19937 &gen);

/** Point mutations on CPU */
void point_mutation(program_t prog, program_t p_out, const param &params, std::mt19937 &gen);

/** Crossover mutations on CPU */
void crossover(program_t prog, program_t donor, program_t p_out, const param &params, std::mt19937 &gen);

/** Subtree mutations on CPU*/
void subtree_mutation(program_t prog, program_t p_out, const param &params, std::mt19937 &gen);

/** Hoist mutation on CPU*/
void hoist_mutation(program_t prog, program_t p_out, const param &params, std::mt19937 &gen);
}  // namespace genetic
}  // namespace cuml
