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

#include "common.h"
#include <random>
#include <raft/handle.hpp>

namespace cuml {
namespace genetic 
{
  
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
   * @param[dst] dst boolean indicating location of nodes(true for host, false for device)
   */
  explicit program(const program &src, const bool &dst);
  
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

/** Execute given programs on the dataset */
void execute( const raft::handle_t &h, const program_t d_progs, const int n_samples, const int n_progs,
              const float* data, float* y_pred);

/** 
 * Function to compute scores for given y and y_pred on the given dataset.
 */
void compute_metric(const raft::handle_t &h, int n_samples, int n_progs,
                    const float* y, const float* y_pred, const float* w, 
                    float* score, const param& params);

/** 
 * Computes the fitness scores for a single program on the given dataset
 */
void compute_fitness(const raft::handle_t &h, program_t d_prog, float* score,
                     const param &params, const int n_samples, const float* data, 
                     const float* y, const float* sample_weights);

/** 
 * Computes the fitness scores for all programs on the given dataset 
 */
void compute_batched_fitness(const raft::handle_t &h, program_t d_progs, float* score,
                             const param &params, const int n_samples, const float* data, 
                             const float* y, const float* sample_weights);         

/** 
 * Computes and sets the fitness scores for a single program w.r.t the passed dataset
 */
void set_fitness(const raft::handle_t &h, program_t d_prog, program &h_prog,
                 const param &params, const int n_samples, const float* data,
                 const float* y, const float* sample_weights);

/** 
 * Computes and sets the fitness scores of all given programs w.r.t the passed dataset
 */
void set_batched_fitness( const raft::handle_t &h, program_t d_progs, std::vector<program> &h_progs,
                          const param &params, const int n_samples, const float* data,
                          const float* y, const float* sample_weights);

/** Returns precomputed fitness score of program on the host */
float fitness(const program &prog, const param &params);

/** build a random program of max-depth */
void build_program(program &p_out, const param &params, std::mt19937 &gen);

/** Point mutations on CPU */
void point_mutation(const program &prog, program &p_out, const param &params, 
                    std::mt19937 &gen);

/** Crossover mutations on CPU */
void crossover(const program &prog, const program &donor, program &p_out, 
               const param &params, std::mt19937 &gen);

/** Subtree mutations on CPU*/
void subtree_mutation(const program &prog, program &p_out, const param &params, 
                      std::mt19937 &gen);

/** Hoist mutation on CPU*/
void hoist_mutation(const program &prog, program &p_out, const param &params, 
                    std::mt19937 &gen);

}  // namespace genetic
}  // namespace cuml
