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

#include <cuml/genetic/program.h>
#include <cuml/genetic/node.h>
#include <raft/cudart_utils.h>

#include "reg_stack.cuh"
#include "constants.cuh"
#include "fitness.cuh"

namespace cuml {
namespace genetic {

void metric(const raft::handle_t &h, program_t p, int len, float* y, float* y_pred, float* w, float* score){
  // Call appropriate metric function based on metric defined in p
  cudaStream_t stream = h.get_stream();

  if(p->metric == metric_t::pearson){
    _weighted_pearson(stream, len, y, y_pred, w, score);
  } else if(p->metric == metric_t::spearman){
    _weighted_spearman(stream, len, y, y_pred, w, score);
  } else if(p->metric == metric_t::mae){
    _mean_absolute_error(stream, len, y, y_pred, w, score);
  } else if(p->metric == metric_t::mse){
    _mean_square_error(stream, len, y, y_pred, w, score);
  } else if(p->metric == metric_t::rmse){
    _root_mean_square_error(stream, len, y, y_pred, w, score);
  } else if(p->metric == metric_t::logloss){
    _log_loss(stream, len, y, y_pred, w, score);
  } else{
    // None of the above - error
  }
}
  

template<int MaxSize>
__global__ void execute_k(program_t p, float** X, float* y_hat){
  
  stack<float, MaxSize> eval_s;
}

/** returns predictions for given dataset and program */
void execute(const raft::handle_t &h, program_t p, float** X, float* y_pred, int num_rows, float* sample_weights){

  cudaStream_t stream = h.get_stream();

  // config kernel launch params - single program
  // dim3 numBlocks(num_rows/GENE_TPB,1,1);
  // dim3 numThreads(GENE_TPB,1,1);
  
  // execute_k<MAXSSIZE><<<numBlocks,numThreads,0,stream>>>(p, X, y_pred);
}

/** Computes the raw fitness metric for a single program on a given dataset */
void raw_fitness(const raft::handle_t &h, program_t p, float** X, float* y, 
                int num_rows, float* sample_weights, float* score){
    
  cudaStream_t stream = h.get_stream();
    
  float* y_pred;
  raft::allocate<float>(y_pred, num_rows,true);   
  
  // execute program on dataset
  execute(h, p, X, y_pred, num_rows, sample_weights);  
  
  // Populate score according to fitness metric
  metric(h, p, num_rows, y, y_pred, sample_weights, score);
}   


void fitness(const raft::handle_t &h, program_t p, float parsimony_coeff, float* score) {
    
    float penalty = parsimony_coeff * p->len;
    *score = p->raw_fitness_ - penalty;
}

}
}