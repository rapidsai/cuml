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
#include <rmm/device_uvector.hpp>

#include "node.cuh"
#include "reg_stack.cuh"
#include "constants.h"
#include "fitness.cuh"

namespace cuml {
namespace genetic {

/** 
  Execution kernel for a single program. We assume that the input data 
  is stored in column major format.
 */
template<int MaxSize>
__global__ void execute_kernel(program_t p,
                               float* data, 
                               float* y_pred, 
                               int n_rows) {
  // Single evaluation stack per thread
  stack<float, MaxSize> eval_stack;
  for(size_t row_idx = blockIdx.x*blockDim.x + threadIdx.x; 
      row_idx < (size_t)n_rows;
      row_idx += blockDim.x * gridDim.x) {
    // Arithmetic expr stored in postfix form
    for(int e=0;e<p->len;++e) {
      node* curr = &p->nodes[e];
      if(detail::is_terminal(curr->t)) {
        // function
        int ar = detail::arity(curr->t);
        float inval = ar > 0 ? eval_stack.pop() : 0.0f;
        ar--;
        float inval1 = ar > 0 ? eval_stack.pop() : 0.0f;
        ar--;
        eval_stack.push(detail::evaluate_node(*curr,
                                      data,
                                      n_rows,
                                      row_idx,
                                      inval,
                                      inval1));
      } 
      else{
        // constant or variable
        eval_stack.push(detail::evaluate_node(*curr, 
                                          data, 
                                          n_rows, 
                                          row_idx, 
                                          0.0f,
                                          0.0f ));
      }
    } 
    y_pred[row_idx] = eval_stack.pop();
  }                                   
}

/** 
  Internal function which computes the score of program p on the given dataset.
  */
void metric(const raft::handle_t &h, program_t p, 
            int len, float* y, float* y_pred, 
            float* w, float* score) {
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
  
/** Returns predictions for a single program on input data*/
void execute_single(const raft::handle_t &h, program_t p, 
                     float* data, float* y_pred, int n_rows) {

  cudaStream_t stream = h.get_stream();

  execute_kernel<MAX_STACK_SIZE><<<raft::ceildiv(n_rows,GENE_TPB),
                                  GENE_TPB,0,stream>>>(
                                    p,
                                    data, 
                                    y_pred,
                                    n_rows );
  CUDA_CHECK(cudaPeekAtLastError());
}

/** Computes the raw fitness metric for a single program on a given dataset */
void raw_fitness(const raft::handle_t &h, program_t p, 
                 float* data, float* y, int num_rows, 
                 float* sample_weights, float* score){
    
  cudaStream_t stream = h.get_stream();

  rmm::device_uvector<float> y_pred(num_rows,stream);  
  
  execute_single(h, p, data, y_pred.data(), num_rows);
  metric(h, p, num_rows, y, y_pred.data(), sample_weights, score);
}   

/** 
  Returns the pre-computed fitness score for the given program p. 
  We assume that p is stored on the CPU.
  */
void fitness(const raft::handle_t &h, program_t p, 
             float parsimony_coeff, float* score) { 
  float penalty = parsimony_coeff * p->len;
  *score = p->raw_fitness_ - penalty;
}

}
}