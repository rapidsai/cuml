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

#include <random>
#include <stack>

#include "node.cuh"
#include "reg_stack.cuh"
#include "constants.h"
#include "fitness.cuh"

namespace cuml {
namespace genetic {

/** 
 * Execution kernel for a single program. We assume that the input data 
 * is stored in column major format.
 */
template<int MaxSize>
__global__ void execute_kernel(program_t p, float* data, float* y_pred, int n_rows) {

  // Single evaluation stack per thread
  stack<float, MaxSize> eval_stack;

  for(size_t row_idx = blockIdx.x*blockDim.x + threadIdx.x; row_idx < (size_t)n_rows;
      row_idx += blockDim.x * gridDim.x) {

    // Arithmetic expr stored in prefix form
    for(int e=p->len-1;e>=0;--e) {

      node* curr = &p->nodes[e];
      
      if(detail::is_terminal(curr->t)) {

        // function
        int ar = detail::arity(curr->t);
        float inval = ar > 0 ? eval_stack.pop() : 0.0f;
        ar--;
        float inval1 = ar > 0 ? eval_stack.pop() : 0.0f;
        ar--;
        eval_stack.push(detail::evaluate_node(*curr,data,n_rows,row_idx,inval,inval1));

      } else {

        // constant or variable
        eval_stack.push(detail::evaluate_node(*curr,data, n_rows, row_idx, 0.0f,0.0f ));
      }
    }

    y_pred[row_idx] = eval_stack.pop();

  }                                   
}

program::program() {
  depth         = MAX_STACK_SIZE;
  len           = 0;
  raw_fitness_  = 0.0f;
  metric        = metric_t::mse;
  mut_type      = mutation_t::none;
}

program::program(const program& src) : len(src.len), depth(src.depth), raw_fitness_(src.raw_fitness_), metric(src.metric) {
  nodes = new node[len];
  for(auto i=0; i<len; ++i){
    nodes[i] = src.nodes[i];
  }
}

program& program::operator=(const program& src){
  // Deep copy
  len           = src.len;
  depth         = src.depth;
  raw_fitness_  = src.raw_fitness_;
  metric        = src.metric;

  nodes         = new node[len];

  for(auto i=0; i<len; ++i) {
    nodes[i]    = src.nodes[i];
  }

  return *this;
}

/** 
 * Internal function which computes the score of program p on the given dataset.
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
  
void execute_single(const raft::handle_t &h, program_t p, 
                     float* data, float* y_pred, int n_rows) {

  cudaStream_t stream = h.get_stream();

  execute_kernel<MAX_STACK_SIZE><<<raft::ceildiv(n_rows,GENE_TPB),
                                  GENE_TPB,0,stream>>>(p,
                                                      data, 
                                                      y_pred,
                                                      n_rows );
  CUDA_CHECK(cudaPeekAtLastError());
}

void raw_fitness(const raft::handle_t &h, program_t p, 
                 float* data, float* y, int num_rows, 
                 float* sample_weights, float* score){
    
  cudaStream_t stream = h.get_stream();

  rmm::device_uvector<float> y_pred(num_rows,stream);  
  
  execute_single(h, p, data, y_pred.data(), num_rows);
  metric(h, p, num_rows, y, y_pred.data(), sample_weights, score);
}   

void fitness(const raft::handle_t &h, program_t p, 
             float parsimony_coeff, float* score) { 
  float penalty = parsimony_coeff * p->len;
  *score = p->raw_fitness_ - penalty;
}

/**
 * Get a random subtree of the current program nodes (on CPU)
 */
std::pair<int, int> get_subtree(node* pnodes, int len, std::mt19937 &gen) {
  
  int start,end;
  start=end=0;

  // Specify RNG
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float bound = dist(gen);

  // Specify subtree start probs acc to Koza's selection approach 
  std::vector<float> node_probs(len,0.1);
  float sum = 0.1 * len;
  for(int i=0; i< len ; ++i){
    if(pnodes[i].is_nonterminal()){
      node_probs[i] = 0.9; 
      sum += 0.8;
    }
  }

  // Normalize vector
  for(int i=0;i<len;++i){
    node_probs[i] /= sum;
  }

  start = std::lower_bound(node_probs.begin(),node_probs.end(),bound) - node_probs.begin();
  end = start;

  // Iterate until all function arguments are satisfied in current subtree
  int num_args = 1;
  while(num_args > end - start){
    node curr;
    curr = pnodes[end];
    if(curr.is_nonterminal())num_args += curr.arity();
    ++end;
  }

  return std::make_pair(start,end);
}

void build_program(program &p_out, const param &params,std::mt19937 &gen){
  
  // Define tree
  std::stack<int> arity_stack;
  std::vector<node> nodelist(0);

  // Specify RNG
  std::uniform_int_distribution<> dist_func(0,params.function_set.size()-1);
  std::uniform_int_distribution<> dist_depth(1, MAX_STACK_SIZE);
  std::uniform_int_distribution<> dist_t(0,params.num_features);
  std::uniform_int_distribution<> dist_choice(0,params.num_features+params.function_set.size()-1);
  std::uniform_real_distribution<float> dist_const(params.const_range[0],
                                              params.const_range[1]);

  // Initialize nodes
  int max_depth = dist_depth(gen);
  node::type func = params.function_set[dist_func(gen)];
  node* curr_node = new node(func);
  nodelist.push_back(*curr_node);
  arity_stack.push(curr_node->arity());

  // Fill tree
  while(!arity_stack.empty()){
    int depth = arity_stack.size();
    int ch = dist_choice(gen);
    if(ch <= params.function_set.size() && depth < max_depth){
      // Add a function to node list
      curr_node = new node(params.function_set[dist_func(gen)]);
      nodelist.push_back(*curr_node);
      arity_stack.push(curr_node->arity());
    }
    else{
      // Add terminal
      int vorc = dist_t(gen);
      if(vorc == params.num_features){
        // Add constant
        float val = dist_const(gen);
        curr_node = new node(val);        
      }
      else{
        // Add variable
        int fid = vorc;
        curr_node = new node(fid);
      }

      // Modify nodelist and stack
      nodelist.push_back(*curr_node);
      int end_elem = arity_stack.top();
      arity_stack.pop();
      arity_stack.push(--end_elem);
      while(arity_stack.top() == 0){
        arity_stack.pop();
        if(arity_stack.empty()){
          break;
        }
        end_elem = arity_stack.top();
        arity_stack.pop();
        arity_stack.push(--end_elem);
      }
    }
  }

  // Set new program parameters
  p_out.nodes = &nodelist[0];
  p_out.len = nodelist.size();
  p_out.metric = params.metric;
  p_out.depth = MAX_STACK_SIZE;
  p_out.raw_fitness_ = 0.0f;
}

void point_mutation(const program &prog, program &p_out, const param& params, std::mt19937 &gen){
  
  // Copy program
  p_out = prog;
  
  // Specify RNG
  std::uniform_real_distribution<float> dist_01(0.0f, 1.0f);
  std::uniform_int_distribution<> dist_t(0,params.num_features);
  std::uniform_real_distribution<float> dist_c(params.const_range[0],
                                              params.const_range[1]);
  // Fill with uniform numbers
  std::vector<float> node_probs(p_out.len);
  std::generate(node_probs.begin(),node_probs.end(),[&]{return dist_01(gen);});

  // Mutate nodes
  int len = p_out.len;
  for(int i=0;i<len;++i){
    node curr;
    curr = prog.nodes[i];
    if(node_probs[i] < params.p_point_replace){
      if(curr.is_terminal()){
        // Replace with a var or const
        int ch = dist_t(gen);
        if(ch == (params.num_features + 1)){
          // Add random constant
          p_out.nodes[i] = *(new node(dist_c(gen)));
        }
        else{
          // Add variable ch
          p_out.nodes[i] = *(new node(ch));
        }
      }
      else{
        // Replace current function with another function of the same arity
        auto space_size = params.arity_set.at(curr.arity()).size();
        std::uniform_int_distribution<> dist_nt(0,space_size-1);
        p_out.nodes[i] = *(new node(params.arity_set.at(curr.arity())[dist_nt(gen)]));
      }
    }
  }
}

void crossover(const program &prog, const program &donor, program &p_out, const param &params, std::mt19937 &gen){

  // Get a random subtree of prog to replace
  std::pair<int, int> prog_slice = get_subtree(prog.nodes, prog.len, gen);
  int prog_start = prog_slice.first;
  int prog_end = prog_slice.second;

  // Get subtree of donor
  std::pair<int, int> donor_slice = get_subtree(donor.nodes, donor.len, gen);
  int donor_start = donor_slice.first;
  int donor_end = donor_slice.second;

  // Evolve 
  p_out.len = (prog_start) + (donor_end - donor_start + 1) + (prog.len-prog_end);
  p_out.nodes = new node[p_out.len];
  
  int i=0;
  for(;i<prog_start;++i){
    p_out.nodes[i] = prog.nodes[i];
  }

  for(int j=donor_start;j<donor_end;++i,++j){
    p_out.nodes[i] = donor.nodes[j];
  }

  for(int j=prog_end;j<prog.len;++j,++i){
    p_out.nodes[i] = prog.nodes[i];
  }
}

void subtree_mutation(const program &prog, program &p_out, const param &params, std::mt19937 &gen){
  // Generate a random program and perform crossover
  program_t new_program = new program();
  build_program(*new_program,params,gen);
  crossover(prog,*new_program,p_out,params,gen);
  delete new_program;
}

void hoist_mutation(const program &prog, program &p_out, const param &params, std::mt19937 &gen){
  // Replace program subtree with a random sub-subtree

  std::pair<int, int> prog_slice = get_subtree(prog.nodes, prog.len, gen);
  int prog_start = prog_slice.first;
  int prog_end = prog_slice.second;

  std::pair<int,int> sub_slice = get_subtree(&prog.nodes[prog_start],prog_end-prog_start,gen);
  int sub_start = sub_slice.first;
  int sub_end = sub_slice.second;

  p_out.len = (prog_start) + (sub_end - sub_start + 1) + (prog.len-prog_end);
  p_out.nodes = new node[p_out.len];
  
  int i=0;
  for(;i<prog_start;++i){
    p_out.nodes[i] = prog.nodes[i];
  }

  for(int j=sub_start;j<sub_end;++i,++j){
    p_out.nodes[i] = prog.nodes[j];
  }

  for(int j=prog_end;j<prog.len;++j,++i){
    p_out.nodes[i] = prog.nodes[i];
  }
}

} // namespace genetic
} // namespace cuml