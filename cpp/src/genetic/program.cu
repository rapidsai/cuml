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
#include <cuml/common/logger.hpp>
#include <rmm/device_uvector.hpp>

#include <random>
#include <stack>
#include <algorithm>
#include <numeric>

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
template<int MaxSize=2*MAX_STACK_SIZE>
__global__ void execute_kernel( const program_t d_progs, const float* data, 
                                float* y_pred, const int n_samples, const int n_progs) {
  
  size_t pid = blockIdx.y;                                      // current program 
  size_t row_id = blockIdx.x * blockDim.x + threadIdx.x;        // current dataset row
  
  if(row_id >= n_samples) { return; }

  stack<float, MaxSize> eval_stack;                             // Maintain stack only for remaining threads
  program_t curr_p = &d_progs[pid];                             // Current program
  
  int e = curr_p->len - 1;
  node* curr_n = &curr_p->nodes[e];
  float res = 0.0f; float in[2] = {0.0f,0.0f};
  
  while(e>=0) {
    if(detail::is_nonterminal(curr_n->t)){
      int ar = detail::arity(curr_n->t);
      in[0]  = eval_stack.pop();                                 // Min arity of function is 1
      if(ar>1) in[1]  = eval_stack.pop();
    }
    res = detail::evaluate_node(*curr_n,data,n_samples,row_id,in);
    eval_stack.push(res);
    curr_n--;
    e--;
  }

  // Outputs stored in col-major format
  y_pred[pid * n_samples + row_id] = eval_stack.pop();
                                     
}

program::program() {
  depth         = MAX_STACK_SIZE;
  len           = 0;
  raw_fitness_  = 0.0f;
  metric        = metric_t::mse;
  mut_type      = mutation_t::none;
}

program::program(const program& src, const bool &dst) : len(src.len), depth(src.depth), raw_fitness_(src.raw_fitness_), metric(src.metric), mut_type(src.mut_type) {
  if(dst) {
    nodes = new node[len];
    // Replace loop with a std::copy
    std::copy(src.nodes, src.nodes + src.len, nodes);
  }
}

program& program::operator=(const program& src){
  // Deep copy
  len           = src.len;
  depth         = src.depth;
  raw_fitness_  = src.raw_fitness_;
  metric        = src.metric;

  nodes         = new node[len];
  
  // Replace loop with a memcpy  
  std::copy(src.nodes, src.nodes + src.len, nodes);
  return *this;
}

void compute_metric(const raft::handle_t &h, int n_samples, int n_progs, 
                    const float* y, const float* y_pred, const float* w, 
                    float* score, const param& params) {
  // Call appropriate metric function based on metric defined in params
  if(params.metric == metric_t::pearson){
    _weighted_pearson(h, n_samples, n_progs, y, y_pred, w, score);
  } 
  else if(params.metric == metric_t::spearman){
    _weighted_spearman(h, n_samples, n_progs, y, y_pred, w, score);
  } 
  else if(params.metric == metric_t::mae){
    _mean_absolute_error(h, n_samples, n_progs, y, y_pred, w, score);
  } 
  else if(params.metric == metric_t::mse){
    _mean_square_error(h, n_samples, n_progs, y, y_pred, w, score);
  } 
  else if(params.metric == metric_t::rmse){
    _root_mean_square_error(h, n_samples, n_progs, y, y_pred, w, score);
  } 
  else if(params.metric == metric_t::logloss){
    _log_loss(h, n_samples, n_progs, y, y_pred, w, score);
  } 
  else{
    // This should not be reachable
  }
}

void execute (const raft::handle_t &h, const program_t d_progs, const int n_samples, 
              const int n_progs, const float* data, float* y_pred){

  cudaStream_t stream = h.get_stream();
  dim3 blks(raft::ceildiv(n_samples,GENE_TPB),n_progs,1);
  // CUML_LOG_DEBUG("Launching Program Execution kernel with (%d, %d, %d) blks ; %d threads",blks.x,blks.y,blks.z,GENE_TPB);

  execute_kernel<<<blks,GENE_TPB,0,stream>>>(d_progs, data, y_pred, n_samples, n_progs);
  CUDA_CHECK(cudaPeekAtLastError());

}

void compute_fitness(const raft::handle_t &h, program_t d_prog, float* score,
                     const param &params, const int n_samples, const float* data, 
                     const float* y, const float* sample_weights) {
  cudaStream_t stream = h.get_stream();

  // Compute predicted values
  rmm::device_uvector<float> y_pred(n_samples, stream);
  execute(h, d_prog, n_samples, 1, data, y_pred.data());

  // Compute error
  compute_metric(h, n_samples, 1, y, y_pred.data(), sample_weights, score, params);
}

void compute_batched_fitness(const raft::handle_t &h, program_t d_progs, float* score,
                             const param &params, const int n_samples, const float* data, 
                             const float* y, const float* sample_weights){
  cudaStream_t stream = h.get_stream();
  int n_progs         = params.population_size;

  rmm::device_uvector<float> y_pred(n_samples * n_progs, stream);
  execute(h, d_progs, n_samples, n_progs, data, y_pred.data());

  // Compute error
  compute_metric(h, n_samples, n_progs, y, y_pred.data(), sample_weights, score, params);
}

void set_fitness(const raft::handle_t &h, program_t d_prog, program &h_prog,
                 const param &params, const int n_samples, const float* data,
                 const float* y, const float* sample_weights) {
  cudaStream_t stream = h.get_stream();

  rmm::device_uvector<float> score(1, stream);

  compute_fitness(h, d_prog, score.data(), params, n_samples, data, y, sample_weights);

  // Update host and device score for program
  CUDA_CHECK(cudaMemcpyAsync( &d_prog[0].raw_fitness_, score.data(), sizeof(float), 
                              cudaMemcpyDeviceToDevice, stream)); 
  h_prog.raw_fitness_ = score.front_element(stream);
}

void set_batched_fitness( const raft::handle_t &h, program_t d_progs,
                     std::vector<program> &h_progs, const param &params, const int n_samples,
                     const float* data, const float* y, const float* sample_weights) {

  cudaStream_t stream   = h.get_stream();
  int n_progs           = params.population_size;
  
  rmm::device_uvector<float> score(n_progs,stream);

  compute_batched_fitness(h,d_progs,score.data(),params,n_samples,data,y,sample_weights);
  
  // Update scores on host and device
  // TODO: Find a way to reduce the number of implicit memory transfers
  for(auto i=0; i < n_progs; ++i){
    CUDA_CHECK(cudaMemcpyAsync( &d_progs[i].raw_fitness_,score.element_ptr(i),sizeof(float),
                                cudaMemcpyDeviceToDevice,stream));
    h_progs[i].raw_fitness_ = score.element(i, stream);
  }
}

float fitness(const program &prog, const param &params) { 
  int crit      = params.criterion();
  float penalty = params.parsimony_coefficient * prog.len * (2*crit - 1);
  return (prog.raw_fitness_ - penalty);
}

/**
 * @brief Checks if the given program is valid or not
 * 
 * @param prog The input program
 */
void validate_program(program &prog){

  std::stack<int> s;
  s.push(0);
  node curr;
  for(int i=0;i<prog.len; ++i){
    curr = prog.nodes[i];
    if(curr.is_nonterminal()){
      s.push(curr.arity());
    }
    else{
      int end_elem = s.top();
      s.pop();
      s.push(end_elem-1);
      while(s.top() == 0){
        s.pop();
        if(s.empty()){break;}
        end_elem = s.top();
        s.pop();
        s.push(end_elem-1);
      }
    }
  }

  if(!(s.size() == 1 && s.top() == -1)){
    CUML_LOG_DEBUG("Invalid program.");
    exit(0);
  }
  
}

/**
 * @brief Get a random subtree of the current program nodes (on CPU)
 * 
 * @param pnodes  AST represented as a list of nodes
 * @param len     The total number of nodes in the AST
 * @param gen     Random number generator for subtree selection
 * @return A tuple [first,last) which contains the required subtree
 */
std::pair<int, int> get_subtree(node* pnodes, int len, std::mt19937 &gen) {
  
  int start,end;
  start=end=0;

  // Specify RNG
  std::uniform_real_distribution<float> dist_U(0.0f, 1.0f);
  float bound = dist_U(gen);

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

  // Compute cumulative sum
  std::partial_sum(node_probs.begin(),node_probs.end(),node_probs.begin());

  // CUML_LOG_DEBUG("Current bound is %f",bound);
  
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

  // Debug when end > len. Ideally, it should never happen
  if(end > len){
    CUML_LOG_DEBUG("Start is -> %d",start);
    CUML_LOG_DEBUG("End is -> %d",end);
    CUML_LOG_DEBUG("Length is -> %d",len);

    for(int i=0;i<end;++i){
      CUML_LOG_DEBUG("Node #%d -> %d (%d inputs)",i,
                    static_cast<std::underlying_type<node::type>::type>(pnodes[i].t)
                    ,pnodes[i].arity());
    }
    exit(0);
  }
  
  return std::make_pair(start,end);
}

void build_program(program &p_out, const param &params,std::mt19937 &gen){
  
  // Define tree
  std::stack<int> arity_stack;
  std::vector<node> nodelist;
  nodelist.reserve(1 << (MAX_STACK_SIZE));

  // Specify RNGs
  std::uniform_int_distribution<> dist_func(0,params.function_set.size()-1);
  std::uniform_int_distribution<int> dist_depth(1, MAX_STACK_SIZE-1); 
  std::uniform_int_distribution<int> dist_t(0,params.num_features);
  std::uniform_int_distribution<int> dist_choice(0,params.num_features+params.function_set.size()-1);
  std::uniform_real_distribution<float> dist_const(params.const_range[0],params.const_range[1]);
  std::uniform_int_distribution<int> dist_U(0,1);

  // MAX_STACK_SIZE can only handle btree of size MAX_STACK_SIZE - max(arity) + 1

  // Initialize nodes
  int max_depth = dist_depth(gen);
  node::type func = params.function_set[dist_func(gen)];
  node curr_node(func);
  nodelist.push_back(curr_node);
  arity_stack.push(curr_node.arity());
  
  init_method_t method = params.init_method;
  if(method == init_method_t::half_and_half){
    // Choose either grow or full for this tree
    int ch = dist_U(gen);
    method = ch == 0 ? init_method_t::grow : init_method_t::full;
  }

  // Fill tree
  while(!arity_stack.empty()){
    int depth = arity_stack.size();
    int node_choice = dist_U(gen);
    if((node_choice == 0 || method == init_method_t::full) && depth < max_depth){
      // Add a function to node list
      curr_node = node(params.function_set[dist_func(gen)]);
      nodelist.push_back(curr_node);
      arity_stack.push(curr_node.arity());
    }
    else{
      // Add terminal
      int terminal_choice = dist_t(gen);
      if(terminal_choice == params.num_features){
        // Add constant
        float val = dist_const(gen);
        curr_node = node(val);        
      }
      else{
        // Add variable
        int fid = terminal_choice;
        curr_node = node(fid);
      }

      // Modify nodelist and stack
      nodelist.push_back(curr_node);
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

  // Set new program parameters - need to do a copy as 
  // nodelist will be deleted using RAII semantics
  p_out.nodes = new node[nodelist.size()];
  std::copy(nodelist.begin(),nodelist.end(),p_out.nodes);

  p_out.len = nodelist.size();
  p_out.metric = params.metric;
  p_out.depth = MAX_STACK_SIZE;
  p_out.raw_fitness_ = 0.0f; 
  p_out.mut_type = mutation_t::none;

  // for(int i=0;i<p_out.len;++i){
  //   CUML_LOG_DEBUG("Node #%d -> %d (%d inputs)",i+1,
  //                 static_cast<std::underlying_type<node::type>::type>(p_out.nodes[i].t)
  //                 ,p_out.nodes[i].arity());
  // }
  validate_program(p_out);
}

void point_mutation(const program &prog, program &p_out, const param& params, std::mt19937 &gen){
  
  // deep-copy program
  p_out = prog;
  
  // Specify RNGs
  std::uniform_real_distribution<float> dist_U(0.0f, 1.0f);
  std::uniform_int_distribution<int> dist_terminal(0,params.num_features);
  std::uniform_real_distribution<float> dist_constant(params.const_range[0],params.const_range[1]);

  // Fill with uniform numbers
  std::vector<float> node_probs(p_out.len);
  std::generate(node_probs.begin(),node_probs.end(),[&dist_U,&gen]{return dist_U(gen);});

  // Mutate nodes
  int len = p_out.len;
  for(int i=0;i<len;++i){

    node curr(prog.nodes[i]);

    if(node_probs[i] < params.p_point_replace){
      if(curr.is_terminal()){

        int choice = dist_terminal(gen);
        
        if(choice == params.num_features){
          // Add a randomly generated constant
          curr = node(dist_constant(gen));
        }
        else{
          // Add a variable with fid=choice
          curr = node(choice);
        }
      }
      else if(curr.is_nonterminal()){
        // Replace current function with another function of the same arity
        int ar = curr.arity();
        std::vector<node::type> fset = params.arity_set.at(ar);
        std::uniform_int_distribution<> dist_fset(0,fset.size()-1);
        int choice = dist_fset(gen);
        curr = node(fset[choice]);
      }

      // Update p_out with updated value
      p_out.nodes[i] = curr;
    }
  }

  validate_program(p_out);
}

void crossover(const program &prog, const program &donor, program &p_out, const param &params, std::mt19937 &gen){

  // CUML_LOG_DEBUG("Starting crossover");
  // Get a random subtree of prog to replace
  CUML_LOG_DEBUG("Parent subtree");
  std::pair<int, int> prog_slice = get_subtree(prog.nodes, prog.len, gen);
  int prog_start = prog_slice.first;
  int prog_end   = prog_slice.second;

  // Get subtree of donor
  CUML_LOG_DEBUG("Donor subtree");
  std::pair<int, int> donor_slice = get_subtree(donor.nodes, donor.len, gen);
  
  int donor_start = donor_slice.first;
  int donor_end   = donor_slice.second;

  // Evolve 
  p_out.len       = (prog_start) + (donor_end - donor_start) + (prog.len-prog_end);
  CUML_LOG_DEBUG("In crossover, par_len = %d, par_start = %d, par_end = %d",prog.len,prog_start,prog_end);
  CUML_LOG_DEBUG("In crossover, donor_len = %d, donor_start = %d, donor_end = %d",donor.len,donor_start,donor_end);
  CUML_LOG_DEBUG("In crossover, new length = %d",p_out.len);
  p_out.nodes     = new node[p_out.len];
  p_out.mut_type  = mutation_t::crossover;
  p_out.metric    = prog.metric;

  if(p_out.len >= (1 << MAX_STACK_SIZE)){
    CUML_LOG_DEBUG("Crossover tree produced is too big!!");
  }

  // CUML_LOG_DEBUG("Exiting crossover");
  // Copy slices using std::copy
  std::copy(prog.nodes,prog.nodes + prog_start,p_out.nodes);
  
  std::copy(donor.nodes + donor_start, donor.nodes + donor_end, p_out.nodes + prog_start);

  std::copy(prog.nodes + prog_end, 
            prog.nodes + prog.len, 
            p_out.nodes + (prog_start) + (donor_end - donor_start));
  
  validate_program(p_out);
}

void subtree_mutation(const program &prog, program &p_out, const param &params, std::mt19937 &gen){
  // Generate a random program and perform crossover
  program new_program;
  build_program(new_program,params,gen);
  crossover(prog,new_program,p_out,params,gen);

  if(p_out.len >= (1 << MAX_STACK_SIZE)){
    CUML_LOG_DEBUG("Subtree mutation tree produced is too big!!");
  }

  p_out.mut_type = mutation_t::subtree;
  validate_program(p_out);
}

void hoist_mutation(const program &prog, program &p_out, const param &params, std::mt19937 &gen){
  
  // Replace program subtree with a random sub-subtree

  std::pair<int, int> prog_slice = get_subtree(prog.nodes, prog.len, gen);
  int prog_start = prog_slice.first;
  int prog_end = prog_slice.second;

  std::pair<int,int> sub_slice = get_subtree(prog.nodes + prog_start,prog_end-prog_start,gen);
  int sub_start = sub_slice.first;
  int sub_end   = sub_slice.second;
  
  // Since subtree indices
  sub_start += prog_start;
  sub_end += prog_start;

  p_out.len = (prog_start) + (sub_end - sub_start) + (prog.len-prog_end);
  p_out.nodes = new node[p_out.len];
  p_out.mut_type = mutation_t::hoist;
  p_out.metric = prog.metric;

  if(p_out.len >= (1 << MAX_STACK_SIZE)){
    CUML_LOG_DEBUG("Hoist tree produced is too big!!");
  }

  CUML_LOG_DEBUG("In hoist, par_len = %d, par_start = %d, par_end = %d",prog.len,prog_start,prog_end);
  CUML_LOG_DEBUG("In hoist, slice_len = %d, slice_start = %d, slice_end = %d",sub_end-sub_start,sub_start,sub_end);
  CUML_LOG_DEBUG("In hoist, new length = %d",p_out.len);

  // Copy node slices using std::copy
  std::copy(prog.nodes, prog.nodes + prog_start, p_out.nodes);
  std::copy(prog.nodes + sub_start, prog.nodes + sub_end, p_out.nodes + prog_start);
  std::copy(prog.nodes + prog_end, prog.nodes + prog.len,
                                   p_out.nodes + (prog_start) + (sub_end - sub_start));
  validate_program(p_out);
}

} // namespace genetic
} // namespace cuml