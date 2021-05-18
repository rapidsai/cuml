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

#include "constants.h"
#include "genetic.cuh"
#include <cuml/genetic/common.h>
#include <cuml/genetic/genetic.h>
#include <cuml/genetic/program.h>
#include <cuml/common/logger.hpp>
#include "node.cuh"

#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/random/rng_impl.cuh>
#include <raft/random/rng.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/linalg/binary_op.cuh>

#include <random>
#include <stack>
#include <algorithm>
#include <numeric>

#include <rmm/device_uvector.hpp>

namespace cuml {
namespace genetic {

/**
 * @brief Simultaneously execute tournaments using online random number generation.
 * 
 * @param progs         Device pointer to programs
 * @param win_indices   Winning indices for every tournament
 * @param seeds         Init seeds for choice selection
 * @param n_progs       Number of programs
 * @param n_tours       No of tournaments to be conducted
 * @param tour_size     No of programs considered per tournament(@c <=n_progs><)
 * @param criterion     Selection criterion for choices(min/max)
 */
__global__ void batched_tournament_kernel(const program_t progs, 
                                          int* win_indices, uint64_t* seeds, 
                                          const uint64_t n_progs, const uint64_t n_tours, 
                                          const uint64_t tour_size, const int criterion) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n_tours) {
    raft::random::detail::PhiloxGenerator gen(seeds[idx],(uint64_t)idx,0);
   
    int r;
    gen.next(r);
    int opt = r % n_progs;
    float opt_score = progs[opt].raw_fitness_;

    for (int s = 1; s < tour_size ; ++s){
      gen.next(r);
      int curr = r % n_progs;
      float curr_score = progs[curr].raw_fitness_;
      
      // Reduce thread divergence
      // criterion = 0 if min is better
      if(opt_score < curr_score) {
        opt = (1 - criterion)*opt + criterion*curr;
      }
      else {
        opt = criterion*opt + (1 - criterion)*curr;
      }

      opt_score = progs[opt].raw_fitness_;
    } 

    win_indices[idx] = opt;
  }
}

/**
 * @brief Driver function for evolving 1 generation
 * 
 * @param h               cuML handle
 * @param h_oldprogs      previous generation host programs
 * @param d_oldprogs      previous generation device programs
 * @param h_nextprogs     next generation host programs
 * @param d_nextprogs     next generation device programs
 * @param n_samples       No of samples in input dataset
 * @param data            Device pointer to input dataset
 * @param y               Device pointer to input predictions
 * @param sample_weights  Device pointer to input weights
 * @param params          Training hyperparameters
 * @param generation      Current generation id
 * @param seed            Random seed for generators
 */
void parallel_evolve(const raft::handle_t &h, 
                     const std::vector<program> &h_oldprogs, const program_t d_oldprogs, 
                     std::vector<program> &h_nextprogs, program_t d_nextprogs, 
                     const int n_samples, const float* data, const float* y, 
                     const float* sample_weights, const param &params, const int generation, const int seed) {

  CUML_LOG_DEBUG("Generation #%d",generation);
  cudaStream_t stream = h.get_stream();
  int n_progs    =   params.population_size;
  int tour_size  =   params.tournament_size;
  int n_tours    =   n_progs;                                 // at least num_progs tournaments

  // Seed engines
  std::mt19937 h_gen(seed);                    // CPU engine
  raft::random::Rng d_gen(seed);               // GPU engine

  std::uniform_real_distribution<float> dist_01(0.0f,1.0f);
  
  // Build, Mutate and Run Tournaments
  if(generation == 1){
    // Build random programs for the first generation
    for(auto i=0; i<n_progs; ++i){
      
      build_program(h_nextprogs[i],params,h_gen);
      CUML_LOG_DEBUG("Gen #1, program #%d, len=%d",i,h_nextprogs[i].len);

      // for(int j=0;j<h_nextprogs[i].len;++j){
      //   CUML_LOG_DEBUG("Node #%d -> %d (%d inputs)",j+1,
      //                   static_cast<std::underlying_type<node::type>::type>(h_nextprogs[i].nodes[j].t)
      //                   ,h_nextprogs[i].nodes[j].arity());
      // }
      // CUML_LOG_DEBUG("Program #%d len=%d",(i+1),h_nextprogs[i].len);
      // std::string eqn = stringify(h_nextprogs[i]);
      // std::cerr << eqn <<"\n";
      // exit(0);
    }
  }
  else{
    // Set mutation type
    float mut_probs[4];
    mut_probs[0] = params.p_crossover;
    mut_probs[1] = params.p_subtree_mutation;
    mut_probs[2] = params.p_hoist_mutation;    
    mut_probs[3] = params.p_point_mutation;
    std::partial_sum(mut_probs,mut_probs+4,mut_probs);

    for(auto i=0; i < n_progs; ++i){
      float prob = dist_01(h_gen);

      if(prob < mut_probs[0]) {
        h_nextprogs[i].mut_type = mutation_t::crossover;
        n_tours++;
      }
      else if(prob < mut_probs[1]) {
        h_nextprogs[i].mut_type = mutation_t::subtree;
      }
      else if(prob < mut_probs[2]) {
        h_nextprogs[i].mut_type = mutation_t::hoist;
      } 
      else if(prob < mut_probs[3]) {
        h_nextprogs[i].mut_type = mutation_t::point;
      }
      else {
        h_nextprogs[i].mut_type = mutation_t::reproduce;
      }
    } 

    // Run tournaments
    // TODO: Find a better way for subset-seed generation
    rmm::device_uvector<uint64_t> tour_seeds(n_tours,stream);
    rmm::device_uvector<int> d_win_indices(n_tours,stream);
    d_gen.uniformInt(tour_seeds.data(), n_tours, (uint64_t)1, (uint64_t)INT_MAX, stream);
    int crit = params.criterion();
    dim3 nblks(raft::ceildiv(n_tours,GENE_TPB),1,1);
    batched_tournament_kernel<<<nblks,GENE_TPB,0,stream>>>(d_oldprogs,d_win_indices.data(),tour_seeds.data(),n_progs,n_tours,tour_size,crit);
    CUDA_CHECK(cudaPeekAtLastError());

    // Make sure tournaments have finished running before copying win indices
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUML_LOG_DEBUG("Finished tournament for Generation #%d",generation);
    
    // Perform host mutations
    auto donor_pos  = n_progs;
    for(auto pos=0; pos < n_progs; ++pos) {
      
      auto parent_index = d_win_indices.element(pos, stream);
      
      if(h_nextprogs[pos].mut_type == mutation_t::crossover){
        // Get secondary index
        auto donor_index = d_win_indices.element(donor_pos, stream);
        donor_pos++; 
        CUML_LOG_DEBUG("Gen #%d, program #%d, parent = #%d, donor = #%d - crossover",generation,pos,parent_index,donor_index);
        crossover(h_oldprogs[parent_index], h_oldprogs[donor_index],h_nextprogs[pos], params, h_gen);
      }
      else if(h_nextprogs[pos].mut_type == mutation_t::subtree){
        CUML_LOG_DEBUG("Gen #%d, program #%d, parent = #%d - subtree",generation,pos,parent_index);
        subtree_mutation(h_oldprogs[parent_index],h_nextprogs[pos],params, h_gen);
      }
      else if(h_nextprogs[pos].mut_type == mutation_t::hoist){
        CUML_LOG_DEBUG("Gen #%d, program #%d, parent = #%d - hoist",generation,pos,parent_index);
        hoist_mutation(h_oldprogs[parent_index],h_nextprogs[pos],params,h_gen);
      }
      else if(h_nextprogs[pos].mut_type == mutation_t::point){
        CUML_LOG_DEBUG("Gen #%d, program #%d, parent = #%d - point mut",generation,pos,parent_index);
        point_mutation(h_oldprogs[parent_index],h_nextprogs[pos],params,h_gen);
      }
      else if(h_nextprogs[pos].mut_type == mutation_t::reproduce){
        CUML_LOG_DEBUG("Gen #%d, program #%d, parent = #%d - reproduce",generation,pos,parent_index);
        h_nextprogs[pos] = h_oldprogs[parent_index];
      }
      else{
        // Should not come here
      }

      // for(int j=0;j<h_nextprogs[pos].len;++j){
      //   CUML_LOG_DEBUG("Node #%d -> %d (%d inputs)",j+1,
      //                   static_cast<std::underlying_type<node::type>::type>(h_nextprogs[pos].nodes[j].t)
      //                   ,h_nextprogs[pos].nodes[j].arity());
      // }
      // std::string eqn = stringify(h_nextprogs[pos]);
      // std::cerr << eqn <<std::endl;
    }
  }

  /* Memcpy individual host nodes to device
     TODO: Find a better way to do this. 
     Possibilities include a copy utilizing multiple streams,
     a switch to a unified memory model, or a SoA representation 
     for all programs */
  for(auto i=0;i<n_progs;++i) {
    program tmp(h_nextprogs[i], false);        
    tmp.nodes = (node*)h.get_device_allocator()->allocate(h_nextprogs[i].len*sizeof(node),stream);

    CUDA_CHECK(cudaMemcpyAsync(tmp.nodes, h_nextprogs[i].nodes,
                                h_nextprogs[i].len * sizeof(node),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync( d_nextprogs + i, &tmp,
                                sizeof(program),
                                cudaMemcpyHostToDevice,stream));
  }
  
  // Make sure all copying is done
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Update fitness
  set_batched_fitness(h, d_nextprogs, h_nextprogs, params, n_samples, data, y, sample_weights);
}
  
float param::p_reproduce() const { return detail::p_reproduce(*this); }

int param::max_programs() const { return detail::max_programs(*this); }

int param::criterion() const { return detail::criterion(*this); }

std::string stringify(const program &prog){
  std::string eqn = "";
  std::string delim = "";
  std::stack<int> ar_stack;
  for(int i=0;i<prog.len;++i){
    if(prog.nodes[i].t == node::type::constant){
      eqn += delim;
      eqn += std::to_string(prog.nodes[i].u.val);
      eqn += " ";
      if(!ar_stack.empty()){
        int stop = ar_stack.top();
        ar_stack.pop();
        if(stop > 1){ar_stack.push(stop - 1);}else{eqn += ") ";}
      }
      delim = ", ";
    }
    else if(prog.nodes[i].t == node::type::variable){
      eqn += delim;
      // CUML_LOG_DEBUG("Got variable %d",prog.nodes[i].u.fid);
      eqn += "X";
      eqn += std::to_string(prog.nodes[i].u.fid);
      eqn += " ";
      if(!ar_stack.empty()){
        int stop = ar_stack.top();
        ar_stack.pop();
        if(stop > 1){ar_stack.push(stop - 1);}else{eqn += ") ";}
      }
      delim = ", ";
    }
    else{
      ar_stack.push(prog.nodes[i].arity());
      eqn += delim;
      switch (prog.nodes[i].t){
        // binary operators
        case node::type::add:
          eqn += "add(";
          break;
        case node::type::atan2:
          eqn += "atan2(";
          break;
        case node::type::div:
          eqn += "div(";
          break;
        case node::type::fdim:
          eqn += "fdim(";
          break;
        case node::type::max:
          eqn += "max(";
          break;
        case node::type::min:
          eqn += "min(";
          break;
        case node::type::mul:
          eqn += "mult(";
          break;
        case node::type::pow:
          eqn += "pow(";
          break;
        case node::type::sub:
          eqn += "sub(";
          break;
        // unary operators
        case node::type::abs:
          eqn += "abs(";
          break;
        case node::type::acos:
          eqn += "acos(";
          break;
        case node::type::acosh:
          eqn += "acosh(";
          break;
        case node::type::asin:
          eqn += "asin(";
          break;
        case node::type::asinh:
          eqn += "asinh(";
          break;
        case node::type::atan:
          eqn += "atan(";
          break;
        case node::type::atanh:
          eqn += "atanh(";
          break;
        case node::type::cbrt:
          eqn += "cbrt(";
          break;
        case node::type::cos:
          eqn += "cos(";
          break;
        case node::type::cosh:
          eqn += "cosh(";
          break;
        case node::type::cube:
          eqn += "cube(";
          break;
        case node::type::exp:
          eqn += "exp(";
          break;
        case node::type::inv:
          eqn += "inv(";
          break;
        case node::type::log:
          eqn += "log(";
          break;
        case node::type::neg:
          eqn += "neg(";
          break;
        case node::type::rcbrt:
          eqn += "rcbrt(";
          break;
        case node::type::rsqrt:
          eqn += "rsqrt(";
          break;
        case node::type::sin:
          eqn += "sin(";
          break;
        case node::type::sinh:
          eqn += "sinh(";
          break;
        case node::type::sq:
          eqn += "sq(";
          break;
        case node::type::sqrt:
          eqn += "sqrt(";
          break;
        case node::type::tan:
          eqn += "tan(";
          break;
        case node::type::tanh:
          eqn += "tanh(";
          break;
        default:
          break;
      }
      eqn += " ";
      delim = "";
    }
  }
  if(prog.nodes[0].is_nonterminal()){
    eqn += ")";
  }
  return eqn;
}

void symFit(const raft::handle_t &handle, const float* input, const float* labels, 
            const float* sample_weights, const int n_rows, const int n_cols, param &params, 
            program_t final_progs, std::vector<std::vector<program>> &history) {
  cudaStream_t stream = handle.get_stream();
  
  // Update arity map in params - Need to do this only here, as all operations will call Fit atleast once
  for(auto f : params.function_set){
    int ar = 1;
    if(node::type::binary_begin <= f && f <= node::type::binary_end){ar = 2;}
    
    if(params.arity_set.find(ar) == params.arity_set.end()){
      // Create map entry for current arity
      std::vector<node::type> vec_f(1,f);
      params.arity_set.insert(std::make_pair(ar,vec_f));
    }
    else{
      // Insert into map
      std::vector<node::type> vec_f = params.arity_set.at(ar);
      if(std::find(vec_f.begin(),vec_f.end(),f) == vec_f.end()){
        params.arity_set.at(ar).push_back(f);
      }
    }
  } 

  /* Initializations */
  
  std::vector<program> h_currprogs(params.population_size);
  std::vector<program> h_nextprogs(params.population_size);
  std::vector<float> h_fitness(params.population_size);

  program_t d_currprogs = (program_t)handle.get_device_allocator()->allocate(params.population_size*sizeof(program),stream);
  program_t d_nextprogs = final_progs;     // Reuse memory already allocated for final_progs

  std::mt19937_64 h_gen_engine(params.random_state);
  std::uniform_int_distribution<int> seed_dist;
  
  CUML_LOG_DEBUG("# Programs = %d",params.population_size);

  /* Begin training */
  int gen = 0;
  for(;gen<params.generations;++gen){
    //  Evolve
    parallel_evolve(handle,h_currprogs,d_currprogs,h_nextprogs,d_nextprogs,
                  n_rows,input,labels,sample_weights,params,gen+1,seed_dist(h_gen_engine));
    
    history.push_back(h_nextprogs);
    
    // Swap
    h_currprogs.swap(h_nextprogs);
    program_t tmp = d_currprogs;
    d_currprogs = d_nextprogs;
    d_nextprogs = tmp;

    // Update fitness values
    float opt_fit = h_currprogs[0].raw_fitness_;
    int crit = params.criterion();
    for(int i=0;i<params.population_size;++i){
      h_fitness[i] = h_currprogs[i].raw_fitness_;
      if(crit == 0){
        opt_fit = std::min(opt_fit,h_fitness[i]);
      } else{
        opt_fit = std::max(opt_fit,h_fitness[i]);
      }
    }

    // Check for early stop
    if( (crit==0 && opt_fit <= params.stopping_criteria) || (crit==1 && opt_fit >= params.stopping_criteria)) {
      CUML_LOG_DEBUG("Early stopping reached. num_generations=%d",gen);
      break;
    }
  }

  /* Set return values */
  final_progs = d_currprogs;
}

void symRegPredict(const raft::handle_t &handle, const float* input, const int n_rows, 
                const program_t best_prog, float* output){
  // Assume best_prog is on device
  execute(handle,best_prog,n_rows,1,input,output);
}

void symClfPredictProbs(const raft::handle_t &handle, const float* input, const int n_rows,
                        const param &params, const program_t best_prog, float* output) {
  cudaStream_t stream = handle.get_stream();
  
  // Assume output is of shape [n_rows, 2] in colMajor format
  execute(handle,best_prog,n_rows,1,input,output);
  
  // Apply 2 map operations to get probabilities!
  // TODO: Modification needed for n_classes
  if(params.transformer == transformer_t::sigmoid) {
    raft::linalg::unaryOp(output+n_rows,output,n_rows,
                          [] __device__(float in){return 1.0f / (1.0f + expf(-in));}, stream);
    raft::linalg::unaryOp(output,output+n_rows,n_rows,
                          [] __device__(float in){return 1.0f-in;}, stream);
  }
  else {
    // Only sigmoid supported for now
  }
}

void symClfPredict(const raft::handle_t &handle, const float* input, const int n_rows, 
                   const param &params, const program_t best_prog, float* output){
  cudaStream_t stream = handle.get_stream();
  
  // Memory for probabilities
  float* probs = (float*)handle.get_device_allocator()->allocate(2*n_rows*sizeof(float),stream);
  symClfPredictProbs(handle,input,n_rows,params,best_prog,probs);

  // Take argmax along columns
  // TODO: Further modification needed for n_classes
  raft::linalg::binaryOp(output,probs,probs+n_rows,n_rows,
                         [] __device__ (float p0, float p1){
                           return 1.0f * (p0 <= p1);
                         },stream);
}

void symTransform(const raft::handle_t &handle, const float* input, const param &params, 
                  const program_t final_progs, const int n_rows, const int n_cols, float* output){
  cudaStream_t stream = handle.get_stream();
  // Execute final_progs(ordered by fitness) on input
  // output of size [n_rows,hall_of_fame]
  execute(handle,final_progs,n_rows,params.n_components,input,output);
}

}  // namespace genetic
}  // namespace cuml
