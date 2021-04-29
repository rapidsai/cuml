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
#include "node.cuh"

#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/random/rng_impl.cuh>
#include <raft/random/rng.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/linalg/binary_op.cuh>
#include <random>
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
    }
  }
  else{
    // Set mutation type
    float mut_probs[4];
    mut_probs[0] = params.p_crossover;
    mut_probs[1] = mut_probs[0] + params.p_subtree_mutation;
    mut_probs[2] = mut_probs[1] + params.p_hoist_mutation;    
    mut_probs[3] = mut_probs[2] + params.p_point_mutation;

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

    // Perform host mutations
    auto donor_pos  = n_progs;
    for(auto pos=0; pos < n_progs; ++pos) {
      
      auto parent_index = d_win_indices.element(pos, stream);
      
      if(h_nextprogs[pos].mut_type == mutation_t::crossover){
        // Get secondary index
        auto donor_index = d_win_indices.element(donor_pos, stream);
        donor_pos++; 
        crossover(h_oldprogs[parent_index], h_oldprogs[donor_index],h_nextprogs[pos], params, h_gen);
      }
      else if(h_nextprogs[pos].mut_type == mutation_t::subtree){
        subtree_mutation(h_oldprogs[parent_index],h_nextprogs[pos],params, h_gen);
      }
      else if(h_nextprogs[pos].mut_type == mutation_t::hoist){
        hoist_mutation(h_oldprogs[parent_index],h_nextprogs[pos],params,h_gen);
      }
      else if(h_nextprogs[pos].mut_type == mutation_t::point){
        point_mutation(h_oldprogs[parent_index],h_nextprogs[pos],params,h_gen);
      }
      else if(h_nextprogs[pos].mut_type == mutation_t::reproduce){
        h_nextprogs[pos] = h_oldprogs[parent_index];
      }
      else{
        // Should not come here
      }
    }
  }

  /* Memcpy individual host nodes to device
     TODO: Find a better way to do this. 
     Possibilities include a copy utilizing multiple streams,
     a switch to a unified memory model, or a SoA representation 
     for all programs */
  for(auto i=0;i<n_progs;++i) {
    program_t tmp      = new program(h_nextprogs[i], false);        
    tmp->nodes         = (node*)h.get_device_allocator()->allocate(tmp->len*sizeof(node),stream);

    CUDA_CHECK(cudaMemcpyAsync(tmp->nodes, h_nextprogs[i].nodes,
                                h_nextprogs[i].len * sizeof(node),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync( d_nextprogs + i, tmp,
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


void symFit(const raft::handle_t &handle, const float* input, const float* labels, 
            const float* sample_weights, const int n_rows, const int n_cols, param &params, 
            program_t final_progs, std::vector<std::vector<program>> &history) {
  cudaStream_t stream = handle.get_stream();
  
  /* Initializations */
  history.reserve(params.generations);
  
  std::vector<program> h_currprogs(params.population_size);
  std::vector<program> h_nextprogs(params.population_size);
  std::vector<float> h_fitness(params.population_size);
  // std::vector<float> h_lengths(params.population_size);
  program_t d_currprogs = (program_t)handle.get_device_allocator()->allocate(params.population_size*sizeof(program),stream);
  program_t d_nextprogs = final_progs;                      // Reuse memory already allocated for final_progs

  std::mt19937_64 h_gen_engine(params.random_state);
  std::uniform_int_distribution<int> seed_dist;
  
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
    if( (crit==0 && opt_fit <= params.stopping_criteria) || 
        (crit==1 && opt_fit >= params.stopping_criteria)) {
      break;
    }
  }

  /* Set return values */
  final_progs = d_currprogs;

  std::uniform_int_distribution<uint64_t> rand_state_gen;
  params.random_state = rand_state_gen(h_gen_engine);     // Update random state of hyperparams

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
  execute(handle,final_progs,n_rows,params.hall_of_fame,input,output);
}

}  // namespace genetic
}  // namespace cuml
