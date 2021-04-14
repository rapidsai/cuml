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
#include <cuml/genetic/program.h>
#include "node.cuh"

#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/random/rng_impl.cuh>
#include <raft/random/rng.cuh>
#include <random>
#include <rmm/device_uvector.hpp>

namespace cuml {
namespace genetic {

/**
 * Simultaneous execution of tournaments on the GPU, 
 * using online random number generation.
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
 * Driver function for evolving 1 generation 
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
    for(auto i=0; i < n_progs; ++i){
      float prob = dist_01(h_gen);

      if(prob < params.p_crossover) {
        h_nextprogs[i].mut_type = mutation_t::crossover;
        n_tours++;
      }
      else if(prob < params.p_crossover + params.p_subtree_mutation) {
        h_nextprogs[i].mut_type = mutation_t::subtree;
      }
      else if(prob < params.p_crossover+params.p_subtree_mutation+params.p_hoist_mutation) {
        h_nextprogs[i].mut_type = mutation_t::hoist;
      } 
      else if(prob < params.p_crossover+params.p_subtree_mutation+params.p_hoist_mutation+params.p_point_mutation) {
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
     TODO: Find a better way to do this. Possibilities include a copy utilizing multiple streams,
     a switch to a unified memory model, or a Structure of Arrays representation 
     for all programs */
  for(auto i=0;i<n_progs;++i) {
    program_t tmp      = new program(h_nextprogs[i], false);        
    tmp->nodes         = (node*)h.get_device_allocator()->allocate(tmp->len*sizeof(node),stream);

    CUDA_CHECK(cudaMemcpyAsync(tmp->nodes, h_nextprogs[i].nodes,
                                h_nextprogs[i].len * sizeof(node),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync( &d_nextprogs[i], tmp,
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
}  // namespace genetic
}  // namespace cuml
