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
#include <rmm/device_uvector.hpp>

namespace cuml {
namespace genetic {

/**
 * Simultaneous execution of tournaments on the GPU, 
 * using online random number generation.
 */
__global__ void batched_tournament_kernel(program_t programs, 
                                          int* win_indices,
                                          int* seed,
                                          int num_progs,
                                          int tournament_size,
                                          int criterion){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  raft::random::detail::PhiloxGenerator gen((uint64_t)seed[idx],(uint64_t)idx,0);
  int r;
  gen.next(r);
  int optimum = r % num_progs;
  float opt_score = programs[optimum].raw_fitness_;

  for (int s = 1; s < tournament_size ; ++s){
    gen.next(r);
    int curr = r % num_progs;
    float curr_score = programs[curr].raw_fitness_;

    if (criterion == 0) { // min is better
      optimum = curr_score < opt_score ? curr : optimum;
    } else {
      optimum = curr_score > opt_score ? curr : optimum;
    }
  }
  
  win_indices[idx] = optimum;
}

/**
Driver function for tournaments and program fitness calculation
*/
void parallel_evolve(const raft::handle_t &h, program_t old_progs, 
                     float* data, float* y, float* w, 
                     int num_progs, int tournament_size, int init_seed){
  cudaStream_t stream = h.get_stream();
  // Generate seeds : todo: Find a better way for random seed generation.
  rmm::device_uvector<int> seed(num_progs,stream);
  rmm::device_uvector<int> winners(num_progs,stream);
  raft::random::Rng seedgen((uint64_t)init_seed);
  seedgen.uniformInt(seed.data(), num_progs, 1, num_progs * tournament_size, stream);
  
  int criterion = 0;
  // Perform a tournament
  batched_tournament_kernel<<<raft::ceildiv(num_progs,GENE_TPB),
                              GENE_TPB,0,stream>>>(
                                old_progs,
                                winners.data(),
                                seed.data(),
                                num_progs,
                                tournament_size,
                                criterion
                              );
  CUDA_CHECK(cudaPeekAtLastError());
}
  
float param::p_reproduce() const { return detail::p_reproduce(*this); }

int param::max_programs() const { return detail::max_programs(*this); }

}  // namespace genetic
}  // namespace cuml
