/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

#include <cuml/random_projection/rproj_c.h>

#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/random/rng.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <sys/time.h>

#include <unordered_set>

const int TPB_X = 256;

inline void sample_without_replacement(size_t n_population,
                                       size_t n_samples,
                                       int* indices,
                                       size_t& indices_idx)
{
  std::random_device dev;
  std::mt19937 gen(dev());

  std::uniform_int_distribution<int> uni_dist(0, n_population - 1);

  std::unordered_set<int> s;

  for (size_t i = 0; i < n_samples; i++) {
    int rand_idx = uni_dist(gen);

    while (s.find(rand_idx) != s.end()) {
      rand_idx = uni_dist(gen);
    }
    s.insert(rand_idx);
    indices[indices_idx] = rand_idx;
    indices_idx++;
  }
}

__global__ void sum_bools(bool* in_bools, int n, int* out_val)
{
  int row = (blockIdx.x * TPB_X) + threadIdx.x;
  if (row < n) {
    bool v = in_bools[row];
    if (v) raft::myAtomicAdd(out_val, (int)in_bools[row]);
  }
}

inline size_t binomial(const raft::handle_t& h, size_t n, double p, int random_state)
{
  struct timeval tp;
  gettimeofday(&tp, NULL);
  long long seed = tp.tv_sec * 1000 + tp.tv_usec;

  auto rng = raft::random::Rng(random_state + seed);

  rmm::device_uvector<bool> rand_array(n, h.get_stream());
  rmm::device_scalar<int> successes(h.get_stream());

  rng.bernoulli(rand_array.data(), n, 1 - p, h.get_stream());

  cudaMemsetAsync(successes.data(), 0, sizeof(int), h.get_stream());

  dim3 grid_n(raft::ceildiv(n, (size_t)TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  sum_bools<<<grid_n, blk, 0, h.get_stream()>>>(rand_array.data(), n, successes.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  int ret = 0;
  raft::update_host(&ret, successes.data(), 1, h.get_stream());
  h.sync_stream();
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  return n - ret;
}

inline double check_density(double density, size_t n_features)
{
  if (density == -1.0) { return 1.0 / sqrt(n_features); }
  return density;
}

namespace ML {
/**
 * @brief computes minimum target dimension to preserve information according to error tolerance
 * (eps parameter)
 * @param[in] n_samples: number of samples
 * @param[in] eps: error tolerance
 * @return minimum target dimension
 */
size_t johnson_lindenstrauss_min_dim(size_t n_samples, double eps)
{
  ASSERT(eps > 0.0 && eps < 1.0, "Parameter eps: must be in range (0, 1)");
  ASSERT(n_samples > 0, "Parameter n_samples: must be strictly positive");

  double denominator = (pow(eps, 2.0) / 2.0) - (pow(eps, 3) / 3.0);
  size_t res         = 4.0 * log(n_samples) / denominator;
  return res;
}

inline void check_parameters(paramsRPROJ& params)
{
  ASSERT(params.n_components > 0, "Parameter n_components: must be strictly positive");

  ASSERT(params.n_features > 0, "Parameter n_features: must be strictly positive");

  ASSERT(params.n_features >= params.n_components,
         "Parameters n_features and n_components: n_features must superior "
         "or equal to n_components. If you set eps parameter, please modify its "
         "value."
         "\nCurrent values :\n\tn_features : %d\n\tn_components : %d\n\teps : %lf",
         params.n_features,
         params.n_components,
         params.eps);

  ASSERT(params.gaussian_method || (params.density > 0.0 && params.density <= 1.0),
         "Parameter density: must be in range (0, 1]");
}

inline void build_parameters(paramsRPROJ& params)
{
  if (params.n_components == -1) {
    params.n_components = johnson_lindenstrauss_min_dim(params.n_samples, params.eps);
  }
  if (!params.gaussian_method) {
    params.density = check_density(params.density, params.n_features);
  }
}
}  // namespace ML
