/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <raft/core/handle.hpp>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

namespace ML {
namespace Bench {

/** General information about a time series dataset */
struct TimeSeriesParams {
  int batch_size;
  int n_obs;
  uint64_t seed;
};

/**
 * @brief A simple object to hold the loaded dataset for benchmarking
 * @tparam DataT type of the time series data
 */
template <typename DataT>
struct TimeSeriesDataset {
  TimeSeriesDataset() : X(0, rmm::cuda_stream_default) {}

  /** input data */
  rmm::device_uvector<DataT> X;

  /** allocate space needed for the dataset */
  void allocate(const raft::handle_t& handle, const TimeSeriesParams& p)
  {
    X.resize(p.batch_size * p.n_obs, handle.get_stream());
  }

  /** generate random time series (normal distribution) */
  void random(const raft::handle_t& handle,
              const TimeSeriesParams& p,
              DataT mu    = 0,
              DataT sigma = 1)
  {
    raft::random::Rng gpu_gen(p.seed, raft::random::GenPhilox);
    gpu_gen.normal(X.data(), p.batch_size * p.n_obs, mu, sigma, handle.get_stream());
  }
};

}  // namespace Bench
}  // namespace ML
