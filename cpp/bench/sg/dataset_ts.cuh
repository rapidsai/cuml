/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cuda_utils.h>
#include <common/cumlHandle.hpp>
#include <cuml/cuml.hpp>

#include <random/rng.h>

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
  /** input data */
  DataT* X;

  /** allocate space needed for the dataset */
  void allocate(const cumlHandle& handle, const TimeSeriesParams& p) {
    auto allocator = handle.getDeviceAllocator();
    auto stream = handle.getStream();
    X = (DataT*)allocator->allocate(p.batch_size * p.n_obs * sizeof(DataT),
                                    stream);
  }

  /** free-up the buffers */
  void deallocate(const cumlHandle& handle, const TimeSeriesParams& p) {
    auto allocator = handle.getDeviceAllocator();
    auto stream = handle.getStream();
    allocator->deallocate(X, p.batch_size * p.n_obs * sizeof(DataT), stream);
  }

  /** generate random time series (normal distribution) */
  void random(const cumlHandle& handle, const TimeSeriesParams& p, DataT mu = 0,
              DataT sigma = 1) {
    MLCommon::Random::Rng gpu_gen(p.seed, MLCommon::Random::GenPhilox);
    gpu_gen.normal(X, p.batch_size * p.n_obs, mu, sigma, handle.getStream());
  }
};

}  // namespace Bench
}  // namespace ML
