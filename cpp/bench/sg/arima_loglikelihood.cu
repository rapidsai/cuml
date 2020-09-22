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

#include <cuml/cuml.hpp>

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <cuml/tsa/arima_common.h>
#include <cuml/tsa/batched_arima.hpp>
#include <random/rng.cuh>

#include "benchmark.cuh"

namespace ML {
namespace Bench {
namespace Arima {

struct ArimaParams {
  TimeSeriesParams data;
  ARIMAOrder order;
};

template <typename DataT>
class ArimaLoglikelihood : public TsFixtureRandom<DataT> {
 public:
  ArimaLoglikelihood(const std::string& name, const ArimaParams& p)
    : TsFixtureRandom<DataT>(name, p.data), order(p.order) {}

  // Note: public function because of the __device__ lambda
  void runBenchmark(::benchmark::State& state) override {
    using MLCommon::Bench::CudaEventTimer;

    auto& handle = *this->handle;
    auto stream = handle.get_stream();
    auto counting = thrust::make_counting_iterator(0);

    // Generate random parameters
    int N = order.complexity();
    MLCommon::Random::Rng gpu_gen(this->params.seed,
                                  MLCommon::Random::GenPhilox);
    gpu_gen.uniform(param, N * this->params.batch_size, -1.0, 1.0, stream);
    // Set sigma2 parameters to 1.0
    DataT* x = param;  // copy the object attribute for thrust
    thrust::for_each(thrust::cuda::par.on(stream), counting,
                     counting + this->params.batch_size,
                     [=] __device__(int bid) { x[(bid + 1) * N - 1] = 1.0; });

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Benchmark loop
    this->loopOnState(state, [this]() {
      // Evaluate log-likelihood
      batched_loglike(*this->handle, this->data.X, this->params.batch_size,
                      this->params.n_obs, order, param, loglike, residual, true,
                      false);
    });
  }

  void allocateBuffers(const ::benchmark::State& state) {
    Fixture::allocateBuffers(state);

    auto& handle = *this->handle;
    auto stream = handle.get_stream();
    auto allocator = handle.get_device_allocator();

    // Buffer for the model parameters
    param = (DataT*)allocator->allocate(
      order.complexity() * this->params.batch_size * sizeof(DataT), stream);

    // Buffers for the log-likelihood and residuals
    loglike = (DataT*)allocator->allocate(
      this->params.batch_size * sizeof(DataT), stream);
    residual = (DataT*)allocator->allocate(
      this->params.batch_size * this->params.n_obs * sizeof(DataT), stream);
  }

  void deallocateBuffers(const ::benchmark::State& state) {
    Fixture::deallocateBuffers(state);

    auto& handle = *this->handle;
    auto stream = handle.get_stream();
    auto allocator = handle.get_device_allocator();

    allocator->deallocate(
      param, order.complexity() * this->params.batch_size * sizeof(DataT),
      stream);
    allocator->deallocate(loglike, this->params.batch_size * sizeof(DataT),
                          stream);
    allocator->deallocate(
      residual, this->params.batch_size * this->params.n_obs * sizeof(DataT),
      stream);
  }

 protected:
  ARIMAOrder order;
  DataT* param;
  DataT* loglike;
  DataT* residual;
};

std::vector<ArimaParams> getInputs() {
  struct std::vector<ArimaParams> out;
  ArimaParams p;
  p.data.seed = 12345ULL;
  std::vector<ARIMAOrder> list_order = {{1, 1, 1, 0, 0, 0, 0, 0},
                                        {1, 1, 1, 1, 1, 1, 4, 0},
                                        {1, 1, 1, 1, 1, 1, 12, 0},
                                        {1, 1, 1, 1, 1, 1, 24, 0},
                                        {1, 1, 1, 1, 1, 1, 52, 0}};
  std::vector<int> list_batch_size = {10, 100, 1000, 10000};
  std::vector<int> list_n_obs = {200, 500, 1000};
  for (auto& order : list_order) {
    for (auto& batch_size : list_batch_size) {
      for (auto& n_obs : list_n_obs) {
        p.order = order;
        p.data.batch_size = batch_size;
        p.data.n_obs = n_obs;
        out.push_back(p);
      }
    }
  }
  return out;
}

ML_BENCH_REGISTER(ArimaParams, ArimaLoglikelihood<double>, "arima",
                  getInputs());

}  // namespace Arima
}  // namespace Bench
}  // namespace ML
