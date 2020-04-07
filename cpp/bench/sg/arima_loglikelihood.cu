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
#include <random/rng.h>
#include <cuml/tsa/batched_arima.hpp>

#include "benchmark.cuh"

namespace ML {
namespace Bench {
namespace Arima {

struct ArimaParams {
  TimeSeriesParams data;
  ARIMAOrder order;
};

template <typename D>
class ArimaLoglikelihood : public TsFixtureRandom<D> {
 public:
  ArimaLoglikelihood(const std::string& name, const ArimaParams& p)
    : TsFixtureRandom<D>(p.data), order(p.order) {
    this->SetName(name.c_str());
  }

  // Note: public function because of the __device__ lambda
  void runBenchmark(::benchmark::State& state) override {
    auto& handle = *this->handle;
    auto stream = handle.getStream();
    auto allocator = handle.getDeviceAllocator();
    auto counting = thrust::make_counting_iterator(0);

    // Generate random parameters
    int N = order.complexity();
    D* x =
      (D*)allocator->allocate(N * this->params.batch_size * sizeof(D), stream);
    MLCommon::Random::Rng gpu_gen(this->params.seed,
                                  MLCommon::Random::GenPhilox);
    gpu_gen.uniform(x, N * this->params.batch_size, -1.0, 1.0, stream);
    // Set sigma2 parameters to 1.0
    thrust::for_each(thrust::cuda::par.on(stream), counting,
                     counting + this->params.batch_size,
                     [=] __device__(int bid) { x[(bid + 1) * N - 1] = 1.0; });

    // Create arrays for log-likelihood and residual
    D* loglike =
      (D*)allocator->allocate(this->params.batch_size * sizeof(D), stream);
    D* residual = (D*)allocator->allocate(
      this->params.batch_size * (this->params.n_obs - order.lost_in_diff()) *
        sizeof(D),
      stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Benchmark loop
    for (auto _ : state) {
      CudaEventTimer timer(handle, state, true, stream);
      // Evaluate log-likelihood
      batched_loglike(handle, this->data.X, this->params.batch_size,
                      this->params.n_obs, order, x, loglike, residual, true,
                      false);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Clear memory
    allocator->deallocate(
      x, order.complexity() * this->params.batch_size * sizeof(D), stream);
    allocator->deallocate(loglike, this->params.batch_size * sizeof(D), stream);
    allocator->deallocate(residual,
                          this->params.batch_size *
                            (this->params.n_obs - order.lost_in_diff()) *
                            sizeof(D),
                          stream);
  }

 protected:
  ARIMAOrder order;
};

template <typename D>
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

CUML_BENCH_REGISTER(ArimaParams, ArimaLoglikelihood<double>, "arima",
                    getInputs<double>());

}  // namespace Arima
}  // namespace Bench
}  // namespace ML
