/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <distance/fused_l2_nn.h>
#include <linalg/norm.h>
#include <random/rng.h>
#include <limits>
#include "benchmark.cuh"

namespace MLCommon {
namespace Bench {
namespace Distance {

struct FLNParams {
  int m, n, k;
};  // struct FLNParams

template <typename T>
struct FusedL2NN : public Fixture {
  FusedL2NN(const std::string& name, const FLNParams& p)
    : Fixture(name), params(p) {}

 protected:
  void allocateBuffers(const ::benchmark::State& state) override {
    allocate(x, params.m * params.k);
    allocate(y, params.n * params.k);
    allocate(xn, params.m);
    allocate(yn, params.n);
    allocate(out, params.m);
    allocate(workspace, params.m);
    MLCommon::Random::Rng r(123456ULL);
    r.uniform(x, params.m * params.k, T(-1.0), T(1.0), stream);
    r.uniform(y, params.n * params.k, T(-1.0), T(1.0), stream);
    MLCommon::LinAlg::rowNorm(xn, x, params.k, params.m,
                              MLCommon::LinAlg::L2Norm, true, stream);
    MLCommon::LinAlg::rowNorm(yn, y, params.k, params.n,
                              MLCommon::LinAlg::L2Norm, true, stream);
    auto blks = ceildiv(params.m, 256);
    MLCommon::Distance::initKernel<T, cub::KeyValuePair<int, T>, int>
      <<<blks, 256, 0, stream>>>(out, params.m, std::numeric_limits<T>::max(),
                                 op);
  }

  void deallocateBuffers(const ::benchmark::State& state) override {
    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));
    CUDA_CHECK(cudaFree(xn));
    CUDA_CHECK(cudaFree(yn));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(workspace));
  }

  void runBenchmark(::benchmark::State& state) override {
    for (auto _ : state) {
      CudaEventTimer timer(state, scratchBuffer, stream);
      // it is enough to only benchmark the L2-squared metric
      MLCommon::Distance::fusedL2NN<T, cub::KeyValuePair<int, T>, int>(
        out, x, y, xn, yn, params.m, params.n, params.k, (void*)workspace, op,
        false, false, stream);
    }
  }

 private:
  FLNParams params;
  T *x, *y, *xn, *yn;
  cub::KeyValuePair<int, T>* out;
  int* workspace;
  MLCommon::Distance::MinAndDistanceReduceOp<int, T> op;
};  // struct FusedL2NN

static std::vector<FLNParams> getInputs() {
  return {
    {32, 16384, 16384},    {64, 16384, 16384},  {128, 16384, 16384},
    {256, 16384, 16384},   {512, 16384, 16384}, {1024, 16384, 16384},
    {16384, 32, 16384},    {16384, 64, 16384},  {16384, 128, 16384},
    {16384, 256, 16384},   {16384, 512, 16384}, {16384, 1024, 16384},
    {16384, 16384, 32},    {16384, 16384, 64},  {16384, 16384, 128},
    {16384, 16384, 256},   {16384, 16384, 512}, {16384, 16384, 1024},
    {16384, 16384, 16384},
  };
}

PRIMS_BENCH_REGISTER(FLNParams, FusedL2NN<float>, "fusedL2NN", getInputs());
PRIMS_BENCH_REGISTER(FLNParams, FusedL2NN<double>, "fusedL2NN", getInputs());

}  // namespace Distance
}  // namespace Bench
}  // namespace MLCommon
