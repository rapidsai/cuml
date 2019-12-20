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

struct Params {
  int m, n, k;
  bool sqrt;
};  // struct Params

template <typename T>
struct FusedL2NN : public Fixture {
  FusedL2NN(const std::string& name, const Params& p)
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
      MLCommon::Distance::fusedL2NN<T, cub::KeyValuePair<int, T>, int>(
        out, x, y, xn, yn, params.m, params.n, params.k, (void*)workspace, op,
        params.sqrt, false, stream);
    }
  }

 private:
  Params params;
  T *x, *y, *xn, *yn;
  cub::KeyValuePair<int, T>* out;
  int* workspace;
  MLCommon::Distance::MinAndDistanceReduceOp<int, T> op;
};  // struct FusedL2NN

static std::vector<Params> getInputs() {
  return {
    {32, 16384, 16384, true},     {64, 16384, 16384, true},
    {128, 16384, 16384, true},    {256, 16384, 16384, true},
    {512, 16384, 16384, true},    {1024, 16384, 16384, true},
    {16384, 32, 16384, true},     {16384, 64, 16384, true},
    {16384, 128, 16384, true},    {16384, 256, 16384, true},
    {16384, 512, 16384, true},    {16384, 1024, 16384, true},
    {16384, 16384, 32, true},     {16384, 16384, 64, true},
    {16384, 16384, 128, true},    {16384, 16384, 256, true},
    {16384, 16384, 512, true},    {16384, 16384, 1024, true},
    {16384, 16384, 16384, true},

    {32, 16384, 16384, false},    {64, 16384, 16384, false},
    {128, 16384, 16384, false},   {256, 16384, 16384, false},
    {512, 16384, 16384, false},   {1024, 16384, 16384, false},
    {16384, 32, 16384, false},    {16384, 64, 16384, false},
    {16384, 128, 16384, false},   {16384, 256, 16384, false},
    {16384, 512, 16384, false},   {16384, 1024, 16384, false},
    {16384, 16384, 32, false},    {16384, 16384, 64, false},
    {16384, 16384, 128, false},   {16384, 16384, 256, false},
    {16384, 16384, 512, false},   {16384, 16384, 1024, false},
    {16384, 16384, 16384, false},
  };
}

PRIMS_BENCH_REGISTER(Params, FusedL2NN<float>, "fusedL2NN", getInputs());
PRIMS_BENCH_REGISTER(Params, FusedL2NN<double>, "fusedL2NN", getInputs());

}  // namespace Distance
}  // namespace Bench
}  // namespace MLCommon
