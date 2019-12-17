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

#include <distance/distance.h>
#include "benchmark.cuh"

namespace MLCommon {
namespace Bench {
namespace Distance {

struct Params {
  int m, n, k;
};  // struct Params

template <typename T, MLCommon::Distance::DistanceType DType>
struct Distance : public Fixture {
  Distance(const std::string& name, const Params& p)
    : Fixture(name), params(p) {}

 protected:
  void allocateBuffers(const ::benchmark::State& state) override {
    allocate(x, params.m * params.k, true);
    allocate(y, params.n * params.k, true);
    allocate(out, params.m * params.n, true);
    workspace = nullptr;
    worksize = MLCommon::Distance::getWorkspaceSize<DType, T, T, T>(
      x, y, params.m, params.n, params.k);
    if (worksize != 0) {
      allocate(workspace, worksize);
    }
  }

  void deallocateBuffers(const ::benchmark::State& state) override {
    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(workspace));
  }

  void runBenchmark(::benchmark::State& state) override {
    typedef cutlass::Shape<8, 128, 128> OutputTile_t;
    for (auto _ : state) {
      CudaEventTimer timer(state, scratchBuffer, stream);
      MLCommon::Distance::distance<DType, T, T, T, OutputTile_t>(
        x, y, out, params.m, params.n, params.k, (void*)workspace, worksize,
        stream);
    }
  }

 private:
  Params params;
  T *x, *y, *out;
  char* workspace;
  size_t worksize;
};  // struct Distance

static std::vector<Params> getInputs() {
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

#define DIST_BENCH_REGISTER(Name, Metric)                         \
  using Name##F = Distance<float, Metric>;                        \
  PRIMS_BENCH_REGISTER(Params, Name##F, "distance", getInputs()); \
  using Name##D = Distance<double, Metric>;                       \
  PRIMS_BENCH_REGISTER(Params, Name##D, "distance", getInputs())

}  // namespace Distance
}  // namespace Bench
}  // namespace MLCommon
