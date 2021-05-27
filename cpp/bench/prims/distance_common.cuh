/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <raft/cudart_utils.h>
#include <raft/distance/distance.cuh>
#include <raft/mr/device/allocator.hpp>
#include "common/ml_benchmark.hpp"

namespace MLCommon {
namespace Bench {
namespace Distance {

struct Params {
  int m, n, k;
  bool isRowMajor;
};  // struct Params

template <typename T, raft::distance::DistanceType DType>
struct Distance : public Fixture {
  Distance(const std::string& name, const Params& p)
    : Fixture(name, std::shared_ptr<raft::mr::device::allocator>(
                      new raft::mr::device::default_allocator)),
      params(p) {}

 protected:
  void allocateBuffers(const ::benchmark::State& state) override {
    alloc(x, params.m * params.k, true);
    alloc(y, params.n * params.k, true);
    alloc(out, params.m * params.n, true);
    workspace = nullptr;
    worksize = raft::distance::getWorkspaceSize<DType, T, T, T>(
      x, y, params.m, params.n, params.k);
    if (worksize != 0) {
      alloc(workspace, worksize, false);
    }
  }

  void deallocateBuffers(const ::benchmark::State& state) override {
    dealloc(x, params.m * params.k);
    dealloc(y, params.n * params.k);
    dealloc(out, params.m * params.n);
    dealloc(workspace, worksize);
  }

  void runBenchmark(::benchmark::State& state) override {
    loopOnState(state, [this]() {
      raft::distance::distance<DType, T, T, T>(
        x, y, out, params.m, params.n, params.k, (void*)workspace, worksize,
        stream, params.isRowMajor);
    });
  }

 private:
  Params params;
  T *x, *y, *out;
  char* workspace;
  size_t worksize;
};  // struct Distance

static std::vector<Params> getInputs() {
  return {
    {32, 16384, 16384, true},    {64, 16384, 16384, true},
    {128, 16384, 16384, true},   {256, 16384, 16384, true},
    {512, 16384, 16384, true},   {1024, 16384, 16384, true},
    {16384, 32, 16384, true},    {16384, 64, 16384, true},
    {16384, 128, 16384, true},   {16384, 256, 16384, true},
    {16384, 512, 16384, true},   {16384, 1024, 16384, true},
    {16384, 16384, 32, true},    {16384, 16384, 64, true},
    {16384, 16384, 128, true},   {16384, 16384, 256, true},
    {16384, 16384, 512, true},   {16384, 16384, 1024, true},
    {16384, 16384, 16384, true}, {32, 16384, 16384, false},
    {64, 16384, 16384, false},   {128, 16384, 16384, false},
    {256, 16384, 16384, false},  {512, 16384, 16384, false},
    {1024, 16384, 16384, false}, {16384, 32, 16384, false},
    {16384, 64, 16384, false},   {16384, 128, 16384, false},
    {16384, 256, 16384, false},  {16384, 512, 16384, false},
    {16384, 1024, 16384, false}, {16384, 16384, 32, false},
    {16384, 16384, 64, false},   {16384, 16384, 128, false},
    {16384, 16384, 256, false},  {16384, 16384, 512, false},
    {16384, 16384, 1024, false}, {16384, 16384, 16384, false},
  };
}

#define DIST_BENCH_REGISTER(Name, Metric)              \
  using Name##F = Distance<float, Metric>;             \
  ML_BENCH_REGISTER(Params, Name##F, "", getInputs()); \
  using Name##D = Distance<double, Metric>;            \
  ML_BENCH_REGISTER(Params, Name##D, "", getInputs())

}  // namespace Distance
}  // namespace Bench
}  // namespace MLCommon
