/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <common/ml_benchmark.hpp>
#include <raft/cudart_utils.h>
#include <raft/distance/distance.hpp>
#include <raft/distance/specializations.hpp>

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
    : Fixture(name), params(p), x(0, stream), y(0, stream), out(0, stream), workspace(0, stream)
  {
  }

 protected:
  void allocateBuffers(const ::benchmark::State& state) override
  {
    x.resize(params.m * params.k, stream);
    y.resize(params.n * params.k, stream);
    out.resize(params.m * params.n, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(x.data(), 0, x.size() * sizeof(T), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(y.data(), 0, y.size() * sizeof(T), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(out.data(), 0, out.size() * sizeof(T), stream));
    worksize = raft::distance::getWorkspaceSize<DType, T, T, T>(
      x.data(), y.data(), params.m, params.n, params.k);
    workspace.resize(worksize, stream);
  }

  void deallocateBuffers(const ::benchmark::State& state) override
  {
    x.release();
    y.release();
    out.release();
    workspace.release();
  }
  void runBenchmark(::benchmark::State& state) override
  {
    loopOnState(state, [this]() {
      raft::distance::distance<DType, T, T, T>(x.data(),
                                               y.data(),
                                               out.data(),
                                               params.m,
                                               params.n,
                                               params.k,
                                               (void*)workspace.data(),
                                               worksize,
                                               stream,
                                               params.isRowMajor);
    });
  }

 private:
  Params params;
  rmm::device_uvector<T> x, y, out;
  rmm::device_uvector<char> workspace;
  size_t worksize;
};  // struct Distance

static std::vector<Params> getInputs()
{
  return {
    {32, 16384, 16384, true},    {64, 16384, 16384, true},     {128, 16384, 16384, true},
    {256, 16384, 16384, true},   {512, 16384, 16384, true},    {1024, 16384, 16384, true},
    {16384, 32, 16384, true},    {16384, 64, 16384, true},     {16384, 128, 16384, true},
    {16384, 256, 16384, true},   {16384, 512, 16384, true},    {16384, 1024, 16384, true},
    {16384, 16384, 32, true},    {16384, 16384, 64, true},     {16384, 16384, 128, true},
    {16384, 16384, 256, true},   {16384, 16384, 512, true},    {16384, 16384, 1024, true},
    {16384, 16384, 16384, true}, {32, 16384, 16384, false},    {64, 16384, 16384, false},
    {128, 16384, 16384, false},  {256, 16384, 16384, false},   {512, 16384, 16384, false},
    {1024, 16384, 16384, false}, {16384, 32, 16384, false},    {16384, 64, 16384, false},
    {16384, 128, 16384, false},  {16384, 256, 16384, false},   {16384, 512, 16384, false},
    {16384, 1024, 16384, false}, {16384, 16384, 32, false},    {16384, 16384, 64, false},
    {16384, 16384, 128, false},  {16384, 16384, 256, false},   {16384, 16384, 512, false},
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
