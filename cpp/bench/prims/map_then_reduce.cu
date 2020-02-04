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

#include <linalg/map_then_reduce.h>
#include "benchmark.cuh"

namespace MLCommon {
namespace Bench {
namespace LinAlg {

struct Params {
  int len;
};

template <typename Type>
struct Identity {
  HDI Type operator()(Type a) { return a; }
};

template <typename T>
struct MapThenReduce : public Fixture {
  MapThenReduce(const std::string& name, const Params& p)
    : Fixture(name), params(p) {}

 protected:
  void allocateBuffers(const ::benchmark::State& state) override {
    allocate(in, params.len, true);
    allocate(out, 1, true);
  }

  void deallocateBuffers(const ::benchmark::State& state) override {
    CUDA_CHECK(cudaFree(in));
    CUDA_CHECK(cudaFree(out));
  }

  void runBenchmark(::benchmark::State& state) override {
    for (auto _ : state) {
      CudaEventTimer timer(state, scratchBuffer, stream);
      MLCommon::LinAlg::mapThenSumReduce(out, params.len, Identity<T>(), stream,
                                         in);
    }
  }

 private:
  Params params;
  T *out, *in;
};  // struct MapThenReduce

static std::vector<Params> getInputs() {
  return {
    {1024 * 1024},     {32 * 1024 * 1024},     {1024 * 1024 * 1024},
    {1024 * 1024 + 2}, {32 * 1024 * 1024 + 2}, {1024 * 1024 * 1024 + 2},
    {1024 * 1024 + 1}, {32 * 1024 * 1024 + 1}, {1024 * 1024 * 1024 + 1},
  };
}

PRIMS_BENCH_REGISTER(Params, MapThenReduce<float>, "mapReduce", getInputs());
PRIMS_BENCH_REGISTER(Params, MapThenReduce<double>, "mapReduce", getInputs());

}  // namespace LinAlg
}  // namespace Bench
}  // namespace MLCommon
