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

#include "benchmark.cuh"
#include <linalg/coalesced_reduction.h>

namespace MLCommon {
namespace Bench {
namespace LinAlg {

struct Params {
  int rows, cols;
};  // struct Params

template <typename T>
struct CoalescedReduction : public Fixture {
  CoalescedReduction(const std::string& name, const Params& p) : Fixture(name),
                                                                 params(p) {}

 protected:
  void allocateBuffers(const ::benchmark::State& state) override {
    allocate(data, params.rows * params.cols, true);
    allocate(dots, params.rows, true);
  }

  void deallocateBuffers(const ::benchmark::State& state) override {
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(dots));
  }

  void runBenchmark(::benchmark::State& state) override {
    for (auto _ : state) {
      CudaEventTimer timer(state, true, stream);
      MLCommon::LinAlg::coalescedReduction(dots, data, params.cols, params.rows,
                                           T(0.f), stream);
    }
  }

 private:
  Params params;
  T *data, *dots;
};  // struct CoalescedReduction

static std::vector<Params> getInputs() {
  return {
    {8 * 1024, 1024},
    {1024, 8 * 1024},
    {8 * 1024, 8 * 1024},
    {32 * 1024, 1024},
    {1024, 32 * 1024},
    {32 * 1024, 32 * 1024},
  };
}

PRIMS_BENCH_REGISTER(Params, CoalescedReduction<float>, "coalescedReduction",
                     getInputs());
PRIMS_BENCH_REGISTER(Params, CoalescedReduction<double>, "coalescedReduction",
                     getInputs());

}  // namespace LinAlg
}  // namespace Bench
}  // namespace MLCommon
