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

#include <linalg/add.h>
#include "benchmark.cuh"

namespace MLCommon {
namespace Bench {
namespace LinAlg {

struct AddParams {
  int len;
};  // struct AddParams

template <typename T>
struct AddBench : public Fixture {
  AddBench(const std::string& name, const AddParams& p)
    : Fixture(name), params(p) {}

 protected:
  void allocateBuffers(const ::benchmark::State& state) override {
    allocate(ptr0, params.len, true);
    allocate(ptr1, params.len, true);
  }

  void deallocateBuffers(const ::benchmark::State& state) override {
    CUDA_CHECK(cudaFree(ptr0));
    CUDA_CHECK(cudaFree(ptr1));
  }

  void runBenchmark(::benchmark::State& state) override {
    for (auto _ : state) {
      CudaEventTimer timer(state, scratchBuffer, stream);
      MLCommon::LinAlg::add(ptr0, ptr0, ptr1, params.len, stream);
    }
  }

 private:
  AddParams params;
  T *ptr0, *ptr1;
};  // struct AddBench

static std::vector<AddParams> getInputs() {
  return {
    {256 * 1024 * 1024},
    {256 * 1024 * 1024 + 2},
    {256 * 1024 * 1024 + 1},
  };
}

PRIMS_BENCH_REGISTER(AddParams, AddBench<float>, "add", getInputs());
PRIMS_BENCH_REGISTER(AddParams, AddBench<double>, "add", getInputs());

}  // namespace LinAlg
}  // namespace Bench
}  // namespace MLCommon
