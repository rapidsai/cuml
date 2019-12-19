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

#include <random/permute.h>
#include <random/rng.h>
#include "benchmark.cuh"

namespace MLCommon {
namespace Bench {
namespace Random {

struct Params {
  int rows, cols;
  bool needPerms, needShuffle, rowMajor;
};  // struct Params

template <typename T>
struct Permute : public Fixture {
  Permute(const std::string& name, const Params& p)
    : Fixture(name), params(p) {}

 protected:
  void allocateBuffers(const ::benchmark::State& state) override {
    auto matLen = params.rows * params.cols;
    auto vecLen = params.rows;
    if (params.needPerms) {
      allocate(perms, vecLen);
    } else {
      perms = nullptr;
    }
    MLCommon::Random::Rng r(123456ULL);
    if (params.needShuffle) {
      allocate(out, matLen);
      allocate(in, matLen);
      r.uniform(in, vecLen, T(-1.0), T(1.0), stream);
    } else {
      out = in = nullptr;
    }
  }

  void deallocateBuffers(const ::benchmark::State& state) override {
    if (params.needShuffle) {
      CUDA_CHECK(cudaFree(out));
      CUDA_CHECK(cudaFree(in));
    }
    if (params.needPerms) {
      CUDA_CHECK(cudaFree(perms));
    }
  }

  void runBenchmark(::benchmark::State& state) override {
    MLCommon::Random::Rng r(123456ULL);
    for (auto _ : state) {
      CudaEventTimer timer(state, scratchBuffer, stream);
      MLCommon::Random::permute(perms, out, in, params.cols, params.rows,
                                params.rowMajor, stream);
    }
  }

 private:
  Params params;
  T *out, *in;
  int* perms;
};  // struct Permute

static std::vector<Params> getInputs() {
  return {
    {32 * 1024, 128, true, true, true},
    {1024 * 1024, 128, true, true, true},
    {32 * 1024, 128 + 2, true, true, true},
    {1024 * 1024, 128 + 2, true, true, true},
    {32 * 1024, 128 + 1, true, true, true},
    {1024 * 1024, 128 + 1, true, true, true},

    {32 * 1024, 128, true, true, false},
    {1024 * 1024, 128, true, true, false},
    {32 * 1024, 128 + 2, true, true, false},
    {1024 * 1024, 128 + 2, true, true, false},
    {32 * 1024, 128 + 1, true, true, false},
    {1024 * 1024, 128 + 1, true, true, false},
  };
}

PRIMS_BENCH_REGISTER(Params, Permute<float>, "permute", getInputs());
PRIMS_BENCH_REGISTER(Params, Permute<double>, "permute", getInputs());

}  // namespace Random
}  // namespace Bench
}  // namespace MLCommon
