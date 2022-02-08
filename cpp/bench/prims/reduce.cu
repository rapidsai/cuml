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
#include <raft/linalg/reduce.hpp>

namespace MLCommon {
namespace Bench {
namespace LinAlg {

struct Params {
  int rows, cols;
  bool alongRows;
};  // struct Params

template <typename T>
struct Reduce : public Fixture {
  Reduce(const std::string& name, const Params& p) : Fixture(name), params(p) {}

 protected:
  void allocateBuffers(const ::benchmark::State& state) override
  {
    alloc(data, params.rows * params.cols, true);
    alloc(dots, params.rows, true);
  }

  void deallocateBuffers(const ::benchmark::State& state) override
  {
    dealloc(data, params.rows * params.cols);
    dealloc(dots, params.rows);
  }

  void runBenchmark(::benchmark::State& state) override
  {
    loopOnState(state, [this]() {
      raft::linalg::reduce(
        dots, data, params.cols, params.rows, T(0.f), true, params.alongRows, stream);
    });
  }

 private:
  Params params;
  T *data, *dots;
};  // struct Reduce

static std::vector<Params> getInputs()
{
  return {
    {8 * 1024, 1024, false},
    {1024, 8 * 1024, false},
    {8 * 1024, 8 * 1024, false},
    {32 * 1024, 1024, false},
    {1024, 32 * 1024, false},
    {32 * 1024, 32 * 1024, false},

    {8 * 1024, 1024, true},
    {1024, 8 * 1024, true},
    {8 * 1024, 8 * 1024, true},
    {32 * 1024, 1024, true},
    {1024, 32 * 1024, true},
    {32 * 1024, 32 * 1024, true},
  };
}

ML_BENCH_REGISTER(Params, Reduce<float>, "", getInputs());
ML_BENCH_REGISTER(Params, Reduce<double>, "", getInputs());

}  // namespace LinAlg
}  // namespace Bench
}  // namespace MLCommon
