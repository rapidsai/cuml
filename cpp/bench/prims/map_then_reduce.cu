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
#include <raft/linalg/map_then_reduce.hpp>

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
  MapThenReduce(const std::string& name, const Params& p) : Fixture(name), params(p) {}

 protected:
  void allocateBuffers(const ::benchmark::State& state) override
  {
    alloc(in, params.len, true);
    alloc(out, 1, true);
  }

  void deallocateBuffers(const ::benchmark::State& state) override
  {
    dealloc(in, params.len);
    dealloc(out, 1);
  }

  void runBenchmark(::benchmark::State& state) override
  {
    loopOnState(state, [this]() {
      raft::linalg::mapThenSumReduce(out, params.len, Identity<T>(), stream, in);
    });
  }

 private:
  Params params;
  T *out, *in;
};  // struct MapThenReduce

static std::vector<Params> getInputs()
{
  return {
    {1024 * 1024},
    {32 * 1024 * 1024},
    {1024 * 1024 * 1024},
    {1024 * 1024 + 2},
    {32 * 1024 * 1024 + 2},
    {1024 * 1024 * 1024 + 2},
    {1024 * 1024 + 1},
    {32 * 1024 * 1024 + 1},
    {1024 * 1024 * 1024 + 1},
  };
}

ML_BENCH_REGISTER(Params, MapThenReduce<float>, "", getInputs());
ML_BENCH_REGISTER(Params, MapThenReduce<double>, "", getInputs());

}  // namespace LinAlg
}  // namespace Bench
}  // namespace MLCommon
