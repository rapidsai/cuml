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
#include <raft/linalg/add.hpp>

namespace MLCommon {
namespace Bench {
namespace LinAlg {

struct AddParams {
  int len;
};  // struct AddParams

template <typename T>
struct AddBench : public Fixture {
  AddBench(const std::string& name, const AddParams& p) : Fixture(name), params(p) {}

 protected:
  void allocateBuffers(const ::benchmark::State& state) override
  {
    alloc(ptr0, params.len, true);
    alloc(ptr1, params.len, true);
  }

  void deallocateBuffers(const ::benchmark::State& state) override
  {
    dealloc(ptr0, params.len);
    dealloc(ptr1, params.len);
  }

  void runBenchmark(::benchmark::State& state) override
  {
    loopOnState(state, [this]() { raft::linalg::add(ptr0, ptr0, ptr1, params.len, stream); });
  }

 private:
  AddParams params;
  T *ptr0, *ptr1;
};  // struct AddBench

static std::vector<AddParams> getInputs()
{
  return {
    {256 * 1024 * 1024},
    {256 * 1024 * 1024 + 2},
    {256 * 1024 * 1024 + 1},
  };
}

ML_BENCH_REGISTER(AddParams, AddBench<float>, "", getInputs());
ML_BENCH_REGISTER(AddParams, AddBench<double>, "", getInputs());

}  // namespace LinAlg
}  // namespace Bench
}  // namespace MLCommon
