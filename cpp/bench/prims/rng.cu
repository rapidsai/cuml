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
#include <raft/random/rng.hpp>

namespace MLCommon {
namespace Bench {
namespace Random {

enum RandomType {
  RNG_Normal,
  RNG_LogNormal,
  RNG_Uniform,
  RNG_Gumbel,
  RNG_Logistic,
  RNG_Exp,
  RNG_Rayleigh,
  RNG_Laplace,
  RNG_Fill
};  // enum RandomType

template <typename T>
struct Params {
  int len;
  RandomType type;
  raft::random::GeneratorType gtype;
  T start, end;
};  // struct Params

template <typename T>
struct RngBench : public Fixture {
  RngBench(const std::string& name, const Params<T>& p) : Fixture(name), params(p) {}

 protected:
  void allocateBuffers(const ::benchmark::State& state) override { alloc(ptr, params.len); }

  void deallocateBuffers(const ::benchmark::State& state) override { dealloc(ptr, params.len); }

  void runBenchmark(::benchmark::State& state) override
  {
    raft::random::Rng r(123456ULL, params.gtype);
    loopOnState(state, [this, &r]() {
      switch (params.type) {
        case RNG_Normal: r.normal(ptr, params.len, params.start, params.end, stream); break;
        case RNG_LogNormal: r.lognormal(ptr, params.len, params.start, params.end, stream); break;
        case RNG_Uniform: r.uniform(ptr, params.len, params.start, params.end, stream); break;
        case RNG_Gumbel: r.gumbel(ptr, params.len, params.start, params.end, stream); break;
        case RNG_Logistic: r.logistic(ptr, params.len, params.start, params.end, stream); break;
        case RNG_Exp: r.exponential(ptr, params.len, params.start, stream); break;
        case RNG_Rayleigh: r.rayleigh(ptr, params.len, params.start, stream); break;
        case RNG_Laplace: r.laplace(ptr, params.len, params.start, params.end, stream); break;
        case RNG_Fill: r.fill(ptr, params.len, params.start, stream); break;
      };
    });
  }

 private:
  Params<T> params;
  T* ptr;
};  // struct RngBench

template <typename T>
static std::vector<Params<T>> getInputs()
{
  using namespace raft::random;
  return {
    {1024 * 1024, RNG_Uniform, GenPhilox, T(-1.0), T(1.0)},
    {32 * 1024 * 1024, RNG_Uniform, GenPhilox, T(-1.0), T(1.0)},
    {1024 * 1024 * 1024, RNG_Uniform, GenPhilox, T(-1.0), T(1.0)},
    {1024 * 1024 + 2, RNG_Uniform, GenPhilox, T(-1.0), T(1.0)},
    {32 * 1024 * 1024 + 2, RNG_Uniform, GenPhilox, T(-1.0), T(1.0)},
    {1024 * 1024 * 1024 + 2, RNG_Uniform, GenPhilox, T(-1.0), T(1.0)},
    {1024 * 1024 + 1, RNG_Uniform, GenPhilox, T(-1.0), T(1.0)},
    {32 * 1024 * 1024 + 1, RNG_Uniform, GenPhilox, T(-1.0), T(1.0)},
    {1024 * 1024 * 1024 + 1, RNG_Uniform, GenPhilox, T(-1.0), T(1.0)},

    {1024 * 1024, RNG_Uniform, GenPC, T(-1.0), T(1.0)},
    {32 * 1024 * 1024, RNG_Uniform, GenPC, T(-1.0), T(1.0)},
    {1024 * 1024 * 1024, RNG_Uniform, GenPC, T(-1.0), T(1.0)},
    {1024 * 1024 + 2, RNG_Uniform, GenPC, T(-1.0), T(1.0)},
    {32 * 1024 * 1024 + 2, RNG_Uniform, GenPC, T(-1.0), T(1.0)},
    {1024 * 1024 * 1024 + 2, RNG_Uniform, GenPC, T(-1.0), T(1.0)},
    {1024 * 1024 + 1, RNG_Uniform, GenPC, T(-1.0), T(1.0)},
    {32 * 1024 * 1024 + 1, RNG_Uniform, GenPC, T(-1.0), T(1.0)},
    {1024 * 1024 * 1024 + 1, RNG_Uniform, GenPC, T(-1.0), T(1.0)},

    {1024 * 1024, RNG_Fill, GenPhilox, T(-1.0), T(1.0)},
    {32 * 1024 * 1024, RNG_Fill, GenPhilox, T(-1.0), T(1.0)},
    {1024 * 1024 * 1024, RNG_Fill, GenPhilox, T(-1.0), T(1.0)},
    {1024 * 1024 + 2, RNG_Fill, GenPhilox, T(-1.0), T(1.0)},
    {32 * 1024 * 1024 + 2, RNG_Fill, GenPhilox, T(-1.0), T(1.0)},
    {1024 * 1024 * 1024 + 2, RNG_Fill, GenPhilox, T(-1.0), T(1.0)},
    {1024 * 1024 + 1, RNG_Fill, GenPhilox, T(-1.0), T(1.0)},
    {32 * 1024 * 1024 + 1, RNG_Fill, GenPhilox, T(-1.0), T(1.0)},
    {1024 * 1024 * 1024 + 1, RNG_Fill, GenPhilox, T(-1.0), T(1.0)},
  };
}

ML_BENCH_REGISTER(Params<float>, RngBench<float>, "", getInputs<float>());
ML_BENCH_REGISTER(Params<double>, RngBench<double>, "", getInputs<double>());

}  // namespace Random
}  // namespace Bench
}  // namespace MLCommon
