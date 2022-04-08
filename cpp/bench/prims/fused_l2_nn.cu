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
#include <limits>
#include <raft/cudart_utils.h>
#include <raft/distance/fused_l2_nn.hpp>
#include <raft/handle.hpp>
#include <raft/linalg/norm.hpp>
#include <raft/random/rng.hpp>
#include <raft/spatial/knn/specializations.hpp>

namespace MLCommon {
namespace Bench {
namespace Distance {

struct FLNParams {
  int m, n, k;
};  // struct FLNParams

template <typename T>
struct FusedL2NN : public Fixture {
  FusedL2NN(const std::string& name, const FLNParams& p) : Fixture(name), params(p) {}

 protected:
  void allocateBuffers(const ::benchmark::State& state) override
  {
    alloc(x, params.m * params.k);
    alloc(y, params.n * params.k);
    alloc(xn, params.m);
    alloc(yn, params.n);
    alloc(out, params.m);
    alloc(workspace, params.m);
    raft::random::Rng r(123456ULL);
    raft::handle_t handle{stream};

    r.uniform(x, params.m * params.k, T(-1.0), T(1.0), stream);
    r.uniform(y, params.n * params.k, T(-1.0), T(1.0), stream);
    raft::linalg::rowNorm(xn, x, params.k, params.m, raft::linalg::L2Norm, true, stream);
    raft::linalg::rowNorm(yn, y, params.k, params.n, raft::linalg::L2Norm, true, stream);
    raft::distance::initialize<T, cub::KeyValuePair<int, T>, int>(
      handle, out, params.m, std::numeric_limits<T>::max(), op);
  }

  void deallocateBuffers(const ::benchmark::State& state) override
  {
    dealloc(x, params.m * params.k);
    dealloc(y, params.n * params.k);
    dealloc(xn, params.m);
    dealloc(yn, params.n);
    dealloc(out, params.m);
    dealloc(workspace, params.m);
  }

  void runBenchmark(::benchmark::State& state) override
  {
    loopOnState(state, [this]() {
      // it is enough to only benchmark the L2-squared metric
      raft::distance::fusedL2NN<T, cub::KeyValuePair<int, T>, int>(out,
                                                                   x,
                                                                   y,
                                                                   xn,
                                                                   yn,
                                                                   params.m,
                                                                   params.n,
                                                                   params.k,
                                                                   (void*)workspace,
                                                                   op,
                                                                   pairRedOp,
                                                                   false,
                                                                   false,
                                                                   stream);
    });
  }

 private:
  FLNParams params;
  T *x, *y, *xn, *yn;
  cub::KeyValuePair<int, T>* out;
  int* workspace;
  raft::distance::KVPMinReduce<int, T> pairRedOp;
  raft::distance::MinAndDistanceReduceOp<int, T> op;
};  // struct FusedL2NN

static std::vector<FLNParams> getInputs()
{
  return {
    {32, 16384, 16384},  {64, 16384, 16384},   {128, 16384, 16384},   {256, 16384, 16384},
    {512, 16384, 16384}, {1024, 16384, 16384}, {16384, 32, 16384},    {16384, 64, 16384},
    {16384, 128, 16384}, {16384, 256, 16384},  {16384, 512, 16384},   {16384, 1024, 16384},
    {16384, 16384, 32},  {16384, 16384, 64},   {16384, 16384, 128},   {16384, 16384, 256},
    {16384, 16384, 512}, {16384, 16384, 1024}, {16384, 16384, 16384},
  };
}

ML_BENCH_REGISTER(FLNParams, FusedL2NN<float>, "", getInputs());
ML_BENCH_REGISTER(FLNParams, FusedL2NN<double>, "", getInputs());

}  // namespace Distance
}  // namespace Bench
}  // namespace MLCommon
