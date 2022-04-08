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
#include <raft/linalg/matrix_vector_op.hpp>

namespace MLCommon {
namespace Bench {
namespace LinAlg {

struct Params {
  int rows, cols;
  bool rowMajor, bcastAlongRows;
};  // struct Params

template <typename T>
struct MatVecOp : public Fixture {
  MatVecOp(const std::string& name, const Params& p) : Fixture(name), params(p) {}

 protected:
  void allocateBuffers(const ::benchmark::State& state) override
  {
    alloc(out, params.rows * params.cols, true);
    alloc(in, params.rows * params.cols, true);
    auto vecLen = params.bcastAlongRows ? params.cols : params.rows;
    alloc(vec, vecLen, true);
  }

  void deallocateBuffers(const ::benchmark::State& state) override
  {
    dealloc(out, params.rows * params.cols);
    dealloc(in, params.rows * params.cols);
    auto vecLen = params.bcastAlongRows ? params.cols : params.rows;
    dealloc(vec, vecLen);
  }

  void runBenchmark(::benchmark::State& state) override
  {
    loopOnState(state, [this]() {
      raft::linalg::matrixVectorOp(out,
                                   in,
                                   vec,
                                   params.cols,
                                   params.rows,
                                   params.rowMajor,
                                   params.bcastAlongRows,
                                   raft::Sum<T>(),
                                   stream);
    });
  }

 private:
  Params params;
  T *out, *in, *vec;
};  // struct MatVecOp

static std::vector<Params> getInputs()
{
  return {
    {1024, 128, true, true},       {1024 * 1024, 128, true, true},
    {1024, 128 + 2, true, true},   {1024 * 1024, 128 + 2, true, true},
    {1024, 128 + 1, true, true},   {1024 * 1024, 128 + 1, true, true},

    {1024, 128, true, false},      {1024 * 1024, 128, true, false},
    {1024, 128 + 2, true, false},  {1024 * 1024, 128 + 2, true, false},
    {1024, 128 + 1, true, false},  {1024 * 1024, 128 + 1, true, false},

    {1024, 128, false, false},     {1024 * 1024, 128, false, false},
    {1024, 128 + 2, false, false}, {1024 * 1024, 128 + 2, false, false},
    {1024, 128 + 1, false, false}, {1024 * 1024, 128 + 1, false, false},

    {1024, 128, false, true},      {1024 * 1024, 128, false, true},
    {1024, 128 + 2, false, true},  {1024 * 1024, 128 + 2, false, true},
    {1024, 128 + 1, false, true},  {1024 * 1024, 128 + 1, false, true},
  };
}

ML_BENCH_REGISTER(Params, MatVecOp<float>, "", getInputs());
ML_BENCH_REGISTER(Params, MatVecOp<double>, "", getInputs());

}  // namespace LinAlg
}  // namespace Bench
}  // namespace MLCommon
