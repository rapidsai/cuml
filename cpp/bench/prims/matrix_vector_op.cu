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

#include <linalg/matrix_vector_op.h>
#include "benchmark.cuh"

namespace MLCommon {
namespace Bench {
namespace LinAlg {

struct Params {
  int rows, cols;
  bool rowMajor, bcastAlongRows;
};  // struct Params

template <typename T>
struct MatVecOp : public Fixture {
  MatVecOp(const std::string& name, const Params& p)
    : Fixture(name), params(p) {}

 protected:
  void allocateBuffers(const ::benchmark::State& state) override {
    allocate(out, params.rows * params.cols, true);
    allocate(in, params.rows * params.cols, true);
    auto vecLen = params.bcastAlongRows ? params.cols : params.rows;
    allocate(vec, vecLen, true);
  }

  void deallocateBuffers(const ::benchmark::State& state) override {
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(in));
    CUDA_CHECK(cudaFree(vec));
  }

  void runBenchmark(::benchmark::State& state) override {
    for (auto _ : state) {
      CudaEventTimer timer(state, scratchBuffer, stream);
      MLCommon::LinAlg::matrixVectorOp(out, in, vec, params.cols, params.rows,
                                       params.rowMajor, params.bcastAlongRows,
                                       Sum<T>(), stream);
    }
  }

 private:
  Params params;
  T *out, *in, *vec;
};  // struct MatVecOp

static std::vector<Params> getInputs() {
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

PRIMS_BENCH_REGISTER(Params, MatVecOp<float>, "MatVecOp", getInputs());
PRIMS_BENCH_REGISTER(Params, MatVecOp<double>, "MatVecOp", getInputs());

}  // namespace LinAlg
}  // namespace Bench
}  // namespace MLCommon
