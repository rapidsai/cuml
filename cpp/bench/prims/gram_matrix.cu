/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cuml/matrix/kernelparams.h>
#include <matrix/grammatrix.cuh>
#include <matrix/kernelfactory.cuh>
#include <memory>
#include <raft/random/rng.cuh>
#include <sstream>
#include <string>
#include <vector>
#include "../common/ml_benchmark.hpp"

namespace MLCommon {
namespace Bench {
namespace Matrix {

using namespace MLCommon::Matrix;

struct GramTestParams {
  int m;  // m parameter of the GEMM
  int k;  // k parameter of the GEMM
  int n;  // n parameter of the GEMM
  KernelParams kernel_params;
};  // struct GramTestParams

template <typename T>
struct GramMatrix : public Fixture {
  GramMatrix(const std::string& name, const GramTestParams& p)
    : Fixture(name, std::shared_ptr<deviceAllocator>(
                      new raft::mr::device::default_allocator)),
      params(p) {
    std::vector<std::string> kernel_names{"linear", "poly", "rbf", "tanh"};
    std::ostringstream oss;
    oss << name << "/" << kernel_names[p.kernel_params.kernel] << "/" << p.m
        << "x" << p.k << "x" << p.n;
    this->SetName(oss.str().c_str());

    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    kernel = std::unique_ptr<GramMatrixBase<T>>(
      KernelFactory<T>::create(p.kernel_params, cublas_handle));
  }

  ~GramMatrix() { CUBLAS_CHECK(cublasDestroy(cublas_handle)); }

 protected:
  void allocateBuffers(const ::benchmark::State& state) override {
    alloc(A, params.m * params.k);
    alloc(B, params.k * params.n);
    alloc(C, params.m * params.n);
    raft::random::Rng r(123456ULL);
    r.uniform(A, params.m * params.k, T(-1.0), T(1.0), stream);
    r.uniform(B, params.k * params.n, T(-1.0), T(1.0), stream);
  }

  void deallocateBuffers(const ::benchmark::State& state) override {
    dealloc(A, params.m * params.k);
    dealloc(B, params.k * params.n);
    dealloc(C, params.m * params.n);
  }

  void runBenchmark(::benchmark::State& state) override {
    if (!this->kernel) {
      state.SkipWithError("Kernel matrix is not initialized");
    }
    loopOnState(state, [this]() {
      (*this->kernel)(this->A, this->params.m, this->params.k, this->B,
                      this->params.n, this->C, this->stream);
    });
  }

 private:
  cublasHandle_t cublas_handle;
  std::unique_ptr<GramMatrixBase<T>> kernel;
  GramTestParams params;

  T* A;  // input matrix A, size [m * k]
  T* B;  // input matrix B, size [n * k]
  T* C;  // output matrix C, size [m*n]
};

static std::vector<GramTestParams> getInputs() {
  std::vector<GramTestParams> param_vec;
  std::vector<KernelParams> kernel_params{
    KernelParams{LINEAR, 3, 1, 0}, KernelParams{POLYNOMIAL, 2, 1.3, 1},
    KernelParams{TANH, 2, 0.5, 2.4}, KernelParams{RBF, 2, 0.5, 0}};
  struct TestSize {
    int m;
    int k;
    int n;
  };
  std::vector<TestSize> data_size{{4096, 10, 1024},    {4096, 100, 1024},
                                  {4096, 1000, 1024},  {4096, 10000, 1024},
                                  {100000, 10, 1024},  {100000, 100, 1024},
                                  {100000, 1000, 1024}};

  param_vec.reserve(kernel_params.size() * data_size.size());
  for (TestSize s : data_size) {
    for (auto kernel : kernel_params) {
      param_vec.push_back(GramTestParams{s.m, s.k, s.n, kernel});
    }
  }
  return param_vec;
}

ML_BENCH_REGISTER(GramTestParams, GramMatrix<float>, "", getInputs());
ML_BENCH_REGISTER(GramTestParams, GramMatrix<double>, "", getInputs());

}  // namespace Matrix
}  // namespace Bench
}  // namespace MLCommon
