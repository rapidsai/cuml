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

#pragma once

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <cuda_utils.h>
#include <utils.h>
#include <sstream>
#include <string>
#include <vector>

namespace MLCommon {
namespace Bench {

/** Main fixture to be inherited and used by all other benchmarks */
class Fixture : public ::benchmark::Fixture {
 public:
  Fixture(const std::string& name) : ::benchmark::Fixture() {
    this->SetName(name.c_str());
  }
  Fixture() = delete;

  void SetUp(const ::benchmark::State& state) override {
    CUDA_CHECK(cudaStreamCreate(&stream));
    allocateBuffers(state);
    int devId = 0;
    CUDA_CHECK(cudaGetDevice(&devId));
    int l2CacheSize = 0;
    CUDA_CHECK(
      cudaDeviceGetAttribute(&l2CacheSize, cudaDevAttrL2CacheSize, devId));
    if (l2CacheSize > 0) allocate(scratchBuffer, l2CacheSize);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void TearDown(const ::benchmark::State& state) override {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    deallocateBuffers(state);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceSynchronize());  // to be safe!
    CUDA_CHECK(cudaFree(scratchBuffer));
  }

  // to keep compiler happy
  void SetUp(::benchmark::State& st) override {
    SetUp(const_cast<const ::benchmark::State&>(st));
  }

  // to keep compiler happy
  void TearDown(::benchmark::State& st) override {
    TearDown(const_cast<const ::benchmark::State&>(st));
  }

 protected:
  // every benchmark should be overriding this
  virtual void runBenchmark(::benchmark::State& state) = 0;
  virtual void generateMetrics(::benchmark::State& state) {}
  virtual void allocateBuffers(const ::benchmark::State& state) {}
  virtual void deallocateBuffers(const ::benchmark::State& state) {}

  void BenchmarkCase(::benchmark::State& state) {
    runBenchmark(state);
    generateMetrics(state);
  }

  cudaStream_t stream;
  char* scratchBuffer;  // mostly used to force explicit L2 flushes
};                      // end class Fixture

/**
 * RAII way of timing cuda calls. This has been shamelessly copied from the
 * cudf codebase. So, credits for this class goes to cudf developers.
 */
struct CudaEventTimer {
 public:
  /**
   * @brief This ctor clears the L2 cache by cudaMemset'ing a buffer of the size
   *        of L2 and then starts the timer.
   * @param st the benchmark::State whose timer we are going to update.
   * @param ptr flush the L2 cache by writing to this buffer before every
   *            iteration. It is the responsibility of the caller to manage this
   *            buffer. Pass a `nullptr` if L2 flush is not needed.
   * @param s The CUDA stream we are measuring time on.
   */
  CudaEventTimer(::benchmark::State& st, char* ptr, cudaStream_t s)
    : state(&st), stream(s) {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    // flush L2?
    if (ptr != nullptr) {
      int devId = 0;
      CUDA_CHECK(cudaGetDevice(&devId));
      int l2CacheSize = 0;
      CUDA_CHECK(
        cudaDeviceGetAttribute(&l2CacheSize, cudaDevAttrL2CacheSize, devId));
      if (l2CacheSize > 0) {
        CUDA_CHECK(cudaMemsetAsync(ptr, sizeof(char) * l2CacheSize, 0, s));
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
    }
    CUDA_CHECK(cudaEventRecord(start, stream));
  }
  CudaEventTimer() = delete;

  /** 
   * @brief The dtor stops the timer and performs a synchroniazation. Time of
   *       the benchmark::State object provided to the ctor will be set  to the
   *       value given by `cudaEventElapsedTime()`.
   */
  ~CudaEventTimer() {
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    state->SetIterationTime(milliseconds / 1000.f);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
  }

 private:
  cudaEvent_t start;
  cudaEvent_t stop;
  ::benchmark::State* state;
  cudaStream_t stream;
};  // end namespace CudaEventTimer

namespace internal {
template <typename Params, typename Class>
struct Registrar {
  Registrar(const std::vector<Params>& paramsList, const std::string& name) {
    int counter = 0;
    for (const auto& param : paramsList) {
      std::stringstream oss;
      oss << counter;
      auto testName = name + "/" + oss.str();
      auto* b = ::benchmark::internal::RegisterBenchmarkInternal(
        new Class(testName, param));
      ///@todo: expose a currying-like interface to the final macro
      b->UseManualTime();
      b->Unit(benchmark::kMillisecond);
      ++counter;
    }
  }
};  // end struct Registrar
};  // end namespace internal

/**
 * This is the entry point macro for all ml-prims benchmarks. This needs to be
 * called for the set of benchmarks to be registered so that the main harness
 * inside google bench can find these benchmarks and run them.
 * @param ParamsClass a struct which contains all the parameters needed to
 *                    generate inputs and run the underlying prim on it.
 *                    Ideally, one such struct is needed for every ml-prim.
 * @param TestClass child class of `MLCommon::Bench::Fixture` which contains the
 *                  logic to generate the dataset and run training on it for a
 *                  given algo. Ideally, once such struct is needed for every
 *                  algo to be benchmarked
 * @param TestName a unique string to identify these tests at the end of run
 * @param params list of params upon which to benchmark the prim. It can be a
 *               statically populated vector or from the result of calling a
 *               function
 */
#define PRIMS_BENCH_REGISTER(ParamsClass, TestClass, TestName, params)       \
  static internal::Registrar<ParamsClass, TestClass> BENCHMARK_PRIVATE_NAME( \
    registrar)(params, #TestClass "/" TestName)

}  // end namespace Bench
}  // end namespace MLCommon
