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

#pragma once

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <raft/cudart_utils.h>
#include <cuml/common/cuml_allocator.hpp>
#include <cuml/common/logger.hpp>
#include <cuml/common/utils.hpp>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace MLCommon {
namespace Bench {

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
   * @param ptr         flush the L2 cache by writing to this buffer before
   *                    every iteration. It is the responsibility of the caller
   *                    to manage this buffer. Pass a `nullptr` if L2 flush is
   *                    not needed.
   * @param l2CacheSize L2 Cache size (in B). Passing this as 0 also disables
   *                    the L2 cache flush.
   * @param s           CUDA stream we are measuring time on.
   */
  CudaEventTimer(::benchmark::State& st, char* ptr, int l2CacheSize,
                 cudaStream_t s)
    : state(&st), stream(s) {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    // flush L2?
    if (ptr != nullptr && l2CacheSize > 0) {
      CUDA_CHECK(cudaMemsetAsync(ptr, 0, sizeof(char) * l2CacheSize, s));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CUDA_CHECK(cudaEventRecord(start, stream));
  }
  CudaEventTimer() = delete;

  /**
   * @brief The dtor stops the timer and performs a synchroniazation. Time of
   *       the benchmark::State object provided to the ctor will be set to the
   *       value given by `cudaEventElapsedTime()`.
   */
  ~CudaEventTimer() {
    CUDA_CHECK_NO_THROW(cudaEventRecord(stop, stream));
    CUDA_CHECK_NO_THROW(cudaEventSynchronize(stop));
    float milliseconds = 0.0f;
    CUDA_CHECK_NO_THROW(cudaEventElapsedTime(&milliseconds, start, stop));
    state->SetIterationTime(milliseconds / 1000.f);
    CUDA_CHECK_NO_THROW(cudaEventDestroy(start));
    CUDA_CHECK_NO_THROW(cudaEventDestroy(stop));
  }

 private:
  ::benchmark::State* state;
  cudaStream_t stream;
  cudaEvent_t start;
  cudaEvent_t stop;
};  // end struct CudaEventTimer

/** Main fixture to be inherited and used by all other c++ benchmarks in cuml */
class Fixture : public ::benchmark::Fixture {
 public:
  Fixture(const std::string& name, std::shared_ptr<deviceAllocator> _alloc)
    : ::benchmark::Fixture(), d_alloc(_alloc) {
    SetName(name.c_str());
  }
  Fixture() = delete;

  void SetUp(const ::benchmark::State& state) override {
    CUDA_CHECK(cudaStreamCreate(&stream));
    allocateBuffers(state);
    int devId = 0;
    CUDA_CHECK(cudaGetDevice(&devId));
    l2CacheSize = 0;
    CUDA_CHECK(
      cudaDeviceGetAttribute(&l2CacheSize, cudaDevAttrL2CacheSize, devId));
    if (l2CacheSize > 0) {
      alloc(scratchBuffer, l2CacheSize, false);
    } else {
      scratchBuffer = nullptr;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void TearDown(const ::benchmark::State& state) override {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (l2CacheSize > 0) {
      dealloc(scratchBuffer, l2CacheSize);
    }
    deallocateBuffers(state);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
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

  template <typename Lambda>
  void loopOnState(::benchmark::State& state, Lambda benchmarkFunc,
                   bool flushL2 = true) {
    char* buff;
    int size;
    if (flushL2) {
      buff = scratchBuffer;
      size = l2CacheSize;
    } else {
      buff = nullptr;
      size = 0;
    }
    for (auto _ : state) {
      CudaEventTimer timer(state, buff, size, stream);
      benchmarkFunc();
    }
  }

  template <typename T>
  void alloc(T*& ptr, size_t len, bool init = false) {
    auto nBytes = len * sizeof(T);
    ptr = (T*)d_alloc->allocate(nBytes, stream);
    if (init) {
      CUDA_CHECK(cudaMemsetAsync(ptr, 0, nBytes, stream));
    }
  }

  template <typename T>
  void dealloc(T* ptr, size_t len) {
    d_alloc->deallocate(ptr, len * sizeof(T), stream);
  }

  std::shared_ptr<deviceAllocator> d_alloc;
  cudaStream_t stream;
  int l2CacheSize;
  char* scratchBuffer;
};  // class Fixture

namespace internal {
template <typename Params, typename Class>
struct Registrar {
  Registrar(const std::vector<Params>& paramsList, const std::string& testClass,
            const std::string& testName) {
    int counter = 0;
    for (const auto& param : paramsList) {
      std::stringstream oss;
      oss << testClass;
      if (!testName.empty()) oss << "/" << testName;
      oss << "/" << counter;
      auto testFullName = oss.str();
      auto* b = ::benchmark::internal::RegisterBenchmarkInternal(
        new Class(testFullName, param));
      ///@todo: expose a currying-like interface to the final macro
      b->UseManualTime();
      b->Unit(benchmark::kMillisecond);
      ++counter;
    }
  }
};  // end struct Registrar
};  // end namespace internal

/**
 * This is the entry point macro for all ml benchmarks. This needs to be called
 * for the set of benchmarks to be registered so that the main harness inside
 * google bench can find these benchmarks and run them.
 *
 * @param ParamsClass a struct which contains all the parameters needed to
 *                    generate inputs and run the underlying prim on it.
 *                    Ideally, one such struct is needed for every ml-prim.
 * @param TestClass   child class of `MLCommon::Bench::Fixture` which contains
 *                    the logic to generate the dataset and run training on it
 *                    for a given algo. Ideally, once such struct is needed for
 *                    every algo to be benchmarked
 * @param TestName    a unique string to identify these tests at the end of run
 *                    This is optional and if choose not to use this, pass an
 *                    empty string
 * @param params      list of params upon which to benchmark the prim. It can be
 *                    a statically populated vector or from the result of
 *                    calling a function
 */
#define ML_BENCH_REGISTER(ParamsClass, TestClass, TestName, params)   \
  static MLCommon::Bench::internal::Registrar<ParamsClass, TestClass> \
    BENCHMARK_PRIVATE_NAME(registrar)(params, #TestClass, TestName)

}  // end namespace Bench
}  // end namespace MLCommon
