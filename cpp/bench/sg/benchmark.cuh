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
#include <utils.h>
#include <cuml/cuml.hpp>
#include <sstream>
#include <vector>
#include "dataset.cuh"

namespace ML {
namespace Bench {

/** Main fixture to be inherited and used by all algos in cuML benchmark */
class Fixture : public ::benchmark::Fixture {
 public:
  Fixture(const DatasetParams p) : ::benchmark::Fixture(), params(p) {}
  Fixture() = delete;

  void SetUp(const ::benchmark::State& state) override {
    CUDA_CHECK(cudaStreamCreate(&stream));
    handle.reset(new cumlHandle(NumStreams));
    handle->setStream(stream);
    allocateData(state);
    allocateBuffers(state);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void TearDown(const ::benchmark::State& state) override {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    deallocateBuffers(state);
    deallocateData(state);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceSynchronize());  // to be safe!
    handle.reset();
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
  virtual void allocateData(const ::benchmark::State& state) {}
  virtual void deallocateData(const ::benchmark::State& state) {}
  virtual void allocateBuffers(const ::benchmark::State& state) {}
  virtual void deallocateBuffers(const ::benchmark::State& state) {}

  void BenchmarkCase(::benchmark::State& state) {
    runBenchmark(state);
    generateMetrics(state);
  }

  DatasetParams params;
  std::unique_ptr<cumlHandle> handle;
  cudaStream_t stream;

  ///@todo: ideally, this should be determined at runtime based on the inputs
  ///       passed to the fixture. That will require a whole lot of plumbing of
  ///       interfaces. Thus, as a quick workaround, defining this static var.
  static const int NumStreams = 16;
};  // end class Fixture

/**
 * Fixture to be used for benchmarking classification algorithms when the input
 * suffices to be generated via `make_blobs`.
 */
template <typename D, typename L = int>
class BlobsFixture : public Fixture {
 public:
  BlobsFixture(const DatasetParams p, const BlobsParams b)
    : Fixture(p), bParams(b) {}
  BlobsFixture() = delete;

 protected:
  void allocateData(const ::benchmark::State& state) override {
    data.allocate(*handle, params);
    data.blobs(*handle, params, bParams);
  }

  void deallocateData(const ::benchmark::State& state) override {
    data.deallocate(*handle, params);
  }

  /** parameters passed to `make_blobs` */
  BlobsParams bParams;
  Dataset<D, L> data;
};  // end class BlobFixture

/**
 * Fixture to be used for benchmarking regression algorithms when the input
 * suffices to be generated via `make_regression`.
 */
template <typename D>
class RegressionFixture : public Fixture {
 public:
  RegressionFixture(const DatasetParams p, const RegressionParams r)
    : Fixture(p), rParams(r) {}
  RegressionFixture() = delete;

 protected:
  void allocateData(const ::benchmark::State& state) override {
    data.allocate(*handle, params);
    data.regression(*handle, params, rParams);
  }

  void deallocateData(const ::benchmark::State& state) override {
    data.deallocate(*handle, params);
  }

  /** parameters passed to `make_regression` */
  RegressionParams rParams;
  Dataset<D, D> data;
};  // end class RegressionFixture

/**
 * RAII way of timing cuda calls. This has been shamelessly copied from the
 * cudf codebase. So, credits for this class goes to cudf developers.
 */
struct CudaEventTimer {
 public:
  /**
   * @brief This ctor clears the L2 cache by cudaMemset'ing a buffer of the size
   *        of L2 and then starts the timer.
   * @param h cuml handle
   * @param st the benchmark::State whose timer we are going to update.
   * @param flushL2 whether or not to flush the L2 cache before every iteration.
   * @param s The CUDA stream we are measuring time on.
   */
  CudaEventTimer(const cumlHandle& h, ::benchmark::State& st, bool flushL2,
                 cudaStream_t s)
    : handle(h), state(&st), stream(s) {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    // flush L2?
    if (flushL2) {
      int devId = 0;
      CUDA_CHECK(cudaGetDevice(&devId));
      int l2CacheSize = 0;
      CUDA_CHECK(
        cudaDeviceGetAttribute(&l2CacheSize, cudaDevAttrL2CacheSize, devId));
      if (l2CacheSize > 0) {
        auto allocator = handle.getDeviceAllocator();
        auto* buffer = (int*)allocator->allocate(l2CacheSize, stream);
        CUDA_CHECK(cudaMemsetAsync(buffer, 0, l2CacheSize, stream));
        allocator->deallocate(buffer, l2CacheSize, stream);
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
  const cumlHandle& handle;
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
 * This is the entry point macro for all cuML benchmarks. This needs to be
 * called for the set of benchmarks to be registered so that the main harness
 * inside google bench can find these benchmarks and run them.
 * @param ParamsClass a struct which contains all the parameters needed to
 *                    generate a dataset and run the underlying ML training algo
 *                    on it. Ideally, one such struct is needed for every source
 * @param BaseClass the child class of `ML::Bench::Fixture` which contains the
 *                  logic to generate the dataset and run training on it for a
 *                  given algo. Ideally, once such struct is needed for every
 *                  algo to be benchmarked
 * @param BaseName a unique string to identify these tests at the end of run
 * @param params list of params upon which to benchmark the algo. It can be a
 *               statically populated vector or from the result of calling a
 *               function
 * @note See at the end of kmeans.cu for a real use-case example.
 */
#define CUML_BENCH_REGISTER(ParamsClass, BaseClass, BaseName, params)        \
  static internal::Registrar<ParamsClass, BaseClass> BENCHMARK_PRIVATE_NAME( \
    registrar)(params, #BaseClass "/" BaseName)

}  // end namespace Bench
}  // end namespace ML
