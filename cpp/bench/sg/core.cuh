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
#include <cuML.hpp>
#include <sstream>
#include <vector>
#include "dataset.cuh"
#include "utils.h"

namespace ML {
namespace Bench {

/** Main fixture to be inherited and used by all algos in cuML benchmark */
class Fixture : public ::benchmark::Fixture {
 public:
  Fixture(const DatasetParams p) : ::benchmark::Fixture(), params(p) {}
  Fixture() = delete;

  void SetUp(const ::benchmark::State& state) override {
    CUDA_CHECK(cudaStreamCreate(&stream));
    handle.reset(new cumlHandle);
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
      ///@todo: add custom functions here
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
