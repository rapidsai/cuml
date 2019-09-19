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
  }

  void SetUp(::benchmark::State& st) override {
    SetUp(const_cast<const ::benchmark::State&>(st));
  }

  void TearDown(::benchmark::State& st) override {
    TearDown(const_cast<const ::benchmark::State&>(st));
  }

 protected:
  // every benchmark should be overriding this
  virtual void runBenchmark(::benchmark::State& state) = 0;
  virtual void allocateData(const ::benchmark::State& state) {}
  virtual void deallocateData(const ::benchmark::State& state) {}
  virtual void allocateBuffers(const ::benchmark::State& state) {}
  virtual void deallocateBuffers(const ::benchmark::State& state) {}

  void BenchmarkCase(::benchmark::State& state) {
    runBenchmark(state);
  }

  DatasetParams params;
  std::unique_ptr<cumlHandle> handle;
  cudaStream_t stream;
}; // end class Fixture


/**
 * Fixture to be used for benchmarking classification algorithms when the input
 * suffices to be generated via `make_blobs`.
 */
template <typename D, typename L = int>
class BlobsFixture : public Fixture {
 public:
  BlobsFixture(const DatasetParams p, const BlobsParams b) :
    Fixture(p), bParams(b) {}
  BlobsFixture() = delete;

 protected:
  void allocateData(const ::benchmark::State& state) override {
    data.allocate(*handle, params);
    data.blobs(*handle, params, bParams);
  }

  void deallocateData(const ::benchmark::State& state) override {
    data.deallocate(*handle, params);
  }

  BlobsParams bParams;
  Dataset<D, L> data;
};

#define CUML_BENCH_PRIVATE_REGISTER_F(BaseClass, Name, TestName, ...)   \
  BENCHMARK_PRIVATE_DECLARE(TestName) =                                 \
    (::benchmark::internal::RegisterBenchmarkInternal(                  \
      new BaseClass(Name, __VA_ARGS__)))

#define CUML_BENCH_REGISTER_F(BaseClass, Method, ...)                   \
  CUML_BENCH_PRIVATE_REGISTER_F(BaseClass, #BaseClass "/" #Method,      \
                                BaseClass##_##Method##_Benchmark,       \
                                __VA_ARGS__)

#define CUML_BENCH_F(BaseClass, Method, ...)                    \
  BENCHMARK_F(BaseClass, Method)(::benchmark::State& st) {      \
    for (auto _ : st) {                                         \
      runBenchmark(st);                                         \
    }                                                           \
  }                                                             \
  CUML_BENCH_REGISTER_F(BaseClass, Method, __VA_ARGS__)

} // end namespace Bench
} // end namespace ML
