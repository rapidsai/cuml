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
#include <cuml/common/logger.hpp>
#include <cuml/cuml.hpp>
#include "../common/ml_benchmark.hpp"
#include "dataset.cuh"
#include "dataset_ts.cuh"

namespace ML {
namespace Bench {

/** Main fixture to be inherited and used by all algos in cuML benchmark */
class Fixture : public MLCommon::Bench::Fixture {
 public:
  Fixture(const std::string& name)
    : MLCommon::Bench::Fixture(name,
                               std::shared_ptr<deviceAllocator>(
                                 new raft::mr::device::default_allocator)) {}
  Fixture() = delete;

  void SetUp(const ::benchmark::State& state) override {
    handle.reset(new raft::handle_t(NumStreams));
    d_alloc = handle->get_device_allocator();
    MLCommon::Bench::Fixture::SetUp(state);
    handle->set_stream(stream);
  }

  void TearDown(const ::benchmark::State& state) override {
    MLCommon::Bench::Fixture::TearDown(state);
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
  virtual void allocateTempBuffers(const ::benchmark::State& state) {}
  virtual void deallocateTempBuffers(const ::benchmark::State& state) {}

  void allocateBuffers(const ::benchmark::State& state) override {
    allocateData(state);
    allocateTempBuffers(state);
  }

  void deallocateBuffers(const ::benchmark::State& state) override {
    deallocateTempBuffers(state);
    deallocateData(state);
  }

  void BenchmarkCase(::benchmark::State& state) {
    runBenchmark(state);
    generateMetrics(state);
  }

  std::unique_ptr<raft::handle_t> handle;

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
  BlobsFixture(const std::string& name, const DatasetParams p,
               const BlobsParams b)
    : Fixture(name), params(p), bParams(b) {}
  BlobsFixture() = delete;

 protected:
  void allocateData(const ::benchmark::State& state) override {
    data.allocate(*handle, params);
    data.blobs(*handle, params, bParams);
  }

  void deallocateData(const ::benchmark::State& state) override {
    data.deallocate(*handle, params);
  }

  DatasetParams params;
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
  RegressionFixture(const std::string& name, const DatasetParams p,
                    const RegressionParams r)
    : Fixture(name), params(p), rParams(r) {}
  RegressionFixture() = delete;

 protected:
  void allocateData(const ::benchmark::State& state) override {
    data.allocate(*handle, params);
    data.regression(*handle, params, rParams);
  }

  void deallocateData(const ::benchmark::State& state) override {
    data.deallocate(*handle, params);
  }

  DatasetParams params;
  /** parameters passed to `make_regression` */
  RegressionParams rParams;
  Dataset<D, D> data;
};  // end class RegressionFixture

/**
 * Fixture to be used for benchmarking time series algorithms when
 * the input suffices to be generated with a normal distribution.
 */
template <typename D>
class TsFixtureRandom : public Fixture {
 public:
  TsFixtureRandom(const std::string& name, const TimeSeriesParams p)
    : Fixture(name), params(p) {}
  TsFixtureRandom() = delete;

 protected:
  void allocateData(const ::benchmark::State& state) override {
    data.allocate(*handle, params);
    data.random(*handle, params);
  }

  void deallocateData(const ::benchmark::State& state) override {
    data.deallocate(*handle, params);
  }

  TimeSeriesParams params;
  TimeSeriesDataset<D> data;
};  // end class TsFixtureRandom

}  // end namespace Bench
}  // end namespace ML
