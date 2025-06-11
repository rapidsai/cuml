/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include "benchmark.cuh"

#include <cuml/cluster/dbscan.hpp>
#include <cuml/common/distance_type.hpp>

#include <utility>

namespace ML {
namespace Bench {
namespace dbscan {

struct AlgoParams {
  int min_pts;
  double eps;
  size_t max_bytes_per_batch;
  bool calc_core_sample_indices;
};

struct Params {
  DatasetParams data;
  BlobsParams blobs;
  AlgoParams dbscan;
};

template <typename D>
class Dbscan : public BlobsFixture<D, int> {
 public:
  Dbscan(const std::string& name, const Params& p)
    : BlobsFixture<D, int>(name, p.data, p.blobs), dParams(p.dbscan), core_sample_indices(nullptr)
  {
  }

 protected:
  void runBenchmark(::benchmark::State& state) override
  {
    using MLCommon::Bench::CudaEventTimer;
    if (!this->params.rowMajor) { state.SkipWithError("Dbscan only supports row-major inputs"); }
    this->loopOnState(state, [this, &state]() {
      ML::Dbscan::fit(*this->handle,
                      this->data.X.data(),
                      this->params.nrows,
                      this->params.ncols,
                      D(dParams.eps),
                      dParams.min_pts,
                      ML::distance::DistanceType::L2SqrtUnexpanded,
                      this->data.y.data(),
                      this->core_sample_indices,
                      nullptr,
                      dParams.max_bytes_per_batch);
      state.SetItemsProcessed(this->params.nrows * this->params.ncols);
    });
  }

  void allocateTempBuffers(const ::benchmark::State& state) override
  {
    if (this->dParams.calc_core_sample_indices) {
      this->alloc(this->core_sample_indices, this->params.nrows);
    }
  }

  void deallocateTempBuffers(const ::benchmark::State& state) override
  {
    this->dealloc(this->core_sample_indices, this->params.nrows);
  }

 private:
  AlgoParams dParams;
  int* core_sample_indices;
};

std::vector<Params> getInputs(bool calc_core_sample_indices)
{
  std::vector<Params> out;
  Params p;
  p.data.rowMajor                          = true;
  p.blobs.cluster_std                      = 1.0;
  p.blobs.shuffle                          = false;
  p.blobs.center_box_min                   = -10.0;
  p.blobs.center_box_max                   = 10.0;
  p.blobs.seed                             = 12345ULL;
  p.dbscan.max_bytes_per_batch             = 0;
  p.dbscan.calc_core_sample_indices        = calc_core_sample_indices;
  std::vector<std::pair<int, int>> rowcols = {
    {10000, 81},
    {20000, 128},
    {40000, 128},
    {50000, 128},
    {100000, 128},
  };
  for (auto& rc : rowcols) {
    p.data.nrows = rc.first;
    p.data.ncols = rc.second;
    for (auto nclass : std::vector<int>({2, 4, 8})) {
      p.data.nclasses = nclass;
      for (auto ep : std::vector<double>({0.1, 1.0})) {
        p.dbscan.eps = ep;
        for (auto mp : std::vector<int>({3, 10})) {
          p.dbscan.min_pts = mp;
          out.push_back(p);
        }
      }
    }
  }
  return out;
}

// Calculate the benchmark with and without calculating the core pts
ML_BENCH_REGISTER(Params, Dbscan<float>, "blobs", getInputs(false));
ML_BENCH_REGISTER(Params, Dbscan<double>, "blobs", getInputs(false));

ML_BENCH_REGISTER(Params, Dbscan<float>, "blobs_core_ind", getInputs(true));
ML_BENCH_REGISTER(Params, Dbscan<double>, "blobs_core_ind", getInputs(true));

}  // end namespace dbscan
}  // end namespace Bench
}  // end namespace ML
