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

#include <cuml/cluster/linkage.hpp>
#include <cuml/cluster/single_linkage_output.hpp>
#include <cuml/common/distance_type.hpp>
#include <cuml/common/logger.hpp>

#include <utility>

namespace ML {
namespace Bench {
namespace linkage {

struct Params {
  DatasetParams data;
  BlobsParams blobs;
};

template <typename D>
class Linkage : public BlobsFixture<D> {
 public:
  Linkage(const std::string& name, const Params& p) : BlobsFixture<D>(name, p.data, p.blobs) {}

 protected:
  void runBenchmark(::benchmark::State& state) override
  {
    using MLCommon::Bench::CudaEventTimer;
    if (!this->params.rowMajor) {
      state.SkipWithError("Single-Linkage only supports row-major inputs");
    }

    this->loopOnState(state, [this]() {
      out_arrs.labels   = labels;
      out_arrs.children = out_children;

      ML::single_linkage_neighbors(*this->handle,
                                   this->data.X.data(),
                                   this->params.nrows,
                                   this->params.ncols,
                                   &out_arrs,
                                   ML::distance::DistanceType::L2Unexpanded,
                                   15,
                                   50);
    });
  }

  void allocateTempBuffers(const ::benchmark::State& state) override
  {
    this->alloc(labels, this->params.nrows);
    this->alloc(out_children, (this->params.nrows - 1) * 2);
  }

  void deallocateTempBuffers(const ::benchmark::State& state) override
  {
    this->dealloc(labels, this->params.nrows);
    this->dealloc(out_children, (this->params.nrows - 1) * 2);
  }

 private:
  int* labels;
  int* out_children;
  ML::single_linkage_output<int> out_arrs;
};

std::vector<Params> getInputs()
{
  std::vector<Params> out;
  Params p;
  p.data.rowMajor                          = true;
  p.blobs.cluster_std                      = 5.0;
  p.blobs.shuffle                          = false;
  p.blobs.center_box_min                   = -10.0;
  p.blobs.center_box_max                   = 10.0;
  p.blobs.seed                             = 12345ULL;
  std::vector<std::pair<int, int>> rowcols = {
    {35000, 128},
    {16384, 128},
    {12288, 128},
    {8192, 128},
    {4096, 128},
  };
  for (auto& rc : rowcols) {
    p.data.nrows = rc.first;
    p.data.ncols = rc.second;
    for (auto nclass : std::vector<int>({1})) {
      p.data.nclasses = nclass;
      out.push_back(p);
    }
  }
  return out;
}

ML_BENCH_REGISTER(Params, Linkage<float>, "blobs", getInputs());

}  // namespace linkage
}  // end namespace Bench
}  // end namespace ML
