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

#include <cuml/cluster/dbscan.hpp>
#include <cuml/cuml.hpp>
#include <utility>
#include "benchmark.cuh"

namespace ML {
namespace Bench {
namespace dbscan {

struct AlgoParams {
  int min_pts;
  double eps;
  size_t max_bytes_per_batch;
};

struct Params {
  DatasetParams data;
  BlobsParams blobs;
  AlgoParams dbscan;
};

template <typename D>
class Dbscan : public BlobsFixture<D, long> {
 public:
  Dbscan(const std::string& name, const Params& p)
    : BlobsFixture<D, long>(p.data, p.blobs), dParams(p.dbscan) {
    this->SetName(name.c_str());
  }

 protected:
  void runBenchmark(::benchmark::State& state) override {
    if (!this->params.rowMajor) {
      state.SkipWithError("Dbscan only supports row-major inputs");
    }
    auto& handle = *this->handle;
    auto stream = handle.getStream();
    for (auto _ : state) {
      CudaEventTimer timer(handle, state, true, stream);
      dbscanFit(handle, this->data.X, this->params.nrows, this->params.ncols,
                D(dParams.eps), dParams.min_pts, this->data.y,
                dParams.max_bytes_per_batch);
    }
  }

 private:
  AlgoParams dParams;
};

std::vector<Params> getInputs() {
  std::vector<Params> out;
  Params p;
  p.data.rowMajor = true;
  p.blobs.cluster_std = 1.0;
  p.blobs.shuffle = false;
  p.blobs.center_box_min = -10.0;
  p.blobs.center_box_max = 10.0;
  p.blobs.seed = 12345ULL;
  p.dbscan.max_bytes_per_batch = 0;
  std::vector<std::pair<int, int>> rowcols = {
    {10000, 81}, {20000, 128}, {40000, 128}, {50000, 128}, {100000, 128},
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

CUML_BENCH_REGISTER(Params, Dbscan<float>, "blobs", getInputs());
CUML_BENCH_REGISTER(Params, Dbscan<double>, "blobs", getInputs());

}  // end namespace dbscan
}  // end namespace Bench
}  // end namespace ML
