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

#include <cuml/cluster/kmeans.hpp>
#include <cuml/cuml.hpp>
#include <utility>
#include "benchmark.cuh"

namespace ML {
namespace Bench {
namespace kmeans {

struct Params {
  DatasetParams data;
  BlobsParams blobs;
  ML::kmeans::KMeansParams kmeans;
};

template <typename D>
class KMeans : public BlobsFixture<D> {
 public:
  KMeans(const std::string& name, const Params& p)
    : BlobsFixture<D>(p.data, p.blobs), kParams(p.kmeans) {
    this->SetName(name.c_str());
  }

 protected:
  void runBenchmark(::benchmark::State& state) override {
    if (!this->params.rowMajor) {
      state.SkipWithError("KMeans only supports row-major inputs");
    }
    auto& handle = *this->handle;
    auto stream = handle.getStream();
    for (auto _ : state) {
      CudaEventTimer timer(handle, state, true, stream);
      ML::kmeans::fit_predict(handle, kParams, this->data.X, this->params.nrows,
                              this->params.ncols, centroids, this->data.y,
                              inertia, nIter);
    }
  }

  void allocateBuffers(const ::benchmark::State& state) override {
    auto allocator = this->handle->getDeviceAllocator();
    auto stream = this->handle->getStream();
    centroids = (D*)allocator->allocate(
      this->params.nclasses * this->params.ncols * sizeof(D), stream);
  }

  void deallocateBuffers(const ::benchmark::State& state) override {
    auto allocator = this->handle->getDeviceAllocator();
    auto stream = this->handle->getStream();
    allocator->deallocate(
      centroids, this->params.nclasses * this->params.ncols * sizeof(D),
      stream);
  }

 private:
  ML::kmeans::KMeansParams kParams;
  D* centroids;
  D inertia;
  int nIter;
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
  p.kmeans.init = ML::kmeans::KMeansParams::InitMethod(0);
  p.kmeans.max_iter = 300;
  p.kmeans.tol = 1e-4;
  p.kmeans.verbose = false;
  p.kmeans.seed = int(p.blobs.seed);
  p.kmeans.metric = 0;  // L2
  p.kmeans.inertia_check = true;
  std::vector<std::pair<int, int>> rowcols = {
    {160000, 64}, {320000, 64}, {640000, 64}, {80000, 500}, {160000, 2000},
  };
  for (auto& rc : rowcols) {
    p.data.nrows = rc.first;
    p.data.ncols = rc.second;
    for (auto nclass : std::vector<int>({8, 16, 32})) {
      p.data.nclasses = nclass;
      p.kmeans.n_clusters = p.data.nclasses;
      for (auto bs_shift : std::vector<int>({16, 18})) {
        p.kmeans.batch_samples = 1 << bs_shift;
        out.push_back(p);
      }
    }
  }
  return out;
}

CUML_BENCH_REGISTER(Params, KMeans<float>, "blobs", getInputs());
CUML_BENCH_REGISTER(Params, KMeans<double>, "blobs", getInputs());

}  // end namespace kmeans
}  // end namespace Bench
}  // end namespace ML
