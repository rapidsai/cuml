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

#include <cuml/cluster/kmeans.hpp>
#include <cuml/cluster/kmeans_params.hpp>
#include <cuml/common/distance_type.hpp>
#include <cuml/common/logger.hpp>

#include <raft/random/rng_state.hpp>

#include <rapids_logger/logger.hpp>

#include <utility>

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
    : BlobsFixture<D>(name, p.data, p.blobs), kParams(p.kmeans)
  {
  }

 protected:
  void runBenchmark(::benchmark::State& state) override
  {
    using MLCommon::Bench::CudaEventTimer;
    if (!this->params.rowMajor) { state.SkipWithError("KMeans only supports row-major inputs"); }
    this->loopOnState(state, [this]() {
      ML::kmeans::fit(*this->handle,
                      kParams,
                      this->data.X.data(),
                      this->params.nrows,
                      this->params.ncols,
                      nullptr,
                      centroids,
                      inertia,
                      nIter);
      ML::kmeans::predict(*this->handle,
                          kParams,
                          centroids,
                          this->data.X.data(),
                          this->params.nrows,
                          this->params.ncols,
                          nullptr,
                          true,
                          this->data.y.data(),
                          inertia);
    });
  }

  void allocateTempBuffers(const ::benchmark::State& state) override
  {
    this->alloc(centroids, this->params.nclasses * this->params.ncols);
  }

  void deallocateTempBuffers(const ::benchmark::State& state) override
  {
    this->dealloc(centroids, this->params.nclasses * this->params.ncols);
  }

 private:
  ML::kmeans::KMeansParams kParams;
  D* centroids;
  D inertia;
  int nIter;
};

std::vector<Params> getInputs()
{
  std::vector<Params> out;
  Params p;
  p.data.rowMajor                          = true;
  p.blobs.cluster_std                      = 1.0;
  p.blobs.shuffle                          = false;
  p.blobs.center_box_min                   = -10.0;
  p.blobs.center_box_max                   = 10.0;
  p.blobs.seed                             = 12345ULL;
  p.kmeans.init                            = ML::kmeans::KMeansParams::InitMethod(0);
  p.kmeans.max_iter                        = 300;
  p.kmeans.tol                             = 1e-4;
  p.kmeans.verbosity                       = rapids_logger::level_enum::info;
  p.kmeans.metric                          = ML::distance::DistanceType::L2Expanded;
  p.kmeans.rng_state                       = raft::random::RngState(p.blobs.seed);
  p.kmeans.inertia_check                   = true;
  std::vector<std::pair<int, int>> rowcols = {
    {160000, 64},
    {320000, 64},
    {640000, 64},
    {80000, 500},
    {160000, 2000},
  };
  for (auto& rc : rowcols) {
    p.data.nrows = rc.first;
    p.data.ncols = rc.second;
    for (auto nclass : std::vector<int>({8, 16, 32})) {
      p.data.nclasses     = nclass;
      p.kmeans.n_clusters = p.data.nclasses;
      for (auto bs_shift : std::vector<int>({16, 18})) {
        p.kmeans.batch_samples = 1 << bs_shift;
        out.push_back(p);
      }
    }
  }
  return out;
}

ML_BENCH_REGISTER(Params, KMeans<float>, "blobs", getInputs());
ML_BENCH_REGISTER(Params, KMeans<double>, "blobs", getInputs());

}  // end namespace kmeans
}  // end namespace Bench
}  // end namespace ML
