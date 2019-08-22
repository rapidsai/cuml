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

#include <cuML.hpp>
#include <kmeans/kmeans.hpp>
#include "dataset.h"
#include "harness.h"

namespace ML {
namespace Bench {
namespace kmeans {

template <typename D>
struct Params : public DatasetParams {
  // dataset generation related
  D cluster_std;
  bool shuffle;
  D center_box_min, center_box_max;
  uint64_t seed;
  // algo related
  kmeans::KMeansParams p;

  Params() : DatasetParams() {}

  Params(int nr, int nc, int ncl, bool row, D std, bool shfl, D cmin, D cmax,
         uint64_t s, int ini, int mi, double tol, bool ver, int mt, int of,
         int bs, bool ic)
    : DatasetParams(nr, nc, ncl, row),
      cluster_std(std),
      shuffle(shfl),
      center_box_min(cmin),
      center_box_max(cmax),
      seed(s),
      p() {
    p.n_clusters = nclasses;
    p.init = (kmeans::KMeansParams::InitMethod)ini;
    p.max_iter = mi;
    p.tol = tol;
    p.verbose = ver;
    p.seed = seed;
    p.metric = mt;
    p.oversampling_factor = of;
    p.batch_size = bs;
    p.interia_check = ic;
  }

  std::string str() const {
    std::ostringstream oss;
    oss << ";cluster_std=" << cluster_std << ";shuffle=" << shuffle
        << ";center_box_min=" << center_box_min
        << ";center_box_max=" << center_box_max << ";seed=" << seed
        << ";init=" << p.init << ";max_iter=" << p.max_iter
        << ";tol=" << p.tol << ";verbose=" << verbose << ";metric=" << p.metric
        << ";oversampling_factor=" << p.oversampling_factor << ";batch_size="
        << p.batch_size << ";inertia-check=" << p.inertia_check;
    return DatasetParams::str() + oss.str();
  }
};

template <typename D>
struct Run : public Benchmark<Params<D>> {
  void setup() {
    const auto& p = this->getParams();
    CUDA_CHECK(cudaStreamCreate(&stream));
    handle.reset(new cumlHandle);
    handle->setStream(stream);
    auto allocator = handle->getDeviceAllocator();
    labels = (int*)allocator->allocate(p.nrows * sizeof(int), stream);
    centroids = (D*)allocator->allocate(p.nclasses * p.ncols * sizeof(D), stream);
    dataset.blobs(*handle, p.nrows, p.ncols, p.rowMajor, p.nclasses,
                  p.cluster_std, p.shuffle, p.center_box_min, p.center_box_max,
                  p.seed);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void teardown() {
    const auto& p = this->getParams();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto allocator = handle->getDeviceAllocator();
    allocator->deallocate(labels, p.nrows * sizeof(int), stream);
    allocator->deallocate(centroids, p.nclasses * p.ncols * sizeof(D), stream);
    dataset.deallocate(*handle);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  ///@todo: implement
  void metrics(RunInfo& ri) {}

  void run() {
    const auto& p = this->getParams();
    ASSERT(p.rowMajor, "Kmeans only supports row-major inputs");
    kmeans::fit_predict(*handle, p.p, dataset.X, p.nrows, p.ncols, centroids,
                        labels, inertia, nIter);
    CUDA_CHECK(cudaStreamSynchronize(handle->getStream()));
  }

 private:
  std::shared_ptr<cumlHandle> handle;
  cudaStream_t stream;
  int* labels;
  D* centroids;
  Dataset<D, int> dataset;
  int nIter;
  D inertia;
};

template <typename D>
std::vector<Params<D>> getInputs() {
  std::vector<Params<D>> out;
  Params<D> p;
  p.rowMajor = true;
  p.cluster_std = (D)1.0;
  p.shuffle = false;
  p.center_box_min = (D)-10.0;
  p.center_box_max = (D)10.0;
  p.seed = 12345ULL;
  p.max_bytes_per_batch = 0;
  std::vector<std::pair<int, int>> rowcols = {
    {10000, 81},
    {20000, 128},
    {40000, 128},
  };
  for (auto& rc : rowcols) {
    p.nrows = rc.first;
    p.ncols = rc.second;
    for (auto nclass : std::vector<int>({2, 4, 8})) {
      p.nclasses = nclass;
      for (auto ep : std::vector<D>({0.1, 1.0})) {
        p.eps = ep;
        for (auto mp : std::vector<int>({3, 10})) {
          p.min_pts = mp;
          out.push_back(p);
        }
      }
    }
  }
  return out;
}

REGISTER_BENCH(Run<float>, Params<float>, kmeansF, getInputs<float>());
REGISTER_BENCH(Run<double>, Params<double>, kmeansD, getInputs<double>());

}  // end namespace kmeans
}  // end namespace Bench
}  // end namespace ML
