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
#include <utility>
#include "core.cuh"

namespace ML {
namespace Bench {
namespace kmeans {

typedef ML::kmeans::KMeansParams Params;

template <typename D>
class KMeans : public BlobsFixture<D> {
 public:
  KMeans(const std::string& name, const DatasetParams p, const BlobsParams b,
         const Params kp)
    : BlobsFixture<D>(p, b), kParams(kp) {
    this->SetName(name.c_str());
  }

 protected:
  void runBenchmark(::benchmark::State& state) override {
    if (!this->params.rowMajor) {
      state.SkipWithError("KMeans only supports row-major inputs");
    }
    for (auto _ : state) {
      ///@todo: cuda event timer
      ML::kmeans::fit_predict(*this->handle, kParams, this->data.X,
                              this->params.nrows, this->params.ncols, centroids,
                              labels, inertia, nIter);
      CUDA_CHECK(cudaStreamSynchronize(this->handle->getStream()));
    }
  }

  void allocateBuffers(const ::benchmark::State& state) override {
    auto allocator = this->handle->getDeviceAllocator();
    auto stream = this->handle->getStream();
    labels =
      (int*)allocator->allocate(this->params.nrows * sizeof(int), stream);
    centroids = (D*)allocator->allocate(
      this->params.nclasses * this->params.ncols * sizeof(D), stream);
  }

  void deallocateBuffers(const ::benchmark::State& state) override {
    auto allocator = this->handle->getDeviceAllocator();
    auto stream = this->handle->getStream();
    allocator->deallocate(
      centroids, this->params.nclasses * this->params.ncols * sizeof(D),
      stream);
    allocator->deallocate(labels, this->params.nrows * sizeof(int), stream);
  }

 private:
  Params kParams;
  int* labels;
  D* centroids;
  D inertia;
  int nIter;
};

typedef KMeans<float> KMeansF;
typedef KMeans<double> KMeansD;

const DatasetParams p = {160000, 64, 8, true};
const BlobsParams bp1 = {1.0, false, -10.0, 10.0, 12345ULL};
// const Params kp1 = {8,          ML::kmeans::KMeansParams::KMeansPlusPlus,
//                     300,        1e-4,
//                     0,          int(bp1.seed),
//                     0 /* L2 */, 2,
//                     1 << 16,    true};
Params kp1;

CUML_BENCH_REGISTER_F(KMeansF, bench1, p, bp1, kp1);
CUML_BENCH_REGISTER_F(KMeansD, bench1, p, bp1, kp1);

}  // end namespace kmeans
}  // end namespace Bench
}  // end namespace ML
