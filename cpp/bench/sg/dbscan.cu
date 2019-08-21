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
#include <dbscan/dbscan.hpp>
#include "dataset.h"
#include "harness.h"

namespace ML {
namespace Bench {
namespace dbscan {

template <typename D>
struct Params : public DatasetParams {
  // dataset generation related
  D clusterStd;
  bool shuffle;
  D centerBoxMin, centerBoxMax;
  uint64_t seed;
  // algo related
  int minPts;
  D eps;
  size_t maxBytesPerBatch;
};

template <typename D>
struct Run : public Benchmark<Params<D>> {
  void setup() {
    CUDA_CHECK(cudaStreamCreate(&stream));
    handle.reset(new cumlHandle(p.nstreams));
    handle->setStream(stream);
    auto allocator = handle->getDeviceAllocator();
    labels = (int*)allocator->allocate(p.nrows * sizeof(int), stream);
    dataset.blobs(*handle, p.nrows, p.ncols, p.nclasses, p.clusterStd,
                  p.shuffle, p.centerBoxMin, p.centerBoxMax, p.seed);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void teardown() {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto allocator = handle->getDeviceAllocator();
    allocator->deallocate(labels, p.nrows * sizeof(int), stream);
    dataset.deallocate(*handle);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  ///@todo: implement
  void metrics(RunInfo& ri) {
  }

  void run(RunInfo& ri) {
    ASSERT(p.rowMajor, "Dbscan only supports row-major inputs");
    dbscanFit(*handle, dataset.X, p.nrows, p.ncols, p.eps, p.minPts, labels,
              p.maxBytesPerBatch);
    CUDA_CHECK(cudaStreamSynchronize(handle->getStream()));
  }

 private:
  Params<D> p;
  std::shared_ptr<cumlHandle> handle;
  cudaStream_t stream;
  int *labels;
  Dataset<D, int> dataset;
};

}  // end namespace dbscan
}  // end namespace Bench
}  // end namespace ML
