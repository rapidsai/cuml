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
#include <randomforest/randomforest.hpp>
#include "dataset.h"
#include "harness.h"

namespace ML {
namespace Bench {
namespace rf {

template <typename D>
struct Params : public BlobsParams<D> {
  // algo related
  RF_params p;

  std::string str() const {
    std::ostringstream oss;
    oss << ";n_trees=" << p.n_trees << ";bootstrap=" << p.bootstrap
        << ";rows_sample=" << p.rows_sample << ";n_streams=" << p.n_streams
        << ";max_depth=" << p.tree_params.max_depth
        << ";max_leaves=" << p.tree_params.max_leaves
        << ";max_features=" << p.tree_params.max_features
        << ";n_bins=" << p.tree_params.n_bins
        << ";split_algo=" << p.tree_params.split_algo
        << ";min_rows_per_node=" << p.tree_params.min_rows_per_node
        << ";bootstrap_features=" << p.tree_params.bootstrap_features
        << ";quantile_per_tree=" << p.tree_params.quantile_per_tree
        << ";split_criterion=" << p.tree_params.split_criterion;
    return BlobsParams<D>::str() + oss.str();
  }
};

template <typename D>
struct Run : public Benchmark<Params<D>> {
  void setup() {
    const auto& p = this->getParams();
    CUDA_CHECK(cudaStreamCreate(&stream));
    ///@todo: enable this after PR: https://github.com/rapidsai/cuml/pull/1015
    // handle.reset(new cumlHandle(p.p.n_streams));
    handle.reset(new cumlHandle);
    handle->setStream(stream);
    auto allocator = handle->getDeviceAllocator();
    labels = (int*)allocator->allocate(p.nrows * sizeof(int), stream);
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
    dataset.deallocate(*handle);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  ///@todo: implement
  void metrics(RunInfo& ri) {}

 protected:
  std::shared_ptr<cumlHandle> handle;
  cudaStream_t stream;
  int* labels;
  Dataset<D, int> dataset;
};

struct RunF : public Run<float> {
  void run() {
    const auto& p = this->getParams();
    const auto& h = *handle;
    auto* mPtr = &model;
    mPtr->trees = nullptr;
    ASSERT(!p.rowMajor, "RF only supports col-major inputs");
    fit(h, mPtr, dataset.X, p.nrows, p.ncols, labels, p.nclasses, p.p);
    CUDA_CHECK(cudaStreamSynchronize(handle->getStream()));
  }

 private:
  ML::RandomForestClassifierF model;
};

struct RunD : public Run<double> {
  void run() {
    const auto& p = this->getParams();
    const auto& h = *handle;
    auto* mPtr = &model;
    mPtr->trees = nullptr;
    ASSERT(!p.rowMajor, "RF only supports col-major inputs");
    fit(h, mPtr, dataset.X, p.nrows, p.ncols, labels, p.nclasses, p.p);
    CUDA_CHECK(cudaStreamSynchronize(handle->getStream()));
  }

 private:
  ML::RandomForestClassifierD model;
};

template <typename D>
std::vector<Params<D>> getInputs() {
  std::vector<Params<D>> out;
  Params<D> p;
  p.rowMajor = false;
  p.cluster_std = (D)1.0;
  p.shuffle = false;
  p.center_box_min = (D)-10.0;
  p.center_box_max = (D)10.0;
  p.seed = 12345ULL;
  p.p.bootstrap = true;
  p.p.rows_sample = 1.f;
  p.p.tree_params.max_leaves = 1 << 20;
  p.p.tree_params.max_features = 1.f;
  p.p.tree_params.min_rows_per_node = 3;
  p.p.tree_params.n_bins = 32;
  p.p.tree_params.bootstrap_features = true;
  p.p.tree_params.quantile_per_tree = true;
  p.p.tree_params.split_algo = 0;
  p.p.tree_params.split_criterion = (ML::CRITERION)0;
  std::vector<std::pair<int, int>> rowcols = {
    {160000, 64},
    {640000, 64},
    {1280000, 64},
  };
  for (auto& rc : rowcols) {
    p.nrows = rc.first;
    p.ncols = rc.second;
    for (auto nclass : std::vector<int>({2, 8})) {
      p.nclasses = nclass;
      for (auto trees : std::vector<int>({500, 1000})) {
        p.p.n_trees = trees;
        for (auto max_depth : std::vector<int>({8, 10})) {
          p.p.tree_params.max_depth = max_depth;
          for (auto streams : std::vector<int>({4, 8})) {
            p.p.n_streams = streams;
            out.push_back(p);
          }
        }
      }
    }
  }
  return out;
}

REGISTER_BENCH(RunF, Params<float>, rfClassifierF, getInputs<float>());
REGISTER_BENCH(RunD, Params<double>, rfClassifierD, getInputs<double>());

}  // end namespace rf
}  // end namespace Bench
}  // end namespace ML
