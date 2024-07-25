/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cuml/ensemble/randomforest.hpp>

#include <cmath>
#include <utility>

namespace ML {
namespace Bench {
namespace rf {

struct Params {
  DatasetParams data;
  BlobsParams blobs;
  RF_params rf;
};

template <typename D>
struct RFClassifierModel {};

template <>
struct RFClassifierModel<float> {
  ML::RandomForestClassifierF model;
};

template <>
struct RFClassifierModel<double> {
  ML::RandomForestClassifierD model;
};

template <typename D>
class RFClassifier : public BlobsFixture<D> {
 public:
  RFClassifier(const std::string& name, const Params& p)
    : BlobsFixture<D>(name, p.data, p.blobs), rfParams(p.rf)
  {
  }

 protected:
  void runBenchmark(::benchmark::State& state) override
  {
    using MLCommon::Bench::CudaEventTimer;
    if (this->params.rowMajor) {
      state.SkipWithError("RFClassifier only supports col-major inputs");
    }
    this->loopOnState(state, [this]() {
      auto* mPtr = &model.model;
      fit(*this->handle,
          mPtr,
          this->data.X.data(),
          this->params.nrows,
          this->params.ncols,
          this->data.y.data(),
          this->params.nclasses,
          rfParams);
      this->handle->sync_stream(this->stream);
    });
  }

 private:
  RFClassifierModel<D> model;
  RF_params rfParams;
};

template <typename D>
std::vector<Params> getInputs()
{
  struct Triplets {
    int nrows, ncols, nclasses;
  };
  std::vector<Params> out;
  Params p;
  p.data.rowMajor = false;
  p.blobs         = {10.0,         // cluster_std
                     false,        // shuffle
                     -10.0,        // center_box_min
                     10.0,         // center_box_max
                     2152953ULL};  // seed

  p.rf = set_rf_params(10,                  /*max_depth */
                       (1 << 20),           /* max_leaves */
                       0.3,                 /* max_features */
                       32,                  /* max_n_bins */
                       3,                   /* min_samples_leaf */
                       3,                   /* min_samples_split */
                       0.0f,                /* min_impurity_decrease */
                       true,                /* bootstrap */
                       500,                 /* n_trees */
                       1.f,                 /* max_samples */
                       1234ULL,             /* seed */
                       ML::CRITERION::GINI, /* split_criterion */
                       8,                   /* n_streams */
                       128                  /* max_batch_size */
  );

  std::vector<Triplets> rowcols = {
    {160000, 64, 2}, {640000, 64, 8}, {1184000, 968, 2},  // Mimicking Bosch dataset
  };
  for (auto& rc : rowcols) {
    // Let's run Bosch only for float type
    if (!std::is_same<D, float>::value && rc.ncols == 968) continue;
    p.data.nrows                  = rc.nrows;
    p.data.ncols                  = rc.ncols;
    p.data.nclasses               = rc.nclasses;
    p.rf.tree_params.max_features = 1.f / std::sqrt(float(rc.ncols));
    for (auto max_depth : std::vector<int>({7, 9})) {
      p.rf.tree_params.max_depth = max_depth;
      out.push_back(p);
    }
  }
  return out;
}

ML_BENCH_REGISTER(Params, RFClassifier<float>, "blobs", getInputs<float>());
ML_BENCH_REGISTER(Params, RFClassifier<double>, "blobs", getInputs<double>());

}  // end namespace rf
}  // end namespace Bench
}  // end namespace ML
