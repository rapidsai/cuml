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

#include <cmath>
#include <cuml/cuml.hpp>
#include <cuml/ensemble/randomforest.hpp>
#include <utility>
#include "benchmark.cuh"

namespace ML {
namespace Bench {
namespace rf {

struct RegParams {
  DatasetParams data;
  RegressionParams regression;
  RF_params rf;
};

template <typename D>
struct RFRegressorModel {};

template <>
struct RFRegressorModel<float> {
  ML::RandomForestRegressorF model;
};

template <>
struct RFRegressorModel<double> {
  ML::RandomForestRegressorD model;
};

template <typename D>
class RFRegressor : public RegressionFixture<D> {
 public:
  RFRegressor(const std::string& name, const RegParams& p)
    : RegressionFixture<D>(p.data, p.regression), rfParams(p.rf) {
    this->SetName(name.c_str());
  }

 protected:
  void runBenchmark(::benchmark::State& state) override {
    if (this->params.rowMajor) {
      state.SkipWithError("RFRegressor only supports col-major inputs");
    }
    auto& handle = *this->handle;
    auto stream = handle.getStream();
    auto* mPtr = &model.model;
    for (auto _ : state) {
      CudaEventTimer timer(handle, state, true, stream);
      mPtr->trees = nullptr;
      fit(handle, mPtr, this->data.X, this->params.nrows, this->params.ncols,
          this->data.y, rfParams);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      delete[] mPtr->trees;
    }
  }

 private:
  RFRegressorModel<D> model;
  RF_params rfParams;
};

template <typename D>
std::vector<RegParams> getInputs() {
  struct DimInfo {
    int nrows, ncols, n_informative;
  };
  struct std::vector<RegParams> out;
  RegParams p;
  p.data.rowMajor = false;
  p.regression.shuffle = true;  // better to shuffle when n_informative < ncols
  p.regression.seed = 12345ULL;
  p.regression.effective_rank = -1;  // dataset generation will be faster
  p.regression.bias = 4.5;
  p.regression.tail_strength = 0.5;  // unused when effective_rank = -1
  p.regression.noise = 1.;
  p.rf.bootstrap = true;
  p.rf.rows_sample = 1.f;
  p.rf.tree_params.max_leaves = 1 << 20;
  p.rf.tree_params.min_rows_per_node = 3;
  p.rf.tree_params.n_bins = 32;
  p.rf.tree_params.bootstrap_features = true;
  p.rf.tree_params.quantile_per_tree = false;
  p.rf.tree_params.split_algo = 1;
  p.rf.tree_params.split_criterion = ML::CRITERION::MSE;
  p.rf.n_trees = 500;
  p.rf.n_streams = 8;
  std::vector<DimInfo> dim_info = {{500000, 500, 400}};
  for (auto& di : dim_info) {
    // Let's run Bosch only for float type
    if (!std::is_same<D, float>::value && di.ncols == 968) continue;
    p.data.nrows = di.nrows;
    p.data.ncols = di.ncols;
    p.regression.n_informative = di.n_informative;
    p.rf.tree_params.max_features = 1.f;
    for (auto max_depth : std::vector<int>({8, 12, 16})) {
      p.rf.tree_params.max_depth = max_depth;
      out.push_back(p);
    }
  }
  return out;
}

CUML_BENCH_REGISTER(RegParams, RFRegressor<float>, "regression",
                    getInputs<float>());
CUML_BENCH_REGISTER(RegParams, RFRegressor<double>, "regression",
                    getInputs<double>());

}  // namespace rf
}  // namespace Bench
}  // namespace ML
