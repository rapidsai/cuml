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
    : RegressionFixture<D>(name, p.data, p.regression), rfParams(p.rf)
  {
  }

 protected:
  void runBenchmark(::benchmark::State& state) override
  {
    using MLCommon::Bench::CudaEventTimer;
    if (this->params.rowMajor) {
      state.SkipWithError("RFRegressor only supports col-major inputs");
    }
    this->loopOnState(state, [this]() {
      auto* mPtr = &model.model;
      fit(*this->handle,
          mPtr,
          this->data.X,
          this->params.nrows,
          this->params.ncols,
          this->data.y,
          rfParams);
      handle->sync_stream(this->stream);
    });
  }

 private:
  RFRegressorModel<D> model;
  RF_params rfParams;
};

template <typename D>
std::vector<RegParams> getInputs()
{
  struct DimInfo {
    int nrows, ncols, n_informative;
  };
  struct std::vector<RegParams> out;
  RegParams p;
  p.data.rowMajor = false;
  p.regression    = {.shuffle        = true,  // Better to shuffle when n_informative < ncols
                     .effective_rank = -1,    // dataset generation will be faster
                     .bias           = 4.5,
                     .tail_strength  = 0.5,  // unused when effective_rank = -1
                     .noise          = 1.0,
                     .seed           = 12345ULL};

  p.rf                          = set_rf_params(10,                 /*max_depth */
                       (1 << 20),          /* max_leaves */
                       0.3,                /* max_features */
                       32,                 /* max_n_bins */
                       3,                  /* min_samples_leaf */
                       3,                  /* min_samples_split */
                       0.0f,               /* min_impurity_decrease */
                       true,               /* bootstrap */
                       500,                /* n_trees */
                       1.f,                /* max_samples */
                       1234ULL,            /* seed */
                       ML::CRITERION::MSE, /* split_criterion */
                       8,                  /* n_streams */
                       128                 /* max_batch_size */
  );
  std::vector<DimInfo> dim_info = {{500000, 500, 400}};
  for (auto& di : dim_info) {
    // Let's run Bosch only for float type
    if (!std::is_same<D, float>::value && di.ncols == 968) continue;
    p.data.nrows                  = di.nrows;
    p.data.ncols                  = di.ncols;
    p.regression.n_informative    = di.n_informative;
    p.rf.tree_params.max_features = 1.f;
    for (auto max_depth : std::vector<int>({7, 11, 15})) {
      p.rf.tree_params.max_depth = max_depth;
      out.push_back(p);
    }
  }
  return out;
}

ML_BENCH_REGISTER(RegParams, RFRegressor<float>, "regression", getInputs<float>());
ML_BENCH_REGISTER(RegParams, RFRegressor<double>, "regression", getInputs<double>());

}  // namespace rf
}  // namespace Bench
}  // namespace ML
