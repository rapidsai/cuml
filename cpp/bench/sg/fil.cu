/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cuml/fil/fil.h>
#include <cuml/tree/algo_helper.h>
#include <decisiontree/decisiontree_impl.h>
#include <treelite/c_api.h>
#include <treelite/tree.h>
#include <cuml/common/logger.hpp>
#include <cuml/cuml.hpp>
#include <cuml/ensemble/randomforest.hpp>
#include <utility>
#include "benchmark.cuh"

namespace ML {
namespace Bench {
namespace fil {

struct Params {
  DatasetParams data;
  RegressionParams blobs;
  ModelHandle model;
  bool predict_proba;
  bool fit_model;
  RF_params rf;
};

class FIL : public RegressionFixture<float> {
  typedef RegressionFixture<float> Base;

 public:
  FIL(const std::string& name, const Params& p)
    : RegressionFixture<float>(name, p.data, p.blobs),
      model(p.model),
      predict_proba(p.predict_proba),
      fit_model(p.fit_model),
      rfParams(p.rf) {}

 protected:
  void runBenchmark(::benchmark::State& state) override {
    if (!params.rowMajor) {
      state.SkipWithError("FIL only supports row-major inputs");
    }
    if (predict_proba) {
      // Dataset<D, L> allocates y assuming one output value per input row
      state.SkipWithError("currently only supports scalar prediction");
    }
    // create model
    ML::RandomForestRegressorF rf_model;
    auto* mPtr = &rf_model;
    mPtr->trees = nullptr;
    size_t train_nrows = std::min(params.nrows, 1000);
    fit(*handle, mPtr, data.X, train_nrows, params.ncols, data.y, rfParams);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    ML::build_treelite_forest(&model, &rf_model, params.ncols,
                              REGRESSION_MODEL);
    ML::fil::treelite_params_t tl_params = {
      .algo = ML::fil::algo_t::ALGO_AUTO,
      .output_class = true,                // cuML RF forest
      .threshold = 1.f / params.nclasses,  //Fixture::DatasetParams
      .storage_type = ML::fil::storage_type_t::SPARSE};
    ML::fil::from_treelite(*handle, &forest, model, &tl_params);

    // only time prediction
    this->loopOnState(state, [this]() {
      ML::fil::predict(*this->handle, this->forest, this->data.y, this->data.X,
                       this->params.nrows, this->predict_proba);
    });
  }

  void allocateBuffers(const ::benchmark::State& state) override {
    Base::allocateBuffers(state);
  }

  void deallocateBuffers(const ::benchmark::State& state) override {
    ML::fil::free(*handle, forest);
    Base::deallocateBuffers(state);
  }

 private:
  ML::fil::forest_t forest;
  ModelHandle model;
  bool predict_proba;
  bool fit_model;
  RF_params rfParams;
};

struct FilBenchParams {
  size_t nrows;
  size_t ncols;
  size_t nclasses;
  bool predict_proba;
};

std::vector<Params> getInputs() {
  std::vector<Params> out;
  Params p;
  size_t ncols = 20;
  p.data.rowMajor = true;
  // see src_prims/random/make_regression.h
  p.blobs = {.n_informative = (signed)ncols / 3,
             .effective_rank = 2 * (signed)ncols / 3,
             .bias = 0.f,
             .tail_strength = 0.1,
             .noise = 0.01,
             .shuffle = false,
             .seed = 12345ULL};
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
  p.rf.tree_params.max_features = 1.f;
  p.rf.tree_params.max_depth = 8;
  std::vector<FilBenchParams> var_params = {{10000ul, ncols, 2ul, false}};
  for (auto& i : var_params) {
    p.data.nrows = i.nrows;
    p.data.ncols = i.ncols;
    p.data.nclasses = i.nclasses;
    p.predict_proba = i.predict_proba;
    out.push_back(p);
  }
  return out;
}

ML_BENCH_REGISTER(Params, FIL, "", getInputs());

}  // end namespace fil
}  // end namespace Bench
}  // end namespace ML
