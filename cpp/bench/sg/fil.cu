/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <treelite/c_api.h>
#include <treelite/tree.h>
#include <cuml/common/logger.hpp>
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
  ML::fil::storage_type_t storage;
  ML::fil::algo_t algo;
  RF_params rf;
  int predict_repetitions;
};

class FIL : public RegressionFixture<float> {
  typedef RegressionFixture<float> Base;

 public:
  FIL(const std::string& name, const Params& p)
  /*
        fitting to linear combinations in "y" normally yields trees that check
        values of all significant columns, as well as their linear
        combinations in "X". During inference, the exact threshold
        values do not affect speed. The distribution of column popularity does
        not affect speed barring lots of uninformative columns in succession.
        Hence, this method represents real datasets well enough for both
        classification and regression.
      */
  : RegressionFixture<float>(name, p.data, p.blobs), model(p.model), p_rest(p)
  {
  }

  static void regression_to_classification(float* y, int nrows, int nclasses, cudaStream_t stream)
  {
    raft::linalg::unaryOp(
      y,
      y,
      nrows,
      [=] __device__(float a) { return float(lroundf(fabsf(a) * 1000. * nclasses) % nclasses); },
      stream);
  }

 protected:
  void runBenchmark(::benchmark::State& state) override
  {
    if (!params.rowMajor) { state.SkipWithError("FIL only supports row-major inputs"); }
    if (params.nclasses > 1) {
      // convert regression ranges into [0..nclasses-1]
      regression_to_classification(data.y, params.nrows, params.nclasses, stream);
    }
    // create model
    ML::RandomForestRegressorF rf_model;
    auto* mPtr         = &rf_model;
    mPtr->trees        = nullptr;
    size_t train_nrows = std::min(params.nrows, 1000);
    fit(*handle, mPtr, data.X, train_nrows, params.ncols, data.y, p_rest.rf);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    ML::build_treelite_forest(&model, &rf_model, params.ncols, params.nclasses > 1 ? 2 : 1);
    for (int threads_per_tree : {64, 128, 256}) {
      for (int blocks_per_sm : {8}) {
        for (int i : {1, 2}) {
          char *ostr = nullptr;
          bool once = threads_per_tree == 64 && i == 1;
          ML::fil::treelite_params_t tl_params = {
            .algo = p_rest.algo,
            .output_class = params.nclasses > 1,  // cuML RF forest
            .threshold = 1.f / params.nclasses,   //Fixture::DatasetParams
            .storage_type = p_rest.storage,
            .blocks_per_sm = blocks_per_sm,
            .threads_per_tree = threads_per_tree,
            .n_items = 1,
            .pforest_shape_str = once ? &ostr : nullptr};
          ML::fil::from_treelite(*handle, &forest, model, &tl_params);
          if(once) {
            std::cout << ostr << std::endl;
            ::free(ostr);
          }

          cudaEvent_t start;
          cudaEvent_t stop;
          CUDA_CHECK(cudaEventCreate(&start));
          CUDA_CHECK(cudaEventCreate(&stop));
          for (int i = 1; i < 10 * p_rest.predict_repetitions; ++i) {
            ML::fil::predict(*this->handle, this->forest, this->data.y,
                             this->data.X, this->params.nrows, false);
          }
          CUDA_CHECK(cudaEventRecord(start, 0));
          for (int i = 0; i < p_rest.predict_repetitions; i++) {
            ML::fil::predict(*this->handle, this->forest, this->data.y,
                             this->data.X, this->params.nrows, false);
          }
          CUDA_CHECK_NO_THROW(cudaEventRecord(stop, 0));
          CUDA_CHECK_NO_THROW(cudaEventSynchronize(stop));
          float milliseconds = 0.0f;
          CUDA_CHECK_NO_THROW(cudaEventElapsedTime(&milliseconds, start, stop));
          printf(
            "max_depth %d n_trees %d blocks_per_sm %d threads_per_tree %d %7s "
            "%.2f ms\n",
            p_rest.rf.tree_params.max_depth, p_rest.rf.n_trees, blocks_per_sm,
            threads_per_tree,
            tl_params.storage_type == ML::fil::SPARSE ? "SPARSE16" : "sparse8",
            milliseconds / p_rest.predict_repetitions);
          CUDA_CHECK_NO_THROW(cudaEventDestroy(start));
          CUDA_CHECK_NO_THROW(cudaEventDestroy(stop));
        }
      }
    }

    // only time prediction
    this->loopOnState(state, [this]() {
      // Dataset<D, L> allocates y assuming one output value per input row,
      // so not supporting predict_proba yet
      for (int i = 0; i < p_rest.predict_repetitions; i++) {
        ML::fil::predict(
          *this->handle, this->forest, this->data.y, this->data.X, this->params.nrows, false);
      }
    });
  }

  void allocateBuffers(const ::benchmark::State& state) override { Base::allocateBuffers(state); }

  void deallocateBuffers(const ::benchmark::State& state) override
  {
    ML::fil::free(*handle, forest);
    Base::deallocateBuffers(state);
  }

 private:
  ML::fil::forest_t forest;
  ModelHandle model;
  Params p_rest;
};

struct FilBenchParams {
  int nrows;
  int ncols;
  int nclasses;
  int max_depth;
  int ntrees;
  ML::fil::storage_type_t storage;
  ML::fil::algo_t algo;
};

std::vector<Params> getInputs()
{
  std::vector<Params> out;
  Params p;
  p.data.rowMajor = true;
  p.blobs         = {.n_informative  = -1,  // Just a placeholder value, anyway changed below
             .effective_rank = -1,  // Just a placeholder value, anyway changed below
             .bias           = 0.f,
             .tail_strength  = 0.1,
             .noise          = 0.01,
             .shuffle        = false,
             .seed           = 12345ULL};

  p.rf = set_rf_params(4,                 /*max_depth */
                       1024,          /* max_leaves */
                       1.f,                /* max_features */
                       32,                 /* n_bins */
                       3,                  /* min_samples_leaf */
                       3,                  /* min_samples_split */
                       0.0f,               /* min_impurity_decrease */
                       true,               /* bootstrap */
                       1,                  /* n_trees */
                       1.f,                /* max_samples */
                       1234ULL,            /* seed */
                       ML::CRITERION::MSE, /* split_criterion */
                       8,                  /* n_streams */
                       128                 /* max_batch_size */
  );

  using ML::fil::algo_t;
  using ML::fil::storage_type_t;
  std::vector<FilBenchParams> var_params = {
    {(int)1e6, 20, 1, 4, 1000, storage_type_t::DENSE, algo_t::TREE_REORG}};
  for (auto& i : var_params) {
    p.data.nrows               = i.nrows;
    p.data.ncols               = i.ncols;
    p.blobs.n_informative      = i.ncols / 3;
    p.blobs.effective_rank     = i.ncols / 3;
    p.data.nclasses            = i.nclasses;
    p.rf.tree_params.max_depth = i.max_depth;
    p.rf.n_trees               = i.ntrees;
    p.storage                  = i.storage;
    p.algo                     = i.algo;
    p.predict_repetitions      = 10;
    out.push_back(p);
  }
  return out;
}

ML_BENCH_REGISTER(Params, FIL, "", getInputs());

}  // end namespace fil
}  // end namespace Bench
}  // end namespace ML
