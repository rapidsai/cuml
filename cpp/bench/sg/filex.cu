/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include <cuml/experimental/fil/treelite_importer.hpp>
#include <cuml/experimental/kayak/device_type.hpp>

#include "benchmark.cuh"
#include <chrono>
#include <cuml/common/logger.hpp>
#include <cuml/ensemble/randomforest.hpp>
#include <cuml/tree/algo_helper.h>
#include <treelite/c_api.h>
#include <treelite/tree.h>
#include <utility>

namespace ML {
namespace Bench {
namespace filex {

struct Params {
  DatasetParams data;
  RegressionParams blobs;
  ModelHandle model;
  ML::fil::storage_type_t storage;
  ML::fil::algo_t algo;
  bool use_experimental;
  RF_params rf;
  int predict_repetitions;
};

class FILEX : public RegressionFixture<float> {
  typedef RegressionFixture<float> Base;

 public:
  FILEX(const std::string& name, const Params& p)
  : RegressionFixture<float>(name, p.data, p.blobs), model(p.model), p_rest(p)
  {
    Iterations(100);
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
    /* if (params.nclasses > 1) {
      // convert regression ranges into [0..nclasses-1]
      regression_to_classification(data.y.data(), params.nrows, params.nclasses, stream);
    } */
    // create model
    ML::RandomForestRegressorF rf_model;
    auto* mPtr         = &rf_model;
    size_t train_nrows = std::min(params.nrows, 1000);
    fit(*handle, mPtr, data.X.data(), train_nrows, params.ncols, data.y.data(), p_rest.rf);
    handle->sync_stream(stream);

    ML::build_treelite_forest(&model, &rf_model, params.ncols);

    auto filex_model = ML::experimental::fil::import_from_treelite_handle(
      model,
      128,
      false,
      kayak::device_type::gpu,
      0,
      stream
    );

    ML::fil::treelite_params_t tl_params = {
      .algo              = p_rest.algo,
      .output_class      = false,
      .threshold         = 1.f / params.nclasses,  // Fixture::DatasetParams
      .storage_type      = p_rest.storage,
      .blocks_per_sm     = 8,
      .threads_per_tree  = 1,
      .n_items           = 0,
      .pforest_shape_str = nullptr};
    ML::fil::forest_variant forest_variant;
    auto optimal_chunk_size = 1;
    auto min_time = std::numeric_limits<std::size_t>::max();

    // Iterate through chunk sizes and find optimum
    for (auto chunk_size = 1; chunk_size <= 32; chunk_size *= 2) {
      if (!p_rest.use_experimental) {
        tl_params.threads_per_tree = chunk_size;
        ML::fil::from_treelite(*handle, &forest_variant, model, &tl_params);
        forest = std::get<ML::fil::forest_t<float>>(forest_variant);
      }
      handle->sync_stream();
      handle->sync_stream_pool();
      auto start = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < p_rest.predict_repetitions; i++) {
        // Create FIL forest
        if (p_rest.use_experimental) {
          filex_model.predict(
            *handle,
            data.y.data(),
            data.X.data(),
            params.nrows,
            kayak::device_type::gpu,
            kayak::device_type::gpu,
            chunk_size
          );
        } else {
          ML::fil::predict(*handle,
                           forest,
                           data.y.data(),
                           data.X.data(),
                           params.nrows,
                           false);
        }
        handle->sync_stream();
        handle->sync_stream_pool();
      }
      auto end = std::chrono::high_resolution_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end - start
      ).count();
      if (elapsed < min_time) {
        min_time = elapsed;
        optimal_chunk_size = chunk_size;
      }

      // Clean up from FIL
      if (!p_rest.use_experimental) {
        ML::fil::free(*handle, forest);
      }
    }
    std::cout << p_rest.use_experimental << ": " << optimal_chunk_size;

    // Build optimal FIL tree
    tl_params.threads_per_tree = optimal_chunk_size;
    ML::fil::from_treelite(*handle, &forest_variant, model, &tl_params);
    forest = std::get<ML::fil::forest_t<float>>(forest_variant);

    handle->sync_stream();
    handle->sync_stream_pool();

    // only time prediction
    this->loopOnState(state, [this, &filex_model, optimal_chunk_size]() {
      for (int i = 0; i < p_rest.predict_repetitions; i++) {
        auto nvtx_range = raft::common::nvtx::range{"repetition_loop"};
        if (p_rest.use_experimental) {
          auto nvtx_range2 = raft::common::nvtx::range{"filex_outer_predict"};
          filex_model.predict(
            *handle,
            this->data.y.data(),
            this->data.X.data(),
            this->params.nrows,
            kayak::device_type::gpu,
            kayak::device_type::gpu,
            optimal_chunk_size
          );
        } else {
          auto nvtx_range3 = raft::common::nvtx::range{"legacy_outer_predict"};
          ML::fil::predict(*this->handle,
                           this->forest,
                           this->data.y.data(),
                           this->data.X.data(),
                           this->params.nrows,
                           false);
        }
        auto nvtx_range4 = raft::common::nvtx::range{"syncup"};
        handle->sync_stream();
        handle->sync_stream_pool();
      }
    }, false);
  }

  void allocateBuffers(const ::benchmark::State& state) override { Base::allocateBuffers(state); }

  void deallocateBuffers(const ::benchmark::State& state) override
  {
    ML::fil::free(*handle, forest);
    Base::deallocateBuffers(state);
  }

 private:
  ML::fil::forest_t<float> forest;
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
  int chunk_size;
  bool use_experimental;
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

  p.rf = set_rf_params(10,                 /*max_depth */
                       (1 << 20),          /* max_leaves */
                       1.f,                /* max_features */
                       32,                 /* max_n_bins */
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
    {(int)1e6, 20, 1, 5, 1000, storage_type_t::DENSE, algo_t::BATCH_TREE_REORG, false},
    {(int)1e6, 20, 1, 5, 1000, storage_type_t::DENSE, algo_t::BATCH_TREE_REORG, true},
    /* {(int)1e6, 20, 1, 28, 1000, storage_type_t::SPARSE, algo_t::NAIVE, 16, false},
    {(int)1e6, 20, 1, 28, 1000, storage_type_t::SPARSE, algo_t::NAIVE, true},
    {(int)1e6, 20, 1, 5, 100, storage_type_t::DENSE, algo_t::BATCH_TREE_REORG, false},
    {(int)1e6, 20, 1, 5, 100, storage_type_t::DENSE, algo_t::BATCH_TREE_REORG, true},
    {(int)1e6, 20, 1, 5, 10000, storage_type_t::DENSE, algo_t::BATCH_TREE_REORG, false},
    {(int)1e6, 20, 1, 5, 10000, storage_type_t::DENSE, algo_t::BATCH_TREE_REORG, true},
    {(int)1e6, 200, 1, 5, 1000, storage_type_t::DENSE, algo_t::BATCH_TREE_REORG, false},
    {(int)1e6, 200, 1, 5, 1000, storage_type_t::DENSE,
      algo_t::BATCH_TREE_REORG, true}, */
  };
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
    p.use_experimental         = i.use_experimental;
    p.predict_repetitions      = 10;
    out.push_back(p);
  }
  return out;
}

ML_BENCH_REGISTER(Params, FILEX, "", getInputs());

}  // end namespace fil
}  // end namespace Bench
}  // end namespace ML
