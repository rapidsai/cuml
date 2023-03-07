/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <cuml/experimental/fil/detail/raft_proto/device_type.hpp>
#include <cuml/experimental/fil/tree_layout.hpp>

#include "benchmark.cuh"
#include <chrono>
#include <cstdint>
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
  }

 protected:
  void runBenchmark(::benchmark::State& state) override
  {
    if (!params.rowMajor) { state.SkipWithError("FIL only supports row-major inputs"); }
    // create model
    ML::RandomForestRegressorF rf_model;
    auto* mPtr         = &rf_model;
    auto train_nrows = std::min(params.nrows, 1000);
    fit(*handle, mPtr, data.X.data(), train_nrows, params.ncols, data.y.data(), p_rest.rf);
    handle->sync_stream(stream);

    ML::build_treelite_forest(&model, &rf_model, params.ncols);

    auto filex_model = ML::experimental::fil::import_from_treelite_handle(
      model,
      ML::experimental::fil::tree_layout::breadth_first,
      128,
      false,
      raft_proto::device_type::gpu,
      0,
      stream
    );

    ML::fil::treelite_params_t tl_params = {
      .algo              = ML::fil::algo_t::NAIVE,
      .output_class      = false,
      .threshold         = 1.f / params.nclasses,  // Fixture::DatasetParams
      .storage_type      = p_rest.storage,
      .blocks_per_sm     = 8,
      .threads_per_tree  = 1,
      .n_items           = 0,
      .pforest_shape_str = nullptr};
    ML::fil::forest_variant forest_variant;
    auto optimal_chunk_size = 1;
    auto optimal_storage_type = p_rest.storage;
    auto optimal_algo_type = ML::fil::algo_t::NAIVE;
    auto optimal_layout = ML::experimental::fil::tree_layout::breadth_first;
    auto allowed_storage_types = std::vector<ML::fil::storage_type_t>{};
    if (p_rest.storage == ML::fil::storage_type_t::DENSE) {
      allowed_storage_types.push_back(ML::fil::storage_type_t::DENSE);
      allowed_storage_types.push_back(ML::fil::storage_type_t::SPARSE);
      allowed_storage_types.push_back(ML::fil::storage_type_t::SPARSE8);
    } else {
      allowed_storage_types.push_back(ML::fil::storage_type_t::SPARSE);
      allowed_storage_types.push_back(ML::fil::storage_type_t::SPARSE8);
    }
    auto allowed_layouts = std::vector<ML::experimental::fil::tree_layout>{
      ML::experimental::fil::tree_layout::breadth_first,
      ML::experimental::fil::tree_layout::depth_first,
    };
    auto min_time = std::numeric_limits<std::int64_t>::max();

    // Iterate through storage type, algorithm type, and chunk sizes and find optimum
    for (auto storage_type : allowed_storage_types) {
      auto allowed_algo_types = std::vector<ML::fil::algo_t>{};
      allowed_algo_types.push_back(ML::fil::algo_t::NAIVE);
      if (storage_type == ML::fil::storage_type_t::DENSE) {
        allowed_algo_types.push_back(ML::fil::algo_t::TREE_REORG);
        allowed_algo_types.push_back(ML::fil::algo_t::BATCH_TREE_REORG);
      }
      tl_params.storage_type = storage_type;

      for (auto algo_type : allowed_algo_types) {
        tl_params.algo = algo_type;
        for (auto layout : allowed_layouts) {
          filex_model = ML::experimental::fil::import_from_treelite_handle(
            model,
            layout,
            128,
            false,
            raft_proto::device_type::gpu,
            0,
            stream
          );
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
                  raft_proto::device_type::gpu,
                  raft_proto::device_type::gpu,
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
            }
            handle->sync_stream();
            handle->sync_stream_pool();
            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
              end - start
            ).count();
            if (elapsed < min_time) {
              min_time = elapsed;
              optimal_chunk_size = chunk_size;
              optimal_storage_type = storage_type;
              optimal_algo_type = algo_type;
              optimal_layout = layout;
            }

            // Clean up from FIL
            if (!p_rest.use_experimental) {
              ML::fil::free(*handle, forest);
            }
          }
          if (!p_rest.use_experimental) {
            break;
          }
        }
        if (p_rest.use_experimental) {
          break;
        }
      }
      if (p_rest.use_experimental) {
        break;
      }
    }

    // Build optimal FIL tree
    tl_params.storage_type = optimal_storage_type;
    tl_params.algo = optimal_algo_type;
    tl_params.threads_per_tree = optimal_chunk_size;
    ML::fil::from_treelite(*handle, &forest_variant, model, &tl_params);
    forest = std::get<ML::fil::forest_t<float>>(forest_variant);
    filex_model = ML::experimental::fil::import_from_treelite_handle(
      model,
      optimal_layout,
      128,
      false,
      raft_proto::device_type::gpu,
      0,
      stream
    );

    handle->sync_stream();
    handle->sync_stream_pool();

    // only time prediction
    this->loopOnState(state, [this, &filex_model, optimal_chunk_size]() {
      for (int i = 0; i < p_rest.predict_repetitions; i++) {
        if (p_rest.use_experimental) {
          filex_model.predict(
            *handle,
            this->data.y.data(),
            this->data.X.data(),
            this->params.nrows,
            raft_proto::device_type::gpu,
            raft_proto::device_type::gpu,
            optimal_chunk_size
          );
          handle->sync_stream();
          handle->sync_stream_pool();
        } else {
          ML::fil::predict(*this->handle,
                           this->forest,
                           this->data.y.data(),
                           this->data.X.data(),
                           this->params.nrows,
                           false);
          handle->sync_stream();
          handle->sync_stream_pool();
        }
      }
    }, true);
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
    {(int)1e6, 20, 1, 10, 1000, storage_type_t::DENSE, false},
    {(int)1e6, 20, 1, 10, 1000, storage_type_t::DENSE, true},
    {(int)1e6, 20, 1, 3, 1000, storage_type_t::DENSE, false},
    {(int)1e6, 20, 1, 3, 1000, storage_type_t::DENSE, true},
    {(int)1e6, 20, 1, 28, 1000, storage_type_t::SPARSE, false},
    {(int)1e6, 20, 1, 28, 1000, storage_type_t::SPARSE, true},
    {(int)1e6, 20, 1, 10, 100, storage_type_t::DENSE, false},
    {(int)1e6, 20, 1, 10, 100, storage_type_t::DENSE, true},
    {(int)1e6, 20, 1, 10, 10000, storage_type_t::DENSE, false},
    {(int)1e6, 20, 1, 10, 10000, storage_type_t::DENSE, true},
    {(int)1e6, 200, 1, 10, 1000, storage_type_t::DENSE, false},
    {(int)1e6, 200, 1, 10, 1000, storage_type_t::DENSE, true}
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
