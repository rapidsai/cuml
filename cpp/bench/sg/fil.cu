/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#include <cuml/common/logger.hpp>
#include <cuml/ensemble/randomforest.hpp>
#include <cuml/fil/detail/raft_proto/device_type.hpp>
#include <cuml/fil/infer_kind.hpp>
#include <cuml/fil/tree_layout.hpp>
#include <cuml/fil/treelite_importer.hpp>
#include <cuml/tree/algo_helper.h>

#include <treelite/tree.h>

#include <chrono>
#include <cstdint>
#include <utility>

namespace ML {
namespace Bench {
namespace fil {

struct Params {
  DatasetParams data;
  RegressionParams blobs;
  TreeliteModelHandle model;
  RF_params rf;
  int predict_repetitions;
};

class FIL : public RegressionFixture<float> {
  typedef RegressionFixture<float> Base;

 public:
  FIL(const std::string& name, const Params& p)
    : RegressionFixture<float>(name, p.data, p.blobs), model(p.model), p_rest(p)
  {
  }

 protected:
  void runBenchmark(::benchmark::State& state) override
  {
    if (!params.rowMajor) { state.SkipWithError("FIL only supports row-major inputs"); }
    // create model
    ML::RandomForestRegressorF rf_model;
    auto* mPtr       = &rf_model;
    auto train_nrows = std::min(params.nrows, 1000);
    fit(*handle, mPtr, data.X.data(), train_nrows, params.ncols, data.y.data(), p_rest.rf);
    handle->sync_stream(stream);

    ML::build_treelite_forest(&model, &rf_model, params.ncols);

    auto fil_model = ML::fil::import_from_treelite_handle(model,
                                                          ML::fil::tree_layout::breadth_first,
                                                          128,
                                                          false,
                                                          raft_proto::device_type::gpu,
                                                          0,
                                                          stream);

    auto optimal_chunk_size = 1;
    auto optimal_layout     = ML::fil::tree_layout::breadth_first;
    auto allowed_layouts =
      std::vector<ML::fil::tree_layout>{ML::fil::tree_layout::depth_first,
                                        ML::fil::tree_layout::breadth_first,
                                        ML::fil::tree_layout::layered_children_together};
    auto min_time = std::numeric_limits<std::int64_t>::max();

    // Find optimal configuration
    for (auto layout : allowed_layouts) {
      fil_model = ML::fil::import_from_treelite_handle(
        model, layout, 128, false, raft_proto::device_type::gpu, 0, stream);
      for (auto chunk_size = 1; chunk_size <= 32; chunk_size *= 2) {
        handle->sync_stream();
        handle->sync_stream_pool();
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < p_rest.predict_repetitions; i++) {
          // Create FIL forest
          fil_model.predict(*handle,
                            data.y.data(),
                            data.X.data(),
                            params.nrows,
                            raft_proto::device_type::gpu,
                            raft_proto::device_type::gpu,
                            ML::fil::infer_kind::default_kind,
                            chunk_size);
        }
        handle->sync_stream();
        handle->sync_stream_pool();
        auto end     = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        if (elapsed < min_time) {
          min_time           = elapsed;
          optimal_chunk_size = chunk_size;
          optimal_layout     = layout;
        }
      }
    }

    // Build optimal FIL tree
    fil_model = ML::fil::import_from_treelite_handle(
      model, optimal_layout, 128, false, raft_proto::device_type::gpu, 0, stream);

    handle->sync_stream();
    handle->sync_stream_pool();

    // only time prediction
    this->loopOnState(
      state,
      [this, &fil_model, optimal_chunk_size]() {
        for (int i = 0; i < p_rest.predict_repetitions; i++) {
          fil_model.predict(*handle,
                            this->data.y.data(),
                            this->data.X.data(),
                            this->params.nrows,
                            raft_proto::device_type::gpu,
                            raft_proto::device_type::gpu,
                            ML::fil::infer_kind::default_kind,
                            optimal_chunk_size);
          handle->sync_stream();
          handle->sync_stream_pool();
        }
      },
      true);
  }

  void allocateBuffers(const ::benchmark::State& state) override { Base::allocateBuffers(state); }

  void deallocateBuffers(const ::benchmark::State& state) override
  {
    Base::deallocateBuffers(state);
  }

 private:
  TreeliteModelHandle model;
  Params p_rest;
};

struct FilBenchParams {
  int nrows;
  int ncols;
  int nclasses;
  int max_depth;
  int ntrees;
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

  std::vector<FilBenchParams> var_params = {{(int)1e6, 20, 1, 10, 1000},
                                            {(int)1e6, 20, 1, 3, 1000},
                                            {(int)1e6, 20, 1, 28, 1000},
                                            {(int)1e6, 20, 1, 10, 100},
                                            {(int)1e6, 20, 1, 10, 10000},
                                            {(int)1e6, 200, 1, 10, 1000}};
  for (auto& i : var_params) {
    p.data.nrows               = i.nrows;
    p.data.ncols               = i.ncols;
    p.blobs.n_informative      = i.ncols / 3;
    p.blobs.effective_rank     = i.ncols / 3;
    p.data.nclasses            = i.nclasses;
    p.rf.tree_params.max_depth = i.max_depth;
    p.rf.n_trees               = i.ntrees;
    p.predict_repetitions      = 10;
    out.push_back(p);
  }
  return out;
}

ML_BENCH_REGISTER(Params, FIL, "", getInputs());

}  // namespace fil
}  // end namespace Bench
}  // end namespace ML
