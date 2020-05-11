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
#include <treelite/c_api.h>
#include <treelite/tree.h>
#include <cuml/common/logger.hpp>
#include <cuml/cuml.hpp>
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
};

class FIL : public RegressionFixture<float> {
 public:
  FIL(const std::string& name, const Params& p)
    : RegressionFixture<float>("FIL", p.data, p.blobs),
      model(p.model),
      predict_proba(p.predict_proba) {
    this->SetName(name.c_str());
  }

 protected:
  void runBenchmark(::benchmark::State& state) override {
    if (!this->params.rowMajor) {
      state.SkipWithError("FIL only supports row-major inputs");
    }
    auto& handle = *this->handle;
    auto stream = handle.getStream();
    for (auto _ : state) {
      MLCommon::Bench::CudaEventTimer timer(state, nullptr, 0, stream);
      ML::fil::predict(handle, forest, this->data.y, this->data.X,
                       this->params.nrows, predict_proba);
    }
  }

  void allocateBuffers(const ::benchmark::State& state) override {
    //auto stream = this->handle->getStream();
    ML::fil::treelite_params_t tl_params = {
      .algo = ML::fil::algo_t::ALGO_AUTO,
      .output_class = true,                      // cuML RF forest
      .threshold = 1.f / this->params.nclasses,  //Fixture::DatasetParams
      .storage_type = ML::fil::storage_type_t::AUTO};
    auto& handle = *this->handle;
    ML::fil::from_treelite(handle, &forest, model, &tl_params);
  }

  void deallocateBuffers(const ::benchmark::State& state) override {
    //auto allocator = this->handle->getDeviceAllocator();
    ML::fil::free(*handle, forest);
  }

 private:
  //int nIter;
  ML::fil::forest_t forest;
  ModelHandle model;
  bool predict_proba;
};

struct FilBenchParams {
  size_t nrows;
  size_t ncols;
  size_t nclasses;
  bool predict_proba;
};

size_t getSizeFromEnv(const char* name) {
  const char* s = std::getenv(name);
  ASSERT(s != nullptr, "%s", name);
  signed long size = atol(s);
  // todo: implement proper mechanism to pass benchmark parameters
  ASSERT(size > 0, "%s", name);
  return (size_t)size;
}

std::vector<Params> getInputs() {
  std::vector<Params> out;
  Params p;
  size_t ncols;
  const char* pp = std::getenv("TL_MODEL_PROTO_PATH");
  if (pp != nullptr) {
    TREELITE_CHECK(TreeliteLoadProtobufModel(pp, &p.model));
    ncols = getSizeFromEnv("NCOLS");
  } else
    return out;
  p.data.rowMajor = true;
  // see src_prims/random/make_regression.h
  p.blobs = {.n_informative = (signed)ncols / 3,
             .effective_rank = 2 * (signed)ncols / 3,
             .bias = 0.f,
             .tail_strength = 0.1,
             .noise = 0.01,
             .shuffle = false,
             .seed = 12345ULL};
  std::vector<FilBenchParams> rowcols = {
    {10123ul, ncols, 2ul, false}, {10123ul, ncols, 2ul, true},
    //{1184000, ncols, 2, false},  // Mimicking Bosch dataset
  };
  for (auto& rc : rowcols) {
    p.data.nrows = rc.nrows;
    p.data.ncols = rc.ncols;
    p.data.nclasses = rc.nclasses;
    p.predict_proba = rc.predict_proba;
    //for (auto max_depth : std::vector<int>({8, 10})) {
    //  p.rf.tree_params.max_depth = max_depth;
    out.push_back(p);
    //}
  }
  return out;
}

ML_BENCH_REGISTER(Params, FIL, "", getInputs());

}  // end namespace fil
}  // end namespace Bench
}  // end namespace ML
