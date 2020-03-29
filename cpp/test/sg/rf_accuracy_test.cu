/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cuda_utils.h>
#include <gtest/gtest.h>
#include <random/rng.h>
#include <cuml/ensemble/randomforest.hpp>

namespace ML {

using namespace MLCommon;

struct RFInputs {
  int n_rows_train;
  int n_rows_test;
  uint64_t seed;
  int n_reps;
  float pct_zero_class;
  float min_expected_acc;
};

template <typename T>
class RFClassifierAccuracyTest : public ::testing::TestWithParam<RFInputs> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<RFInputs>::GetParam();
    rng.reset(new Random::Rng(params.seed));
    CUDA_CHECK(cudaStreamCreate(&stream));
    handle.reset(new cumlHandle(1));
    handle->setStream(stream);
    auto allocator = handle->getDeviceAllocator();
    setRFParams();
    X_train = (T *)allocator->allocate(params.n_rows_train * sizeof(T), stream);
    y_train =
      (int *)allocator->allocate(params.n_rows_train * sizeof(int), stream);
    X_test = (T *)allocator->allocate(params.n_rows_test * sizeof(T), stream);
    y_test =
      (int *)allocator->allocate(params.n_rows_test * sizeof(int), stream);
    y_pred =
      (int *)allocator->allocate(params.n_rows_test * sizeof(int), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void TearDown() override {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto allocator = handle->getDeviceAllocator();
    allocator->deallocate(X_train, params.n_rows_train * sizeof(T), stream);
    allocator->deallocate(y_train, params.n_rows_train * sizeof(int), stream);
    allocator->deallocate(X_test, params.n_rows_test * sizeof(T), stream);
    allocator->deallocate(y_test, params.n_rows_test * sizeof(int), stream);
    allocator->deallocate(y_pred, params.n_rows_test * sizeof(int), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    handle.reset();
    rng.reset();
  }

  void runTest() {
    for (int i = 0; i < params.n_reps; ++i) {
      loadData(X_train, y_train, params.n_rows_train, 1);
      loadData(X_test, y_test, params.n_rows_test, 1);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      auto accuracy = runTrainAndTest();
      ASSERT_GT(accuracy, params.min_expected_acc) << " @repetition=" << i;
    }
  }

 private:
  void setRFParams() {
    DecisionTree::DecisionTreeParams tree_params;
    auto algo = SPLIT_ALGO::GLOBAL_QUANTILE;
    auto sc = CRITERION::CRITERION_END;
    set_tree_params(tree_params, 1, /* max_depth */
                    -1,             /* max_leaves */
                    1.0,            /* max_features */
                    16,             /* n_bins */
                    algo,           /* split_algo */
                    2,              /* min_rows_per_node */
                    0.f,            /* min_impurity_decrease */
                    false,          /* bootstrap_features */
                    sc,             /* split_criterion */
                    false,          /* quantile_per_tree */
                    false           /* shuffle_features */
    );
    set_all_rf_params(rfp, 1, /* n_trees */
                      true,   /* bootstrap */
                      1.0,    /* rows_sample */
                      -1,     /* seed */
                      1,      /* n_streams */
                      tree_params);
  }

  void loadData(T *X, int *y, int nrows, int ncols) {
    rng->uniform(X, nrows * ncols, T(-1.0), T(1.0), stream);
    rng->bernoulli<float, int>(y, nrows, params.pct_zero_class, stream);
  }

  float runTrainAndTest() {
    auto *forest = new RandomForestMetaData<T, int>;
    forest->trees = nullptr;
    auto &h = *(handle.get());
    fit(h, forest, X_train, params.n_rows_train, 1, y_train, 2, rfp);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    predict(h, forest, X_test, params.n_rows_test, 1, y_pred, false);
    auto metrics = score(h, forest, y_test, params.n_rows_test, y_pred, false);
    delete[] forest->trees;
    delete forest;
    return metrics.accuracy;
  }

  RFInputs params;
  RF_params rfp;
  std::shared_ptr<cumlHandle> handle;
  cudaStream_t stream;
  T *X_train, *X_test;
  int *y_train, *y_test, *y_pred;
  std::shared_ptr<Random::Rng> rng;
};

const std::vector<RFInputs> inputs = {
  {800, 200, 12345ULL, 40, 0.5f, 0.4f},  {800, 200, 12345ULL, 40, 0.8f, 0.7f},
  {800, 200, 67890ULL, 40, 0.5f, 0.4f},  {800, 200, 67890ULL, 40, 0.8f, 0.7f},
  {1000, 250, 67890ULL, 40, 0.9f, 0.8f}, {1000, 250, 67890ULL, 40, 0.1f, 0.8f},
};

#define DEFINE_TEST(clz, name, testName, params) \
  typedef clz name;                              \
  TEST_P(name, Test) { runTest(); }              \
  INSTANTIATE_TEST_CASE_P(testName, name, ::testing::ValuesIn(params))

DEFINE_TEST(RFClassifierAccuracyTest<float>, ClsTestF, RFAccuracy, inputs);
DEFINE_TEST(RFClassifierAccuracyTest<double>, ClsTestD, RFAccuracy, inputs);

#undef DEFINE_TEST

}  // end namespace ML
