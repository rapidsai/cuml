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

#include <test_utils.h>

#include <decisiontree/batched-levelalgo/kernels.cuh>
#include <decisiontree/batched-levelalgo/quantile.cuh>

#include <cuml/fil/fil.h>
#include <cuml/tree/algo_helper.h>
#include <cuml/datasets/make_blobs.hpp>
#include <cuml/ensemble/randomforest.hpp>

#include <random/make_blobs.cuh>

#include <raft/cudart_utils.h>
#include <raft/linalg/transpose.h>
#include <raft/cuda_utils.cuh>
#include <raft/handle.hpp>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <random>
#include <tuple>
#include <type_traits>

namespace ML {

// Utils for changing tuple into struct
namespace detail {
template <typename result_type, typename... types, std::size_t... indices>
result_type make_struct(std::tuple<types...> t,
                        std::index_sequence<indices...>)  // &, &&, const && etc.
{
  return {std::get<indices>(t)...};
}

}  // namespace detail

template <typename result_type, typename... types>
result_type make_struct(std::tuple<types...> t)  // &, &&, const && etc.
{
  return detail::make_struct<result_type, types...>(
    t, std::index_sequence_for<types...>{});  // if there is repeated types, then the change for
                                              // using std::index_sequence_for is trivial
}

template <int I, typename RandomGenT, typename ParamT, typename T>
void SampleWithoutReplacemment(RandomGenT& gen, std::vector<ParamT>& sample, std::vector<T> x)
{
  std::vector<T> parameter_sample(sample.size());
  std::shuffle(x.begin(), x.end(), gen);
  for (size_t i = 0; i < sample.size(); i++) {
    parameter_sample[i] = x[i % x.size()];
  }
  std::shuffle(parameter_sample.begin(), parameter_sample.end(), gen);
  for (size_t i = 0; i < sample.size(); i++) {
    std::get<I>(sample[i]) = parameter_sample[i];
  }
}

template <int I, typename RandomGenT, typename ParamT, typename T, typename... Args>
void AddParameters(RandomGenT& gen, std::vector<ParamT>& sample, std::vector<T> x, Args... args)
{
  SampleWithoutReplacemment<I>(gen, sample, x);
  if constexpr (sizeof...(args) > 0) { AddParameters<I + 1>(gen, sample, args...); }
}

template <typename ParamT, typename... Args>
std::vector<ParamT> SampleParameters(int num_samples, size_t seed, Args... args)
{
  std::vector<typename ParamT::types> tuple_sample(num_samples);
  std::default_random_engine gen(seed);
  AddParameters<0>(gen, tuple_sample, args...);
  std::vector<ParamT> sample(num_samples);
  for (int i = 0; i < num_samples; i++) {
    sample[i] = make_struct<ParamT>(tuple_sample[i]);
  }
  return sample;
}

struct RfTestParams {
  int n_rows;
  int n_cols;
  int n_trees;
  float max_features;
  float max_samples;
  int max_depth;
  int max_leaves;
  bool bootstrap;
  int n_bins;
  int min_samples_leaf;
  int min_samples_split;
  float min_impurity_decrease;
  int n_streams;
  CRITERION split_criterion;
  int seed;
  int n_labels;
  bool double_precision;
  // c++ has no reflection, so we enumerate the types here
  // This must be updated if new fields are added
  using types = std::tuple<int,
                           int,
                           int,
                           float,
                           float,
                           int,
                           int,
                           bool,
                           int,
                           int,
                           int,
                           float,
                           int,
                           CRITERION,
                           int,
                           int,
                           bool>;
};

std::ostream& operator<<(std::ostream& os, const RfTestParams& ps)
{
  os << "n_rows = " << ps.n_rows << ", n_cols = " << ps.n_cols;
  os << ", n_trees = " << ps.n_trees << ", max_features = " << ps.max_features;
  os << ", max_samples = " << ps.max_samples << ", max_depth = " << ps.max_depth;
  os << ", max_leaves = " << ps.max_leaves << ", bootstrap = " << ps.bootstrap;
  os << ", n_bins = " << ps.n_bins << ", min_samples_leaf = " << ps.min_samples_leaf;
  os << ", min_samples_split = " << ps.min_samples_split;
  os << ", min_impurity_decrease = " << ps.min_impurity_decrease
     << ", n_streams = " << ps.n_streams;
  os << ", split_criterion = " << ps.split_criterion << ", seed = " << ps.seed;
  os << ", n_labels = " << ps.n_labels << ", double_precision = " << ps.double_precision;
  return os;
}

template <typename DataT, typename LabelT>
auto FilPredict(const raft::handle_t& handle,
                RfTestParams params,
                DataT* X_transpose,
                RandomForestMetaData<DataT, LabelT>* forest)
{
  auto pred = std::make_shared<thrust::device_vector<float>>(params.n_rows);
  ModelHandle model;
  std::size_t num_outputs = 1;
  if constexpr (std::is_integral_v<LabelT>) { num_outputs = params.n_labels; }
  build_treelite_forest(&model, forest, params.n_cols);
  fil::treelite_params_t tl_params{fil::algo_t::ALGO_AUTO,
                                   num_outputs > 1,
                                   1.f / num_outputs,
                                   fil::storage_type_t::AUTO,
                                   8,
                                   1,
                                   0,
                                   nullptr};
  fil::forest_t fil_forest;
  fil::from_treelite(handle, &fil_forest, model, &tl_params);
  fil::predict(handle, fil_forest, pred->data().get(), X_transpose, params.n_rows, false);
  return pred;
}

template <typename DataT, typename LabelT>
auto FilPredictProba(const raft::handle_t& handle,
                     RfTestParams params,
                     DataT* X_transpose,
                     RandomForestMetaData<DataT, LabelT>* forest)
{
  std::size_t num_outputs = params.n_labels;
  auto pred = std::make_shared<thrust::device_vector<float>>(params.n_rows * num_outputs);
  ModelHandle model;
  static_assert(std::is_integral_v<LabelT>, "Must be classification");
  build_treelite_forest(&model, forest, params.n_cols);
  fil::treelite_params_t tl_params{
    fil::algo_t::ALGO_AUTO, 0, 0.0f, fil::storage_type_t::AUTO, 8, 1, 0, nullptr};
  fil::forest_t fil_forest;
  fil::from_treelite(handle, &fil_forest, model, &tl_params);
  fil::predict(handle, fil_forest, pred->data().get(), X_transpose, params.n_rows, true);
  return pred;
}
template <typename DataT, typename LabelT>
auto TrainScore(
  const raft::handle_t& handle, RfTestParams params, DataT* X, DataT* X_transpose, LabelT* y)
{
  RF_params rf_params = set_rf_params(params.max_depth,
                                      params.max_leaves,
                                      params.max_features,
                                      params.n_bins,
                                      params.min_samples_leaf,
                                      params.min_samples_split,
                                      params.min_impurity_decrease,
                                      params.bootstrap,
                                      params.n_trees,
                                      params.max_samples,
                                      0,
                                      params.split_criterion,
                                      params.n_streams,
                                      128);

  auto forest     = std::make_shared<RandomForestMetaData<DataT, LabelT>>();
  auto forest_ptr = forest.get();
  if constexpr (std::is_integral_v<LabelT>) {
    fit(handle, forest_ptr, X, params.n_rows, params.n_cols, y, params.n_labels, rf_params);
  } else {
    fit(handle, forest_ptr, X, params.n_rows, params.n_cols, y, rf_params);
  }

  auto pred = std::make_shared<thrust::device_vector<LabelT>>(params.n_rows);
  predict(handle, forest_ptr, X_transpose, params.n_rows, params.n_cols, pred->data().get());

  // Predict and compare against known labels
  RF_metrics metrics = score(handle, forest_ptr, y, params.n_rows, pred->data().get());
  return std::make_tuple(forest, pred, metrics);
}

template <typename DataT, typename LabelT>
class RfSpecialisedTest {
 public:
  RfSpecialisedTest(RfTestParams params) : params(params)
  {
    raft::handle_t handle(params.n_streams);
    X.resize(params.n_rows * params.n_cols);
    X_transpose.resize(params.n_rows * params.n_cols);
    y.resize(params.n_rows);
    // Make data
    if constexpr (std::is_integral<LabelT>::value) {
      Datasets::make_blobs(handle,
                           X.data().get(),
                           y.data().get(),
                           params.n_rows,
                           params.n_cols,
                           params.n_labels,
                           false,
                           nullptr,
                           nullptr,
                           5.0,
                           false,
                           -10.0f,
                           10.0f,
                           params.seed);
    } else {
      thrust::device_vector<int> y_temp(params.n_rows);
      Datasets::make_blobs(handle,
                           X.data().get(),
                           y_temp.data().get(),
                           params.n_rows,
                           params.n_cols,
                           params.n_labels,
                           false,
                           nullptr,
                           nullptr,
                           5.0,
                           false,
                           -10.0f,
                           10.0f,
                           params.seed);
      // if regression, make the labels normally distributed
      raft::random::Rng r(4);
      thrust::device_vector<double> normal(params.n_rows);
      r.normal(normal.data().get(), normal.size(), 0.0, 2.0, nullptr);
      thrust::transform(
        normal.begin(), normal.end(), y_temp.begin(), y.begin(), thrust::plus<LabelT>());
    }
    raft::linalg::transpose(
      handle, X.data().get(), X_transpose.data().get(), params.n_rows, params.n_cols, nullptr);
    forest.reset(new typename ML::RandomForestMetaData<DataT, LabelT>);
    std::tie(forest, predictions, training_metrics) =
      TrainScore(handle, params, X.data().get(), X_transpose.data().get(), y.data().get());

    Test();
  }
  // Current model should be at least as accurate as a model with depth - 1
  void TestAccuracyImprovement()
  {
    if (params.max_depth <= 1) { return; }
    // avereraging between models can introduce variance
    if (params.n_trees > 1) { return; }
    // accuracy is not guaranteed to improve with bootstrapping
    if (params.bootstrap) { return; }
    raft::handle_t handle(params.n_streams);
    RfTestParams alt_params = params;
    alt_params.max_depth--;
    auto [alt_forest, alt_predictions, alt_metrics] =
      TrainScore(handle, alt_params, X.data().get(), X_transpose.data().get(), y.data().get());
    double eps = 1e-8;
    if (params.split_criterion == MSE) {
      EXPECT_LE(training_metrics.mean_squared_error, alt_metrics.mean_squared_error + eps);
    } else if (params.split_criterion == MAE) {
      EXPECT_LE(training_metrics.mean_abs_error, alt_metrics.mean_abs_error + eps);
    } else {
      EXPECT_GE(training_metrics.accuracy, alt_metrics.accuracy);
    }
  }
  // Regularisation parameters are working correctly
  void TestTreeSize()
  {
    for (int i = 0u; i < forest->rf_params.n_trees; i++) {
      // Check we have actually built something, otherwise these tests can all pass when the tree
      // algorithm produces only stumps
      size_t effective_rows = params.n_rows * params.max_samples;
      if (params.max_depth > 0 && params.min_impurity_decrease == 0 && effective_rows >= 100) {
        EXPECT_GT(forest->trees[i]->leaf_counter, 1);
      }

      // Check number of leaves is accurate
      int num_leaves = 0;
      for (auto n : forest->trees[i]->sparsetree) {
        num_leaves += n.IsLeaf();
      }
      EXPECT_EQ(num_leaves, forest->trees[i]->leaf_counter);
      if (params.max_leaves > 0) { EXPECT_LE(forest->trees[i]->leaf_counter, params.max_leaves); }

      EXPECT_LE(forest->trees[i]->depth_counter, params.max_depth);
      EXPECT_LE(forest->trees[i]->leaf_counter,
                raft::ceildiv(params.n_rows, params.min_samples_leaf));
    }
  }
  void TestMinImpurity()
  {
    for (int i = 0u; i < forest->rf_params.n_trees; i++) {
      for (auto n : forest->trees[i]->sparsetree) {
        if (!n.IsLeaf()) { EXPECT_GT(n.BestMetric(), params.min_impurity_decrease); }
      }
    }
  }
  void TestDeterminism()

  {
    // Regression models use floating point atomics, so are not bitwise reproducible
    bool is_regression = params.split_criterion == MSE || params.split_criterion == MAE;
    if (is_regression) return;

    // Repeat training
    raft::handle_t handle(params.n_streams);
    auto [alt_forest, alt_predictions, alt_metrics] =
      TrainScore(handle, params, X.data().get(), X_transpose.data().get(), y.data().get());

    for (int i = 0u; i < forest->rf_params.n_trees; i++) {
      EXPECT_EQ(forest->trees[i]->sparsetree, alt_forest->trees[i]->sparsetree);
    }
  }
  // Instance counts in children sums up to parent.
  void TestInstanceCounts()
  {
    for (int i = 0u; i < forest->rf_params.n_trees; i++) {
      const auto& tree = forest->trees[i]->sparsetree;
      for (auto n : tree) {
        if (!n.IsLeaf()) {
          auto sum = tree[n.LeftChildId()].InstanceCount() + tree[n.RightChildId()].InstanceCount();
          EXPECT_EQ(sum, n.InstanceCount());
        }
      }
    }
  }

  // Difference between the largest element and second largest
  DataT MinDifference(DataT* begin, std::size_t len)
  {
    std::size_t max_element_index = 0;
    DataT max_element             = 0.0;
    for (std::size_t i = 0; i < len; i++) {
      if (begin[i] > max_element) {
        max_element_index = i;
        max_element       = begin[i];
      }
    }
    DataT second_max_element = 0.0;
    for (std::size_t i = 0; i < len; i++) {
      if (begin[i] > second_max_element && i != max_element_index) {
        second_max_element = begin[i];
      }
    }

    return std::abs(max_element - second_max_element);
  }

  // Compare fil against native rf predictions
  // Only for single precision models
  void TestFilPredict()
  {
    if constexpr (std::is_same_v<DataT, double>) {
      return;
    } else {
      raft::handle_t handle(params.n_streams);
      auto fil_pred = FilPredict(handle, params, X_transpose.data().get(), forest.get());

      thrust::host_vector<float> h_fil_pred(*fil_pred);
      thrust::host_vector<float> h_pred(*predictions);

      thrust::host_vector<float> h_fil_pred_prob;
      if constexpr (std::is_integral_v<LabelT>) {
        h_fil_pred_prob = *FilPredictProba(handle, params, X_transpose.data().get(), forest.get());
      }

      float tol = 1e-2;
      for (std::size_t i = 0; i < h_fil_pred.size(); i++) {
        // If the output probabilities are very similar for different classes
        // FIL may output a different class due to numerical differences
        // Skip these cases
        if constexpr (std::is_integral_v<LabelT>) {
          int num_outputs = forest->trees[0]->num_outputs;
          auto min_diff   = MinDifference(&h_fil_pred_prob[i * num_outputs], num_outputs);
          if (min_diff < tol) continue;
        }

        EXPECT_LE(abs(h_fil_pred[i] - h_pred[i]), tol);
      }
    }
  }
  void Test()
  {
    TestAccuracyImprovement();
    TestDeterminism();
    TestMinImpurity();
    TestTreeSize();
    TestInstanceCounts();
    TestFilPredict();
  }

  RF_metrics training_metrics;
  thrust::device_vector<DataT> X;
  thrust::device_vector<DataT> X_transpose;
  thrust::device_vector<LabelT> y;
  RfTestParams params;
  std::shared_ptr<RandomForestMetaData<DataT, LabelT>> forest;
  std::shared_ptr<thrust::device_vector<LabelT>> predictions;
};

// Dispatch tests based on any template parameters
class RfTest : public ::testing::TestWithParam<RfTestParams> {
 public:
  void SetUp() override
  {
    RfTestParams params = ::testing::TestWithParam<RfTestParams>::GetParam();
    bool is_regression  = params.split_criterion == MSE || params.split_criterion == MAE;
    if (params.double_precision) {
      if (is_regression) {
        RfSpecialisedTest<double, double> test(params);
      } else {
        RfSpecialisedTest<double, int> test(params);
      }
    } else {
      if (is_regression) {
        RfSpecialisedTest<float, float> test(params);
      } else {
        RfSpecialisedTest<float, int> test(params);
      }
    }
  }
};

TEST_P(RfTest, PropertyBasedTest) {}

// Parameter ranges to test
std::vector<int> n_rows                  = {10, 100, 1452};
std::vector<int> n_cols                  = {1, 5, 152, 1014};
std::vector<int> n_trees                 = {1, 5, 17};
std::vector<float> max_features          = {0.1f, 0.5f, 1.0f};
std::vector<float> max_samples           = {0.1f, 0.5f, 1.0f};
std::vector<int> max_depth               = {1, 10, 30};
std::vector<int> max_leaves              = {-1, 16, 50};
std::vector<bool> bootstrap              = {false, true};
std::vector<int> n_bins                  = {2, 57, 128, 256};
std::vector<int> min_samples_leaf        = {1, 10, 30};
std::vector<int> min_samples_split       = {2, 10};
std::vector<float> min_impurity_decrease = {0.0f, 1.0f, 10.0f};
std::vector<int> n_streams               = {1, 2, 10};
std::vector<CRITERION> split_criterion   = {CRITERION::MSE, CRITERION::GINI, CRITERION::ENTROPY};
std::vector<int> seed                    = {0, 17};
std::vector<int> n_labels                = {2, 10, 20};
std::vector<bool> double_precision       = {false, true};

int n_tests = 100;

INSTANTIATE_TEST_CASE_P(RfTests,
                        RfTest,
                        ::testing::ValuesIn(SampleParameters<RfTestParams>(n_tests,
                                                                           0,
                                                                           n_rows,
                                                                           n_cols,
                                                                           n_trees,
                                                                           max_features,
                                                                           max_samples,
                                                                           max_depth,
                                                                           max_leaves,
                                                                           bootstrap,
                                                                           n_bins,
                                                                           min_samples_leaf,
                                                                           min_samples_split,
                                                                           min_impurity_decrease,
                                                                           n_streams,
                                                                           split_criterion,
                                                                           seed,
                                                                           n_labels,
                                                                           double_precision)));

struct QuantileTestParameters {
  int n_rows;
  int n_bins;
  uint64_t seed;
};

template <typename T>
class RFQuantileBinsLowerBoundTest : public ::testing::TestWithParam<QuantileTestParameters> {
 public:
  void SetUp() override
  {
    auto params = ::testing::TestWithParam<QuantileTestParameters>::GetParam();

    thrust::device_vector<T> data(params.n_rows);
    thrust::host_vector<T> h_data(params.n_rows);
    thrust::host_vector<T> h_quantiles(params.n_bins);
    raft::random::Rng r(8);
    r.normal(data.data().get(), data.size(), T(0.0), T(2.0), nullptr);
    raft::handle_t handle;
    auto quantiles =
      DT::computeQuantiles(params.n_bins, data.data().get(), params.n_rows, 1, handle);
    raft::update_host(
      h_quantiles.data(), quantiles->data(), quantiles->size(), handle.get_stream());
    h_data = data;
    for (std::size_t i = 0; i < h_data.size(); ++i) {
      auto d = h_data[i];
      // golden lower bound from thrust
      auto golden_lb = thrust::lower_bound(
                         thrust::seq, h_quantiles.data(), h_quantiles.data() + params.n_bins, d) -
                       h_quantiles.data();
      // lower bound from custom lower_bound impl
      auto lb = DT::lower_bound(h_quantiles.data(), params.n_bins, d);
      ASSERT_EQ(golden_lb, lb)
        << "custom lower_bound method is inconsistent with thrust::lower_bound" << std::endl;
    }
  }
};

template <typename T>
class RFQuantileTest : public ::testing::TestWithParam<QuantileTestParameters> {
 public:
  void SetUp() override
  {
    auto params = ::testing::TestWithParam<QuantileTestParameters>::GetParam();

    thrust::device_vector<T> data(params.n_rows);
    thrust::device_vector<int> histogram(params.n_bins);
    thrust::host_vector<int> h_histogram(params.n_bins);

    raft::random::Rng r(8);
    r.normal(data.data().get(), data.size(), T(0.0), T(2.0), nullptr);
    raft::handle_t handle;
    std::shared_ptr<rmm::device_uvector<T>> quantiles =
      DT::computeQuantiles(params.n_bins, data.data().get(), params.n_rows, 1, handle);

    auto d_quantiles = quantiles->data();
    auto d_histogram = histogram.data().get();
    thrust::for_each(data.begin(), data.end(), [=] __device__(T x) {
      for (int j = 0; j < params.n_bins; j++) {
        if (x <= d_quantiles[j]) {
          atomicAdd(&d_histogram[j], 1);
          break;
        }
      }
    });

    h_histogram           = histogram;
    int max_items_per_bin = raft::ceildiv(params.n_rows, params.n_bins);
    int min_items_per_bin = max_items_per_bin - 1;
    int total_items       = 0;
    for (int b = 0; b < params.n_bins; b++) {
      ASSERT_TRUE(h_histogram[b] == max_items_per_bin || h_histogram[b] == min_items_per_bin)
        << "No. samples in bin[" << b << "] = " << h_histogram[b] << " Expected "
        << max_items_per_bin << " or " << min_items_per_bin << std::endl;
      total_items += h_histogram[b];
    }
    ASSERT_EQ(params.n_rows, total_items)
      << "Some samples from dataset are either missed of double counted in quantile bins"
      << std::endl;
  }
};

//-------------------------------------------------------------------------------------------------------------------------------------
const std::vector<QuantileTestParameters> inputs = {{1000, 16, 6078587519764079670LLU},
                                                    {1130, 32, 4884670006177930266LLU},
                                                    {1752, 67, 9175325892580481371LLU},
                                                    {2307, 99, 9507819643927052255LLU},
                                                    {5000, 128, 9507819643927052255LLU}};

// float type quantile test
typedef RFQuantileTest<float> RFQuantileTestF;
TEST_P(RFQuantileTestF, test) {}
INSTANTIATE_TEST_CASE_P(RfTests, RFQuantileTestF, ::testing::ValuesIn(inputs));

// double type quantile test
typedef RFQuantileTest<double> RFQuantileTestD;
TEST_P(RFQuantileTestD, test) {}
INSTANTIATE_TEST_CASE_P(RfTests, RFQuantileTestD, ::testing::ValuesIn(inputs));

// float type quantile bins lower bounds test
typedef RFQuantileBinsLowerBoundTest<float> RFQuantileBinsLowerBoundTestF;
TEST_P(RFQuantileBinsLowerBoundTestF, test) {}
INSTANTIATE_TEST_CASE_P(RfTests, RFQuantileBinsLowerBoundTestF, ::testing::ValuesIn(inputs));

// double type quantile bins lower bounds lest
typedef RFQuantileBinsLowerBoundTest<double> RFQuantileBinsLowerBoundTestD;
TEST_P(RFQuantileBinsLowerBoundTestD, test) {}
INSTANTIATE_TEST_CASE_P(RfTests, RFQuantileBinsLowerBoundTestD, ::testing::ValuesIn(inputs));

TEST(RfTest, TextDump)
{
  RF_params rf_params = set_rf_params(2, 2, 1.0, 2, 1, 2, 0.0, true, 1, 1.0, 0, GINI, 1, 128);
  auto forest         = std::make_shared<RandomForestMetaData<float, int>>();

  std::vector<float> X_host      = {1, 2, 3, 6, 7, 8};
  thrust::device_vector<float> X = X_host;
  std::vector<int> y_host        = {0, 0, 1, 1, 1, 0};
  thrust::device_vector<int> y   = y_host;

  raft::handle_t handle(1);
  auto forest_ptr = forest.get();
  fit(handle, forest_ptr, X.data().get(), y.size(), 1, y.data().get(), 2, rf_params);

  std::string expected_start_text = R"(Forest has 1 trees, max_depth 2, and max_leaves 2
Tree #0
 Decision Tree depth --> 1 and n_leaves --> 2
 Tree Fitting - Overall time -->)";

  std::string expected_end_text = R"(└(colid: 0, quesval: 3, best_metric_val: 0.25)
    ├(leaf, prediction: [0.75, 0.25], best_metric_val: 0)
    └(leaf, prediction: [0, 1], best_metric_val: 0))";

  EXPECT_TRUE(get_rf_detailed_text(forest_ptr).find(expected_start_text) != std::string::npos);
  EXPECT_TRUE(get_rf_detailed_text(forest_ptr).find(expected_end_text) != std::string::npos);

  std::string expected_json = R"([
{"nodeid": 0, "split_feature": 0, "split_threshold": 3, "gain": 0.25, "instance_count": 6, "yes": 1, "no": 2, "children": [
  {"nodeid": 1, "leaf_value": [0.75, 0.25], "instance_count": 4},
  {"nodeid": 2, "leaf_value": [0, 1], "instance_count": 2}
]}
])";
  EXPECT_EQ(get_rf_json(forest_ptr), expected_json);
}

}  // end namespace ML
