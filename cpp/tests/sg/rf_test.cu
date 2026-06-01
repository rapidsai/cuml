/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuml/common/logger.hpp>
#include <cuml/datasets/make_blobs.hpp>
#include <cuml/ensemble/randomforest.hpp>
#include <cuml/tree/algo_helper.h>

#include <raft/core/handle.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cub/device/device_segmented_reduce.cuh>
#include <cuda/iterator>
#include <cuda/std/functional>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/transform.h>

#include <decisiontree/batched-levelalgo/kernels/builder_kernels.cuh>
#include <decisiontree/batched-levelalgo/quantiles.cuh>
#include <gtest/gtest.h>
#include <nvforest/detail/raft_proto/device_type.hpp>
#include <nvforest/infer_kind.hpp>
#include <nvforest/tree_layout.hpp>
#include <nvforest/treelite_importer.hpp>
#include <test_utils.h>
#include <treelite/tree.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
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
  std::size_t n_rows;
  std::size_t n_cols;
  int n_trees;
  float max_features;
  float max_samples;
  int max_depth;
  int max_leaves;
  bool bootstrap;
  int max_n_bins;
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
  using types = std::tuple<std::size_t,
                           std::size_t,
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
  os << ", max_n_bins = " << ps.max_n_bins << ", min_samples_leaf = " << ps.min_samples_leaf;
  os << ", min_samples_split = " << ps.min_samples_split;
  os << ", min_impurity_decrease = " << ps.min_impurity_decrease
     << ", n_streams = " << ps.n_streams;
  os << ", split_criterion = " << ps.split_criterion << ", seed = " << ps.seed;
  os << ", n_labels = " << ps.n_labels << ", double_precision = " << ps.double_precision;
  return os;
}

template <typename DataT, typename LabelT>
std::shared_ptr<thrust::device_vector<LabelT>> nvForestPredict(
  const raft::handle_t& handle,
  RfTestParams params,
  DataT* X_transpose,
  RandomForestMetaData<DataT, LabelT>* forest)
{
  auto pred      = std::shared_ptr<thrust::device_vector<LabelT>>();
  auto workspace = std::shared_ptr<thrust::device_vector<DataT>>();  // Scratch space
  if constexpr (std::is_integral_v<LabelT>) {
    // For classifiers, allocate extra scratch space to store probabilities from nvForest
    // We will perform argmax to convert probabilities into class outputs.
    pred      = std::make_shared<thrust::device_vector<LabelT>>(params.n_rows);
    workspace = std::make_shared<thrust::device_vector<DataT>>(params.n_rows * params.n_labels);
  } else {
    // For regressors, no need to post-process predictions from nvForest
    static_assert(std::is_same_v<LabelT, DataT>,
                  "LabelT and DataT must be identical for regression task");
    pred      = std::make_shared<thrust::device_vector<LabelT>>(params.n_rows);
    workspace = pred;
  }
  TreeliteModelHandle model;
  build_treelite_forest(&model, forest, params.n_cols);

  auto nvforest_model = nvforest::import_from_treelite_handle(model,
                                                              nvforest::tree_layout::breadth_first,
                                                              128,
                                                              std::is_same_v<DataT, double>,
                                                              raft_proto::device_type::gpu,
                                                              handle.get_device(),
                                                              handle.get_next_usable_stream());
  handle.sync_stream();
  handle.sync_stream_pool();
  delete static_cast<treelite::Model*>(model);

  nvforest_model.predict(handle,
                         workspace->data().get(),
                         X_transpose,
                         params.n_rows,
                         raft_proto::device_type::gpu,
                         raft_proto::device_type::gpu,
                         nvforest::infer_kind::default_kind,
                         1);
  handle.sync_stream();
  handle.sync_stream_pool();

  if constexpr (std::is_integral_v<LabelT>) {
    // Perform argmax to convert probabilities into class outputs
    auto offsets_it =
      cuda::make_transform_iterator(thrust::make_counting_iterator(0),
                                    [=] __device__(int x) -> int { return x * params.n_labels; });
    using kv_type = cub::KeyValuePair<int, DataT>;

    // Compute size of workspace for the segmented reduce operation
    std::size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedReduce::ArgMax(nullptr,
                                       temp_storage_bytes,
                                       workspace->begin(),
                                       thrust::device_pointer_cast<kv_type>(nullptr),
                                       params.n_rows,
                                       offsets_it,
                                       offsets_it + 1);

    // Allocate workspace and perform segmented reduce
    thrust::device_vector<kv_type> workspace2(params.n_rows + temp_storage_bytes / sizeof(kv_type) +
                                              1);
    cub::DeviceSegmentedReduce::ArgMax(thrust::raw_pointer_cast(workspace2.data() + params.n_rows),
                                       temp_storage_bytes,
                                       workspace->begin(),
                                       workspace2.begin(),
                                       params.n_rows,
                                       offsets_it,
                                       offsets_it + 1);
    thrust::transform(workspace2.begin(),
                      workspace2.begin() + params.n_rows,
                      pred->begin(),
                      [] __device__(kv_type x) -> int { return x.key; });
  }

  return pred;
}

template <typename DataT, typename LabelT>
auto nvForestPredictProba(const raft::handle_t& handle,
                          RfTestParams params,
                          DataT* X_transpose,
                          RandomForestMetaData<DataT, LabelT>* forest)
{
  static_assert(std::is_integral_v<LabelT>, "Must be classification");

  std::size_t num_outputs = params.n_labels;
  auto pred = std::make_shared<thrust::device_vector<float>>(params.n_rows * num_outputs);
  TreeliteModelHandle model;
  build_treelite_forest(&model, forest, params.n_cols);

  auto nvforest_model = nvforest::import_from_treelite_handle(model,
                                                              nvforest::tree_layout::breadth_first,
                                                              128,
                                                              std::is_same_v<DataT, double>,
                                                              raft_proto::device_type::gpu,
                                                              handle.get_device(),
                                                              handle.get_next_usable_stream());
  handle.sync_stream();
  handle.sync_stream_pool();
  delete static_cast<treelite::Model*>(model);

  nvforest_model.predict(handle,
                         pred->data().get(),
                         X_transpose,
                         params.n_rows,
                         raft_proto::device_type::gpu,
                         raft_proto::device_type::gpu,
                         nvforest::infer_kind::default_kind,
                         1);
  handle.sync_stream();
  handle.sync_stream_pool();

  return pred;
}
template <typename DataT, typename LabelT>
auto TrainScore(
  const raft::handle_t& handle, RfTestParams params, DataT* X, DataT* X_transpose, LabelT* y)
{
  RF_params rf_params = set_rf_params(params.max_depth,
                                      params.max_leaves,
                                      params.max_features,
                                      params.max_n_bins,
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
    auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(params.n_streams);
    raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
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
        normal.begin(), normal.end(), y_temp.begin(), y.begin(), cuda::std::plus<LabelT>());
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
    auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(params.n_streams);
    raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
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
                raft::ceildiv(int(params.n_rows), params.min_samples_leaf));
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
    bool is_regression = params.split_criterion != GINI and params.split_criterion != ENTROPY;
    if (is_regression) return;

    // Repeat training
    auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(params.n_streams);
    raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
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

  // Compare nvForest against native rf predictions
  // Only for single precision models
  void TestNvForestPredict()
  {
    if constexpr (std::is_same_v<DataT, double>) {
      return;
    } else {
      auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(params.n_streams);
      raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
      auto nvforest_pred = nvForestPredict(handle, params, X_transpose.data().get(), forest.get());

      thrust::host_vector<float> h_nvforest_pred(*nvforest_pred);
      thrust::host_vector<float> h_pred(*predictions);

      thrust::host_vector<float> h_nvforest_pred_prob;
      if constexpr (std::is_integral_v<LabelT>) {
        h_nvforest_pred_prob =
          *nvForestPredictProba(handle, params, X_transpose.data().get(), forest.get());
      }

      float tol = 1e-2;
      for (std::size_t i = 0; i < h_nvforest_pred.size(); i++) {
        // If the output probabilities are very similar for different classes
        // nvForest may output a different class due to numerical differences
        // Skip these cases
        if constexpr (std::is_integral_v<LabelT>) {
          int num_outputs = forest->trees[0]->num_outputs;
          auto min_diff   = MinDifference(&h_nvforest_pred_prob[i * num_outputs], num_outputs);
          if (min_diff < tol) continue;
        }

        EXPECT_LE(abs(h_nvforest_pred[i] - h_pred[i]), tol);
      }
    }
  }

  void TestFeatureImportances()
  {
    // Test feature importances for both regression and classification
    std::vector<DataT> importances(params.n_cols);
    ML::compute_feature_importances(forest.get(), importances.data());

    // Basic checks for feature importances
    EXPECT_EQ(importances.size(), static_cast<size_t>(params.n_cols));

    bool has_splits = false;
    for (int i = 0; i < forest->rf_params.n_trees; i++) {
      if (forest->trees[i]->leaf_counter > 1) {
        has_splits = true;
        break;
      }
    }

    if (!has_splits) {
      for (auto v : importances) {
        EXPECT_EQ(v, 0.0);
      }
      return;
    }

    double sum = 0.0;
    for (auto v : importances) {
      EXPECT_GE(v, 0.0);
      sum += v;
    }

    EXPECT_NEAR(sum, 1.0, 1e-6);
  }

  void Test()
  {
    TestAccuracyImprovement();
    TestDeterminism();
    TestMinImpurity();
    TestTreeSize();
    TestInstanceCounts();
    TestNvForestPredict();
    TestFeatureImportances();
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
    bool is_regression  = params.split_criterion != GINI and params.split_criterion != ENTROPY;
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
std::vector<int> max_n_bins              = {2, 57, 128, 256};
std::vector<int> min_samples_leaf        = {1, 10, 30};
std::vector<int> min_samples_split       = {2, 10};
std::vector<float> min_impurity_decrease = {0.0f, 1.0f, 10.0f};
std::vector<int> n_streams               = {1, 2, 10};
std::vector<CRITERION> split_criterion   = {CRITERION::INVERSE_GAUSSIAN,
                                            CRITERION::GAMMA,
                                            CRITERION::POISSON,
                                            CRITERION::MSE,
                                            CRITERION::GINI,
                                            CRITERION::ENTROPY};
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
                                                                           max_n_bins,
                                                                           min_samples_leaf,
                                                                           min_samples_split,
                                                                           min_impurity_decrease,
                                                                           n_streams,
                                                                           split_criterion,
                                                                           seed,
                                                                           n_labels,
                                                                           double_precision)));

// Unit-weight invariance: classifier fit with `sample_weight = 1.0` matches
// the `nullptr` path byte-for-byte.
TEST(RfTests, ClassifierSampleWeightOnesMatchesNullptr)
{
  constexpr int m = 500;
  constexpr int n = 8;
  thrust::device_vector<float> X(m * n);
  thrust::device_vector<float> w(m, 1.0f);
  raft::random::Rng r(7);
  r.normal(X.data().get(), X.size(), 0.0f, 1.0f, nullptr);
  std::vector<int> h_y(m);
  std::mt19937 host_rng(7);
  for (int i = 0; i < m; ++i)
    h_y[i] = host_rng() & 1;
  thrust::device_vector<int> y = h_y;

  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
  raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
  RF_params rf_params =
    set_rf_params(4, 32, 1.0, 32, 1, 2, 0.0, false, 5, 1.0, 42, CRITERION::GINI, 1, 128);

  auto forest_null = std::make_shared<RandomForestMetaData<float, int>>();
  auto forest_ones = std::make_shared<RandomForestMetaData<float, int>>();
  fit(handle,
      forest_null.get(),
      X.data().get(),
      m,
      n,
      y.data().get(),
      2,
      rf_params,
      rapids_logger::level_enum::warn);
  fit(handle,
      forest_ones.get(),
      X.data().get(),
      m,
      n,
      y.data().get(),
      2,
      rf_params,
      rapids_logger::level_enum::warn,
      nullptr,
      w.data().get());

  ASSERT_EQ(forest_null->trees.size(), forest_ones->trees.size());
  for (size_t t = 0; t < forest_null->trees.size(); ++t) {
    auto& tn = forest_null->trees[t];
    auto& to = forest_ones->trees[t];
    ASSERT_EQ(tn->leaf_counter, to->leaf_counter) << "tree " << t;
    ASSERT_EQ(tn->depth_counter, to->depth_counter) << "tree " << t;
    ASSERT_EQ(tn->sparsetree, to->sparsetree) << "tree " << t;
    ASSERT_EQ(tn->vector_leaf.size(), to->vector_leaf.size()) << "tree " << t;
    for (size_t i = 0; i < tn->vector_leaf.size(); ++i) {
      ASSERT_FLOAT_EQ(tn->vector_leaf[i], to->vector_leaf[i])
        << "tree " << t << " leaf-element " << i;
    }
  }
}

// Unit-weight invariance: regressor fit with `sample_weight = 1.0` matches
// the `nullptr` path byte-for-byte.
TEST(RfTests, RegressorSampleWeightOnesMatchesNullptr)
{
  constexpr int m = 400;
  constexpr int n = 6;
  thrust::device_vector<float> X(m * n);
  thrust::device_vector<float> y(m);
  thrust::device_vector<float> w(m, 1.0f);
  raft::random::Rng r(11);
  r.normal(X.data().get(), X.size(), 0.0f, 1.0f, nullptr);
  r.normal(y.data().get(), y.size(), 0.0f, 1.0f, nullptr);

  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
  raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
  RF_params rf_params =
    set_rf_params(4, 32, 1.0, 32, 1, 2, 0.0, false, 5, 1.0, 42, CRITERION::MSE, 1, 128);

  auto forest_null = std::make_shared<RandomForestMetaData<float, float>>();
  auto forest_ones = std::make_shared<RandomForestMetaData<float, float>>();
  fit(handle,
      forest_null.get(),
      X.data().get(),
      m,
      n,
      y.data().get(),
      rf_params,
      rapids_logger::level_enum::warn);
  fit(handle,
      forest_ones.get(),
      X.data().get(),
      m,
      n,
      y.data().get(),
      rf_params,
      rapids_logger::level_enum::warn,
      nullptr,
      w.data().get());

  ASSERT_EQ(forest_null->trees.size(), forest_ones->trees.size());
  for (size_t t = 0; t < forest_null->trees.size(); ++t) {
    auto& tn = forest_null->trees[t];
    auto& to = forest_ones->trees[t];
    ASSERT_EQ(tn->leaf_counter, to->leaf_counter) << "tree " << t;
    ASSERT_EQ(tn->depth_counter, to->depth_counter) << "tree " << t;
    ASSERT_EQ(tn->sparsetree, to->sparsetree) << "tree " << t;
    ASSERT_EQ(tn->vector_leaf.size(), to->vector_leaf.size()) << "tree " << t;
    for (size_t i = 0; i < tn->vector_leaf.size(); ++i) {
      ASSERT_FLOAT_EQ(tn->vector_leaf[i], to->vector_leaf[i])
        << "tree " << t << " leaf-element " << i;
    }
  }
}

namespace DT {

// Shared two-bin AggregateBin CDF for the four regressor weighted ground-truth
// tests: hist[0] = LEFT prefix (label_sum=8, count=2), hist[1] = total
// (label_sum=20, count=4). W_total=6, W_left=4, W_right=2, nLeft=2, len=4.
static void RegressorWeightedHist(AggregateBin (&hist)[2])
{
  hist[0] = AggregateBin{8.0, 2};
  hist[1] = AggregateBin{20.0, 4};
}

template <typename ObjectiveT, typename DataT>
static DataT RegressorWeightedGainAt(DataT W_total, DataT W_left)
{
  AggregateBin hist[2];
  RegressorWeightedHist(hist);
  ObjectiveT objective(/* nclasses */ 1, /* min_samples_leaf */ 0);
  return objective.GainPerSplit(hist,
                                /* i */ 0,
                                /* n_bins */ 2,
                                /* len */ 4,
                                /* nLeft */ 2,
                                static_cast<double>(W_total),
                                static_cast<double>(W_left));
}

// Weighted MSE proxy = 0.5 * (-L^2/W + L_l^2/W_l + L_r^2/W_r) / W; expected 16/9.
TEST(RfTests, MSEWeightedGainPerSplitGroundTruth)
{
  auto g_d = RegressorWeightedGainAt<MSEObjectiveFunction<double, double, int>, double>(6.0, 4.0);
  ASSERT_NEAR(16.0 / 9.0, g_d, 1e-9);
  auto g_f = RegressorWeightedGainAt<MSEObjectiveFunction<float, float, int>, float>(6.0f, 4.0f);
  ASSERT_NEAR(16.0f / 9.0f, g_f, 1e-5f);
}

// Weighted Poisson half-deviance proxy: -sum L_k * log(L_k/W_k) / W. Expected
// computed inline so a sign-flip in the implementation surfaces here.
TEST(RfTests, PoissonWeightedGainPerSplitGroundTruth)
{
  const double parent   = -20.0 * std::log(20.0 / 6.0);
  const double left     = -8.0 * std::log(8.0 / 4.0);
  const double right    = -12.0 * std::log(12.0 / 2.0);
  const double expected = (parent - (left + right)) / 6.0;
  auto g_d =
    RegressorWeightedGainAt<PoissonObjectiveFunction<double, double, int>, double>(6.0, 4.0);
  ASSERT_NEAR(expected, g_d, 1e-9);
  auto g_f =
    RegressorWeightedGainAt<PoissonObjectiveFunction<float, float, int>, float>(6.0f, 4.0f);
  ASSERT_NEAR(static_cast<float>(expected), g_f, 1e-5f);
}

// Weighted Gamma half-deviance proxy: sum W_k * log(L_k/W_k) / W (parent term
// scaled by W, not -L). Expected computed inline.
TEST(RfTests, GammaWeightedGainPerSplitGroundTruth)
{
  const double parent   = 6.0 * std::log(20.0 / 6.0);
  const double left     = 4.0 * std::log(8.0 / 4.0);
  const double right    = 2.0 * std::log(12.0 / 2.0);
  const double expected = (parent - (left + right)) / 6.0;
  auto g_d = RegressorWeightedGainAt<GammaObjectiveFunction<double, double, int>, double>(6.0, 4.0);
  ASSERT_NEAR(expected, g_d, 1e-9);
  auto g_f = RegressorWeightedGainAt<GammaObjectiveFunction<float, float, int>, float>(6.0f, 4.0f);
  ASSERT_NEAR(static_cast<float>(expected), g_f, 1e-5f);
}

// Weighted Inverse Gaussian half-deviance proxy: -W_k^2 / L_k summed, scaled
// by 1/(2*W). Expected = 2/45 from the shared histogram.
TEST(RfTests, InverseGaussianWeightedGainPerSplitGroundTruth)
{
  const double expected = 2.0 / 45.0;
  auto g_d = RegressorWeightedGainAt<InverseGaussianObjectiveFunction<double, double, int>, double>(
    6.0, 4.0);
  ASSERT_NEAR(expected, g_d, 1e-9);
  auto g_f =
    RegressorWeightedGainAt<InverseGaussianObjectiveFunction<float, float, int>, float>(6.0f, 4.0f);
  ASSERT_NEAR(static_cast<float>(expected), g_f, 1e-5f);
}

// GainPerSplit honors its integer nLeft argument for the min_samples_leaf
// gate, independent of the weighted W_left. Asymmetric class counts so the
// finite gain is provably nonzero (no per-class cancellation).
TEST(RfTests, ClassifierGainPerSplitNLeftGateRespectsIntegerCount)
{
  const int n_bins                = 2;
  const int nclass                = 2;
  CountBin shist[n_bins * nclass] = {
    CountBin{1.0},
    CountBin{3.0},  // class 0: bin-0 prefix, bin-1 CDF total
    CountBin{2.0},
    CountBin{3.0},  // class 1
  };
  GiniObjectiveFunction<float, int, int> gini_strict(nclass, /* min_samples_leaf */ 3);
  // Same weighted (W_total, W_left) both calls; only integer nLeft differs.
  // len=10 keeps nRight >= leaf so the assertion isolates the nLeft side.
  float gain_fail = gini_strict.GainPerSplit(shist,
                                             /* i */ 0,
                                             n_bins,
                                             /* len */ 10,
                                             /* nLeft */ 2,
                                             /* W_total */ 6.0,
                                             /* W_left */ 3.0);
  float gain_ok   = gini_strict.GainPerSplit(shist,
                                           /* i */ 0,
                                           n_bins,
                                           /* len */ 10,
                                           /* nLeft */ 5,
                                           /* W_total */ 6.0,
                                           /* W_left */ 3.0);
  ASSERT_EQ(gain_fail, -std::numeric_limits<float>::max());
  ASSERT_GT(gain_ok, 0.0f);  // non-trivial positive gain proves no cancellation
}

// SetLeafVector is __device__-only; one-thread kernel wraps it for host gtests.
template <typename ObjectiveT, typename BinT, typename DataT>
__global__ void RunSetLeafVectorKernel(BinT const* shist,
                                       int nclasses,
                                       DataT* out,
                                       double weighted_total)
{
  ObjectiveT::SetLeafVector(shist, nclasses, out, weighted_total);
}

template <typename ObjectiveT, typename BinT, typename DataT>
static void RunSetLeafVectorOnDevice(std::vector<BinT> const& shist_host,
                                     std::vector<DataT>& out_host,
                                     double weighted_total)
{
  thrust::device_vector<BinT> shist_dev = shist_host;
  thrust::device_vector<DataT> out_dev  = out_host;
  RunSetLeafVectorKernel<ObjectiveT, BinT, DataT><<<1, 1>>>(shist_dev.data().get(),
                                                            static_cast<int>(out_host.size()),
                                                            out_dev.data().get(),
                                                            weighted_total);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  thrust::copy(out_dev.begin(), out_dev.end(), out_host.begin());
}

// All-zero-weight leaf: classifier SetLeafVector emits a uniform distribution
// instead of NaN (0/0). Covers Gini and Entropy.
TEST(RfTests, ClassifierSetLeafVectorAllZeroWeightYieldsUniform)
{
  const int nclasses          = 3;
  std::vector<CountBin> shist = {CountBin{0.0}, CountBin{0.0}, CountBin{0.0}};
  std::vector<float> out(nclasses, -1.0f);
  RunSetLeafVectorOnDevice<GiniObjectiveFunction<float, int, int>, CountBin, float>(
    shist, out, /* weighted_total */ 0.0);
  for (int i = 0; i < nclasses; ++i)
    ASSERT_FLOAT_EQ(out[i], 1.0f / static_cast<float>(nclasses));
  std::fill(out.begin(), out.end(), -1.0f);
  RunSetLeafVectorOnDevice<EntropyObjectiveFunction<float, int, int>, CountBin, float>(
    shist, out, /* weighted_total */ 0.0);
  for (int i = 0; i < nclasses; ++i)
    ASSERT_FLOAT_EQ(out[i], 1.0f / static_cast<float>(nclasses));
}

// All-zero-weight leaf: regressor SetLeafVector emits 0 instead of NaN. Covers
// all four regressor objectives.
TEST(RfTests, RegressorSetLeafVectorAllZeroWeightYieldsZero)
{
  std::vector<AggregateBin> shist = {AggregateBin{0.0, 0}};
  std::vector<float> out(1, -1.0f);
  RunSetLeafVectorOnDevice<MSEObjectiveFunction<float, float, int>, AggregateBin, float>(
    shist, out, /* weighted_total */ 0.0);
  ASSERT_FLOAT_EQ(out[0], 0.0f);
  out[0] = -1.0f;
  RunSetLeafVectorOnDevice<PoissonObjectiveFunction<float, float, int>, AggregateBin, float>(
    shist, out, /* weighted_total */ 0.0);
  ASSERT_FLOAT_EQ(out[0], 0.0f);
  out[0] = -1.0f;
  RunSetLeafVectorOnDevice<GammaObjectiveFunction<float, float, int>, AggregateBin, float>(
    shist, out, /* weighted_total */ 0.0);
  ASSERT_FLOAT_EQ(out[0], 0.0f);
  out[0]            = -1.0f;
  using IGObjective = InverseGaussianObjectiveFunction<float, float, int>;
  RunSetLeafVectorOnDevice<IGObjective, AggregateBin, float>(shist, out, /* weighted_total */ 0.0);
  ASSERT_FLOAT_EQ(out[0], 0.0f);
}

}  // namespace DT

// Default args (mode=NONE, array=nullptr) must produce byte-identical
// trees vs the pre-class_weight_mode signature.
TEST(RfTests, ClassWeightModeNoneIsByteIdentical)
{
  constexpr int m = 400;
  constexpr int n = 6;
  thrust::device_vector<float> X(m * n);
  raft::random::Rng r(11);
  r.normal(X.data().get(), X.size(), 0.0f, 1.0f, nullptr);
  std::vector<int> h_y(m);
  std::mt19937 host_rng(11);
  for (int i = 0; i < m; ++i)
    h_y[i] = host_rng() & 1;
  thrust::device_vector<int> y = h_y;

  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
  raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
  RF_params rf_params =
    set_rf_params(4, 32, 1.0, 32, 1, 2, 0.0, true, 5, 1.0, 17, CRITERION::GINI, 1, 128);

  auto forest_default       = std::make_shared<RandomForestMetaData<float, int>>();
  auto forest_explicit_none = std::make_shared<RandomForestMetaData<float, int>>();
  fit(handle,
      forest_default.get(),
      X.data().get(),
      m,
      n,
      y.data().get(),
      2,
      rf_params,
      rapids_logger::level_enum::warn);
  fit(handle,
      forest_explicit_none.get(),
      X.data().get(),
      m,
      n,
      y.data().get(),
      2,
      rf_params,
      rapids_logger::level_enum::warn,
      /*bootstrap_masks=*/nullptr,
      /*sample_weight=*/nullptr,
      /*class_weight_mode=*/static_cast<int>(ClassWeightMode::NONE),
      /*class_weight_array=*/nullptr);

  ASSERT_EQ(forest_default->trees.size(), forest_explicit_none->trees.size());
  for (size_t t = 0; t < forest_default->trees.size(); ++t) {
    auto& td = forest_default->trees[t];
    auto& te = forest_explicit_none->trees[t];
    ASSERT_EQ(td->leaf_counter, te->leaf_counter) << "tree " << t;
    ASSERT_EQ(td->depth_counter, te->depth_counter) << "tree " << t;
    ASSERT_EQ(td->sparsetree, te->sparsetree) << "tree " << t;
    ASSERT_EQ(td->vector_leaf.size(), te->vector_leaf.size()) << "tree " << t;
    for (size_t i = 0; i < td->vector_leaf.size(); ++i) {
      ASSERT_FLOAT_EQ(td->vector_leaf[i], te->vector_leaf[i])
        << "tree " << t << " leaf-element " << i;
    }
  }
}

// Contract: (mode == NONE) iff (array == nullptr). Both halves should
// raise a typed exception when violated.
TEST(RfTests, ClassWeightModeContractViolationAsserts)
{
  constexpr int m = 64;
  constexpr int n = 4;
  thrust::device_vector<float> X(m * n);
  raft::random::Rng r(7);
  r.normal(X.data().get(), X.size(), 0.0f, 1.0f, nullptr);
  std::vector<int> h_y(m, 0);
  for (int i = 0; i < m; ++i)
    h_y[i] = i & 1;
  thrust::device_vector<int> y = h_y;
  thrust::device_vector<double> cw(2, 1.0);

  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
  raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
  RF_params rf_params =
    set_rf_params(3, 16, 1.0, 16, 1, 2, 0.0, true, 2, 1.0, 7, CRITERION::GINI, 1, 64);

  auto forest_a = std::make_shared<RandomForestMetaData<float, int>>();
  EXPECT_THROW(fit(handle,
                   forest_a.get(),
                   X.data().get(),
                   m,
                   n,
                   y.data().get(),
                   2,
                   rf_params,
                   rapids_logger::level_enum::warn,
                   nullptr,
                   nullptr,
                   static_cast<int>(ClassWeightMode::BALANCED_SUBSAMPLE),
                   /*class_weight_array=*/nullptr),
               raft::exception);

  auto forest_b = std::make_shared<RandomForestMetaData<float, int>>();
  EXPECT_THROW(fit(handle,
                   forest_b.get(),
                   X.data().get(),
                   m,
                   n,
                   y.data().get(),
                   2,
                   rf_params,
                   rapids_logger::level_enum::warn,
                   nullptr,
                   nullptr,
                   static_cast<int>(ClassWeightMode::NONE),
                   cw.data().get()),
               raft::exception);
}

// BALANCED_SUBSAMPLE on a 90/10 fixture must produce a different forest
// than the unweighted baseline (engagement check, not sklearn parity).
TEST(RfTests, ClassWeightModeBalancedSubsampleChangesForest)
{
  constexpr int m = 600;
  constexpr int n = 6;
  thrust::device_vector<float> X(m * n);
  raft::random::Rng r(23);
  r.normal(X.data().get(), X.size(), 0.0f, 1.0f, nullptr);
  std::vector<int> h_y(m);
  // 90/10 imbalance.
  for (int i = 0; i < m; ++i)
    h_y[i] = (i % 10 == 0) ? 1 : 0;
  thrust::device_vector<int> y = h_y;
  // Full-y balanced reciprocals: minority class (count=60) gets 5x, majority
  // (count=540) gets ~0.556. Passed through the ABI as the diagnostic array.
  thrust::device_vector<double> class_weights(2);
  class_weights[0] = static_cast<double>(m) / (2.0 * 540.0);
  class_weights[1] = static_cast<double>(m) / (2.0 * 60.0);

  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
  raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
  RF_params rf_params =
    set_rf_params(5, 64, 1.0, 32, 1, 2, 0.0, true, 8, 1.0, 23, CRITERION::GINI, 1, 128);

  auto forest_baseline     = std::make_shared<RandomForestMetaData<float, int>>();
  auto forest_balanced_sub = std::make_shared<RandomForestMetaData<float, int>>();
  fit(handle,
      forest_baseline.get(),
      X.data().get(),
      m,
      n,
      y.data().get(),
      2,
      rf_params,
      rapids_logger::level_enum::warn);
  fit(handle,
      forest_balanced_sub.get(),
      X.data().get(),
      m,
      n,
      y.data().get(),
      2,
      rf_params,
      rapids_logger::level_enum::warn,
      /*bootstrap_masks=*/nullptr,
      /*sample_weight=*/nullptr,
      /*class_weight_mode=*/static_cast<int>(ClassWeightMode::BALANCED_SUBSAMPLE),
      /*class_weight_array=*/class_weights.data().get());

  // Either the leaf counts or the tree structure should change; a forest
  // that ignores the per-tree reweighting would be byte-identical here.
  bool any_differ = false;
  for (size_t t = 0; t < forest_baseline->trees.size(); ++t) {
    auto& tb  = forest_baseline->trees[t];
    auto& tbs = forest_balanced_sub->trees[t];
    if (tb->leaf_counter != tbs->leaf_counter || tb->sparsetree.size() != tbs->sparsetree.size()) {
      any_differ = true;
      break;
    }
    for (size_t i = 0; i < tb->vector_leaf.size(); ++i) {
      if (tb->vector_leaf[i] != tbs->vector_leaf[i]) {
        any_differ = true;
        break;
      }
    }
    if (any_differ) break;
  }
  ASSERT_TRUE(any_differ) << "BALANCED_SUBSAMPLE produced identical forest to unweighted; the "
                             "per-tree compute path is not engaging.";
}

TEST(RfTests, IntegerOverflow)
{
  std::size_t m = 1000000;
  std::size_t n = 2150;
  EXPECT_GE(m * n, 1ull << 31);
  thrust::device_vector<float> X(m * n);
  thrust::device_vector<float> y(m);
  raft::random::Rng r(4);
  r.normal(X.data().get(), X.size(), 0.0f, 2.0f, nullptr);
  r.normal(y.data().get(), y.size(), 0.0f, 2.0f, nullptr);
  auto forest      = std::make_shared<RandomForestMetaData<float, float>>();
  auto forest_ptr  = forest.get();
  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(4);
  raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
  RF_params rf_params =
    set_rf_params(3, 100, 1.0, 256, 1, 2, 0.0, false, 1, 1.0, 0, CRITERION::MSE, 4, 128);
  fit(handle, forest_ptr, X.data().get(), m, n, y.data().get(), rf_params);

  // Check we have actually learned something
  EXPECT_GT(forest->trees[0]->leaf_counter, 1);

  // See if nvForest overflows
  thrust::device_vector<float> pred(m);
  TreeliteModelHandle model;
  build_treelite_forest(&model, forest_ptr, n);

  auto nvforest_model = nvforest::import_from_treelite_handle(model,
                                                              nvforest::tree_layout::breadth_first,
                                                              128,
                                                              false,
                                                              raft_proto::device_type::gpu,
                                                              handle.get_device(),
                                                              handle.get_next_usable_stream());
  handle.sync_stream();
  handle.sync_stream_pool();
  delete static_cast<treelite::Model*>(model);

  nvforest_model.predict(handle,
                         pred.data().get(),
                         X.data().get(),
                         m,
                         raft_proto::device_type::gpu,
                         raft_proto::device_type::gpu,
                         nvforest::infer_kind::default_kind,
                         1);
  handle.sync_stream();
  handle.sync_stream_pool();
}

//-------------------------------------------------------------------------------------------------------------------------------------
struct QuantileTestParameters {
  int n_rows;
  int max_n_bins;
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
    thrust::host_vector<T> h_quantiles(params.max_n_bins);
    raft::random::Rng r(8);
    r.normal(data.data().get(), data.size(), T(0.0), T(2.0), nullptr);
    auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
    raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);

    // computing the quantiles
    auto [quantiles, quantiles_array, n_bins_array] =
      DT::computeQuantiles(handle, data.data().get(), params.max_n_bins, params.n_rows, 1);

    raft::update_host(
      h_quantiles.data(), quantiles.quantiles_array, params.max_n_bins, handle.get_stream());

    int n_unique_bins;
    raft::copy(&n_unique_bins, quantiles.n_bins_array, 1, handle.get_stream());
    if (n_unique_bins < params.max_n_bins) {
      return;  // almost impossible that this happens, skip if so
    }

    h_data = data;
    for (std::size_t i = 0; i < h_data.size(); ++i) {
      auto d = h_data[i];
      // golden lower bound from thrust
      auto golden_lb =
        thrust::lower_bound(
          thrust::seq, h_quantiles.data(), h_quantiles.data() + params.max_n_bins, d) -
        h_quantiles.data();
      // lower bound from custom lower_bound impl
      auto lb = DT::lower_bound(h_quantiles.data(), params.max_n_bins, d);
      if (golden_lb == params.max_n_bins) {
        ASSERT_EQ(lb, params.max_n_bins - 1)
          << "custom lower_bound should clamp values above the last quantile to the last bin"
          << std::endl;
        continue;
      }
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

    raft::random::Rng r(8);
    r.normal(data.data().get(), data.size(), T(0.0), T(2.0), nullptr);
    auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
    raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);

    // computing the quantiles
    auto [quantiles, quantiles_array, n_bins_array] =
      DT::computeQuantiles(handle, data.data().get(), params.max_n_bins, params.n_rows, 1);

    int n_unique_bins;
    raft::copy(&n_unique_bins, quantiles.n_bins_array, 1, handle.get_stream());
    if (n_unique_bins < params.max_n_bins) { ASSERT_GT(n_unique_bins, 1); }
    ASSERT_LE(n_unique_bins, params.max_n_bins);

    thrust::host_vector<T> h_quantiles(params.max_n_bins);
    raft::update_host(
      h_quantiles.data(), quantiles.quantiles_array, params.max_n_bins, handle.get_stream());
    handle.sync_stream();
    for (int b = 1; b < n_unique_bins; b++) {
      ASSERT_LT(h_quantiles[b - 1], h_quantiles[b]);
    }
  }
};

// test to make sure that the quantiles and offsets calculated implement
// variable binning properly for categorical data, with unique values less than the `max_n_bins`
template <typename T>
class RFQuantileVariableBinsTest : public ::testing::TestWithParam<QuantileTestParameters> {
 public:
  void SetUp() override
  {
    auto params = ::testing::TestWithParam<QuantileTestParameters>::GetParam();
    srand(params.seed);

    auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
    raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
    thrust::device_vector<T> data(params.n_rows);

    // n_uniques guaranteed to be non-zero and smaller than `max_n_bins`
    int n_uniques;
    while ((n_uniques = rand() % params.max_n_bins) == 0) {}

    // populating random elements in data in [0, n_uniques)
    thrust::counting_iterator<float> first(0);
    thrust::copy(first, first + data.size(), data.begin());
    thrust::transform(data.begin(), data.end(), data.begin(), [=] __device__(auto& x) {
      x = T(int(x) % n_uniques);
      return x;
    });
    thrust::shuffle(data.begin(), data.end(), thrust::default_random_engine(n_uniques));

    // Use full-sample mode to verify duplicate compaction exactly.
    auto [quantiles, quantiles_array, n_bins_array] = DT::computeQuantiles(
      handle, data.data().get(), params.max_n_bins, params.n_rows, 1, params.n_rows, params.seed);
    int n_uniques_obtained;
    raft::copy(&n_uniques_obtained, n_bins_array->data(), 1, handle.get_stream());

    ASSERT_EQ(n_uniques_obtained, n_uniques) << "No. of unique bins is supposed to be " << n_uniques
                                             << ", but got " << n_uniques_obtained << std::endl;

    thrust::device_vector<int> histogram(n_uniques);
    thrust::host_vector<int> h_histogram(n_uniques);
    auto d_quantiles = quantiles.quantiles_array;
    auto d_histogram = histogram.data().get();

    // creating a cumulative histogram from data based on the quantiles
    // where histogram[i] has number of elements that are less-than-equal quantiles[i]
    thrust::for_each(data.begin(), data.end(), [=] __device__(T x) {
      for (int j = 0; j < n_uniques; j++) {
        if (x <= d_quantiles[j]) {
          atomicAdd(&d_histogram[j], 1);
          break;
        }
      }
    });

    // since the elements are randomly and equally distributed, we verify the calculated histogram
    h_histogram           = histogram;
    int max_items_per_bin = raft::ceildiv(params.n_rows, n_uniques);
    int min_items_per_bin = max_items_per_bin - 1;
    int total_items       = 0;
    for (int b = 0; b < n_uniques; b++) {
      ASSERT_TRUE(h_histogram[b] == max_items_per_bin or h_histogram[b] == min_items_per_bin)
        << "No. samples in bin[" << b << "] = " << h_histogram[b] << " Expected "
        << max_items_per_bin << " or " << min_items_per_bin << std::endl;
      total_items += h_histogram[b];
    }

    // recalculate the items for checking proper counting
    ASSERT_EQ(params.n_rows, total_items)
      << "Some samples from dataset are either missed of double counted in quantile bins"
      << std::endl;
  }
};

template <typename T>
class RFSampledQuantileExactFallbackTest : public ::testing::TestWithParam<QuantileTestParameters> {
 public:
  void SetUp() override
  {
    auto params = ::testing::TestWithParam<QuantileTestParameters>::GetParam();

    auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
    raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
    thrust::device_vector<T> data(params.n_rows);
    thrust::sequence(data.begin(), data.end(), T(0));

    auto [sampled_quantiles, sampled_quantiles_array, sampled_n_bins_array] = DT::computeQuantiles(
      handle, data.data().get(), params.max_n_bins, params.n_rows, 1, params.n_rows, params.seed);

    int sampled_n_bins;
    raft::copy(&sampled_n_bins, sampled_n_bins_array->data(), 1, handle.get_stream());
    handle.sync_stream();

    ASSERT_EQ(sampled_n_bins, params.max_n_bins);

    thrust::host_vector<T> h_sampled(params.max_n_bins);
    raft::update_host(
      h_sampled.data(), sampled_quantiles.quantiles_array, params.max_n_bins, handle.get_stream());
    handle.sync_stream();

    double bin_width = static_cast<double>(params.n_rows) / params.max_n_bins;
    for (int bin = 0; bin < sampled_n_bins; ++bin) {
      int idx = int(round((bin + 1) * bin_width)) - 1;
      idx     = std::min(std::max(0, idx), params.n_rows - 1);
      ASSERT_EQ(h_sampled[bin], T(idx));
    }
  }
};

template <typename T>
class RFSampledQuantileDeterminismTest : public ::testing::TestWithParam<QuantileTestParameters> {
 public:
  void SetUp() override
  {
    auto params = ::testing::TestWithParam<QuantileTestParameters>::GetParam();

    auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
    raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
    thrust::device_vector<T> data(params.n_rows);
    raft::random::Rng r(params.seed);
    r.normal(data.data().get(), data.size(), T(0.0), T(2.0), nullptr);

    auto [quantiles_a, quantiles_array_a, n_bins_array_a] = DT::computeQuantiles(
      handle, data.data().get(), params.max_n_bins, params.n_rows, 1, 4, params.seed);
    auto [quantiles_b, quantiles_array_b, n_bins_array_b] = DT::computeQuantiles(
      handle, data.data().get(), params.max_n_bins, params.n_rows, 1, 4, params.seed);

    int n_bins_a;
    int n_bins_b;
    raft::copy(&n_bins_a, n_bins_array_a->data(), 1, handle.get_stream());
    raft::copy(&n_bins_b, n_bins_array_b->data(), 1, handle.get_stream());
    handle.sync_stream();

    ASSERT_EQ(n_bins_a, n_bins_b);
    ASSERT_GT(n_bins_a, 1);
    ASSERT_LE(n_bins_a, params.max_n_bins);

    thrust::host_vector<T> h_quantiles_a(params.max_n_bins);
    thrust::host_vector<T> h_quantiles_b(params.max_n_bins);
    raft::update_host(
      h_quantiles_a.data(), quantiles_a.quantiles_array, params.max_n_bins, handle.get_stream());
    raft::update_host(
      h_quantiles_b.data(), quantiles_b.quantiles_array, params.max_n_bins, handle.get_stream());
    handle.sync_stream();

    for (int i = 0; i < n_bins_a; ++i) {
      ASSERT_EQ(h_quantiles_a[i], h_quantiles_b[i]);
      if (i > 0) { ASSERT_LT(h_quantiles_a[i - 1], h_quantiles_a[i]); }
    }
  }
};

template <typename T>
class RFSampledQuantileRankErrorTest : public ::testing::TestWithParam<QuantileTestParameters> {
 public:
  void SetUp() override
  {
    auto params = ::testing::TestWithParam<QuantileTestParameters>::GetParam();

    auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
    raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
    thrust::device_vector<T> data(params.n_rows);
    thrust::sequence(data.begin(), data.end(), T(0));

    auto [quantiles, quantiles_array, n_bins_array] = DT::computeQuantiles(
      handle, data.data().get(), params.max_n_bins, params.n_rows, 1, 4, params.seed);

    int n_bins;
    raft::copy(&n_bins, n_bins_array->data(), 1, handle.get_stream());
    handle.sync_stream();

    ASSERT_EQ(n_bins, params.max_n_bins);

    thrust::host_vector<T> h_quantiles(params.max_n_bins);
    raft::update_host(
      h_quantiles.data(), quantiles.quantiles_array, params.max_n_bins, handle.get_stream());
    handle.sync_stream();

    double total_abs_rank_error = 0.0;
    double max_abs_rank_error   = 0.0;
    for (int bin = 0; bin < n_bins; ++bin) {
      double expected_rank = static_cast<double>(bin + 1) / params.max_n_bins;
      double actual_rank   = (static_cast<double>(h_quantiles[bin]) + 1.0) / params.n_rows;
      double rank_error    = std::abs(actual_rank - expected_rank);
      total_abs_rank_error += rank_error;
      max_abs_rank_error = std::max(max_abs_rank_error, rank_error);
    }

    double mean_abs_rank_error = total_abs_rank_error / n_bins;
    double sample_count        = static_cast<double>(params.max_n_bins * 4);
    double rank_error_scale    = 1.0 / std::sqrt(sample_count);
    EXPECT_LT(mean_abs_rank_error, rank_error_scale);
    EXPECT_LT(max_abs_rank_error, 3.0 * rank_error_scale);
  }
};

const std::vector<QuantileTestParameters> inputs = {{1000, 16, 6078587519764079670LLU},
                                                    {1130, 32, 4884670006177930266LLU},
                                                    {1752, 67, 9175325892580481371LLU},
                                                    {2307, 99, 9507819643927052255LLU},
                                                    {5000, 128, 9507819643927052255LLU}};

const std::vector<QuantileTestParameters> rank_error_inputs = {
  {10000, 128, 9507819643927052255LLU}};

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

// double type quantile bins lower bounds test
typedef RFQuantileBinsLowerBoundTest<double> RFQuantileBinsLowerBoundTestD;
TEST_P(RFQuantileBinsLowerBoundTestD, test) {}
INSTANTIATE_TEST_CASE_P(RfTests, RFQuantileBinsLowerBoundTestD, ::testing::ValuesIn(inputs));

// float type quantile variable binning test
typedef RFQuantileVariableBinsTest<float> RFQuantileVariableBinsTestF;
TEST_P(RFQuantileVariableBinsTestF, test) {}
INSTANTIATE_TEST_CASE_P(RfTests, RFQuantileVariableBinsTestF, ::testing::ValuesIn(inputs));

// double type quantile variable binning test
typedef RFQuantileVariableBinsTest<double> RFQuantileVariableBinsTestD;
TEST_P(RFQuantileVariableBinsTestD, test) {}
INSTANTIATE_TEST_CASE_P(RfTests, RFQuantileVariableBinsTestD, ::testing::ValuesIn(inputs));

typedef RFSampledQuantileExactFallbackTest<float> RFSampledQuantileExactFallbackTestF;
TEST_P(RFSampledQuantileExactFallbackTestF, test) {}
INSTANTIATE_TEST_CASE_P(RfTests, RFSampledQuantileExactFallbackTestF, ::testing::ValuesIn(inputs));

typedef RFSampledQuantileExactFallbackTest<double> RFSampledQuantileExactFallbackTestD;
TEST_P(RFSampledQuantileExactFallbackTestD, test) {}
INSTANTIATE_TEST_CASE_P(RfTests, RFSampledQuantileExactFallbackTestD, ::testing::ValuesIn(inputs));

typedef RFSampledQuantileDeterminismTest<float> RFSampledQuantileDeterminismTestF;
TEST_P(RFSampledQuantileDeterminismTestF, test) {}
INSTANTIATE_TEST_CASE_P(RfTests, RFSampledQuantileDeterminismTestF, ::testing::ValuesIn(inputs));

typedef RFSampledQuantileDeterminismTest<double> RFSampledQuantileDeterminismTestD;
TEST_P(RFSampledQuantileDeterminismTestD, test) {}
INSTANTIATE_TEST_CASE_P(RfTests, RFSampledQuantileDeterminismTestD, ::testing::ValuesIn(inputs));

typedef RFSampledQuantileRankErrorTest<float> RFSampledQuantileRankErrorTestF;
TEST_P(RFSampledQuantileRankErrorTestF, test) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        RFSampledQuantileRankErrorTestF,
                        ::testing::ValuesIn(rank_error_inputs));

typedef RFSampledQuantileRankErrorTest<double> RFSampledQuantileRankErrorTestD;
TEST_P(RFSampledQuantileRankErrorTestD, test) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        RFSampledQuantileRankErrorTestD,
                        ::testing::ValuesIn(rank_error_inputs));

//------------------------------------------------------------------------------------------------------

TEST(RfTest, TextDump)
{
  RF_params rf_params = set_rf_params(2, 2, 1.0, 2, 1, 2, 0.0, false, 1, 1.0, 0, GINI, 1, 128);
  auto forest         = std::make_shared<RandomForestMetaData<float, int>>();

  std::vector<float> X_host      = {1, 2, 3, 6, 7, 8};
  thrust::device_vector<float> X = X_host;
  std::vector<int> y_host        = {0, 0, 1, 1, 1, 0};
  thrust::device_vector<int> y   = y_host;

  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
  raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
  auto forest_ptr = forest.get();
  fit(handle, forest_ptr, X.data().get(), y.size(), 1, y.data().get(), 2, rf_params);

  std::string expected_start_text = R"(Forest has 1 trees, max_depth 2, and max_leaves 2
Tree #0
 Decision Tree depth --> 1 and n_leaves --> 2
 Tree Fitting - Overall time -->)";

  std::string expected_end_text = R"(└(colid: 0, quesval: 3, best_metric_val: 0.0555556)
    ├(leaf, prediction: [0.666667, 0.333333], best_metric_val: 0)
    └(leaf, prediction: [0.333333, 0.666667], best_metric_val: 0))";

  EXPECT_TRUE(get_rf_detailed_text(forest_ptr).find(expected_start_text) != std::string::npos);
  EXPECT_TRUE(get_rf_detailed_text(forest_ptr).find(expected_end_text) != std::string::npos);
  std::string expected_json = R"([
{"nodeid": 0, "split_feature": 0, "split_threshold": 3, "gain": 0.055555582, "instance_count": 6, "yes": 1, "no": 2, "children": [
  {"nodeid": 1, "leaf_value": [0.666666687, 0.333333343], "instance_count": 3},
  {"nodeid": 2, "leaf_value": [0.333333343, 0.666666687], "instance_count": 3}
]}
])";

  EXPECT_EQ(get_rf_json(forest_ptr), expected_json);
}

//-------------------------------------------------------------------------------------------------------------------------------------
namespace DT {

struct ObjectiveTestParameters {
  uint64_t seed;
  int n_rows;
  int max_n_bins;
  int n_classes;
  int min_samples_leaf;
  double tolerance;
};

template <typename ObjectiveT>
class ObjectiveTest : public ::testing::TestWithParam<ObjectiveTestParameters> {
  typedef typename ObjectiveT::DataT DataT;
  typedef typename ObjectiveT::LabelT LabelT;
  typedef typename ObjectiveT::IdxT IdxT;
  typedef typename ObjectiveT::BinT BinT;

  ObjectiveTestParameters params;

 public:
  auto RandUnder(int const end = 10000) { return rand() % end; }

  auto GenRandomData()
  {
    std::default_random_engine rng;
    std::vector<DataT> data(params.n_rows);
    if constexpr (std::is_same<BinT, CountBin>::value)  // classification case
    {
      for (auto& d : data) {
        d = RandUnder(params.n_classes);
      }
    } else {
      std::normal_distribution<DataT> normal(1.0, 2.0);
      for (auto& d : data) {
        auto rand_element{DataT(0)};
        while (1) {
          rand_element = normal(rng);
          if (rand_element > 0) break;  // only positive random numbers
        }
        d = rand_element;
      }
    }
    return data;
  }

  auto GenHist(std::vector<DataT> data)
  {
    std::vector<BinT> cdf_hist, pdf_hist;

    for (auto c = 0; c < params.n_classes; ++c) {
      for (auto b = 0; b < params.max_n_bins; ++b) {
        IdxT bin_width  = raft::ceildiv(params.n_rows, params.max_n_bins);
        auto data_begin = data.begin() + b * bin_width;
        auto data_end   = data_begin + bin_width;
        if constexpr (std::is_same<BinT, CountBin>::value) {  // classification case
          auto count{IdxT(0)};
          std::for_each(data_begin, data_end, [&](auto d) {
            if (d == c) ++count;
          });
          pdf_hist.emplace_back(count);
        } else {  // regression case
          auto label_sum{DataT(0)};
          label_sum = std::accumulate(data_begin, data_end, DataT(0));
          pdf_hist.emplace_back(label_sum, bin_width);
        }

        auto cumulative = b > 0 ? cdf_hist.back() : BinT();
        cdf_hist.emplace_back(pdf_hist.empty() ? BinT() : pdf_hist.back());
        cdf_hist.back() += cumulative;
      }
    }

    return std::make_pair(cdf_hist, pdf_hist);
  }

  auto MSE(std::vector<DataT> const& data)  //  1/n * 1/2 * sum((y - y_pred) * (y - y_pred))
  {
    DataT sum        = std::accumulate(data.begin(), data.end(), DataT(0));
    DataT const mean = sum / data.size();
    auto mse{DataT(0.0)};  // mse: mean squared error

    std::for_each(data.begin(), data.end(), [&](auto d) {
      mse += (d - mean) * (d - mean);  // unit deviance
    });

    mse /= 2 * data.size();
    return std::make_tuple(mse, sum, DataT(data.size()));
  }

  auto MSEGroundTruthGain(std::vector<DataT> const& data, std::size_t split_bin_index)
  {
    auto bin_width = raft::ceildiv(params.n_rows, params.max_n_bins);
    std::vector<DataT> left_data(data.begin(), data.begin() + (split_bin_index + 1) * bin_width);
    std::vector<DataT> right_data(data.begin() + (split_bin_index + 1) * bin_width, data.end());

    auto [parent_mse, label_sum, n]            = MSE(data);
    auto [left_mse, label_sum_left, n_left]    = MSE(left_data);
    auto [right_mse, label_sum_right, n_right] = MSE(right_data);

    auto gain =
      parent_mse - ((n_left / n) * left_mse +  // the minimizing objective function is half deviance
                    (n_right / n) * right_mse);  // gain in long form without proxy

    // edge cases
    if (n_left < params.min_samples_leaf or n_right < params.min_samples_leaf)
      return -std::numeric_limits<DataT>::max();
    else
      return gain;
  }

  auto InverseGaussianHalfDeviance(
    std::vector<DataT> const&
      data)  //  1/n * 2 * sum((y - y_pred) * (y - y_pred)/(y * (y_pred) * (y_pred)))
  {
    DataT sum        = std::accumulate(data.begin(), data.end(), DataT(0));
    DataT const mean = sum / data.size();
    auto ighd{DataT(0.0)};  // ighd: inverse gaussian half deviance

    std::for_each(data.begin(), data.end(), [&](auto d) {
      ighd += (d - mean) * (d - mean) / (d * mean * mean);  // unit deviance
    });

    ighd /= 2 * data.size();
    return std::make_tuple(ighd, sum, DataT(data.size()));
  }

  auto InverseGaussianGroundTruthGain(std::vector<DataT> const& data, std::size_t split_bin_index)
  {
    auto bin_width = raft::ceildiv(params.n_rows, params.max_n_bins);
    std::vector<DataT> left_data(data.begin(), data.begin() + (split_bin_index + 1) * bin_width);
    std::vector<DataT> right_data(data.begin() + (split_bin_index + 1) * bin_width, data.end());

    auto [parent_ighd, label_sum, n]            = InverseGaussianHalfDeviance(data);
    auto [left_ighd, label_sum_left, n_left]    = InverseGaussianHalfDeviance(left_data);
    auto [right_ighd, label_sum_right, n_right] = InverseGaussianHalfDeviance(right_data);

    auto gain = parent_ighd -
                ((n_left / n) * left_ighd +    // the minimizing objective function is half deviance
                 (n_right / n) * right_ighd);  // gain in long form without proxy

    // edge cases
    if (n_left < params.min_samples_leaf or n_right < params.min_samples_leaf or
        label_sum < ObjectiveT::eps_ or label_sum_right < ObjectiveT::eps_ or
        label_sum_left < ObjectiveT::eps_)
      return -std::numeric_limits<DataT>::max();
    else
      return gain;
  }

  auto GammaHalfDeviance(
    std::vector<DataT> const& data)  //  1/n * 2 * sum(log(y_pred/y_true) + y_true/y_pred - 1)
  {
    DataT sum(0);
    sum              = std::accumulate(data.begin(), data.end(), DataT(0));
    DataT const mean = sum / data.size();
    DataT ghd(0);  // gamma half deviance

    std::for_each(data.begin(), data.end(), [&](auto& element) {
      auto log_y = raft::log(element ? element : DataT(1.0));
      ghd += raft::log(mean) - log_y + element / mean - 1;
    });

    ghd /= data.size();
    return std::make_tuple(ghd, sum, DataT(data.size()));
  }

  auto GammaGroundTruthGain(std::vector<DataT> const& data, std::size_t split_bin_index)
  {
    auto bin_width = raft::ceildiv(params.n_rows, params.max_n_bins);
    std::vector<DataT> left_data(data.begin(), data.begin() + (split_bin_index + 1) * bin_width);
    std::vector<DataT> right_data(data.begin() + (split_bin_index + 1) * bin_width, data.end());

    auto [parent_ghd, label_sum, n]            = GammaHalfDeviance(data);
    auto [left_ghd, label_sum_left, n_left]    = GammaHalfDeviance(left_data);
    auto [right_ghd, label_sum_right, n_right] = GammaHalfDeviance(right_data);

    auto gain =
      parent_ghd - ((n_left / n) * left_ghd +  // the minimizing objective function is half deviance
                    (n_right / n) * right_ghd);  // gain in long form without proxy

    // edge cases
    if (n_left < params.min_samples_leaf or n_right < params.min_samples_leaf or
        label_sum < ObjectiveT::eps_ or label_sum_right < ObjectiveT::eps_ or
        label_sum_left < ObjectiveT::eps_)
      return -std::numeric_limits<DataT>::max();
    else
      return gain;
  }

  auto PoissonHalfDeviance(
    std::vector<DataT> const& data)  //  1/n * sum(y_true * log(y_true/y_pred) + y_pred - y_true)
  {
    DataT sum       = std::accumulate(data.begin(), data.end(), DataT(0));
    auto const mean = sum / data.size();
    auto poisson_half_deviance{DataT(0.0)};

    std::for_each(data.begin(), data.end(), [&](auto d) {
      auto log_y = raft::log(d ? d : DataT(1.0));  // we don't want nans
      poisson_half_deviance += d * (log_y - raft::log(mean)) + mean - d;
    });

    poisson_half_deviance /= data.size();
    return std::make_tuple(poisson_half_deviance, sum, DataT(data.size()));
  }

  auto PoissonGroundTruthGain(std::vector<DataT> const& data, std::size_t split_bin_index)
  {
    auto bin_width = raft::ceildiv(params.n_rows, params.max_n_bins);
    std::vector<DataT> left_data(data.begin(), data.begin() + (split_bin_index + 1) * bin_width);
    std::vector<DataT> right_data(data.begin() + (split_bin_index + 1) * bin_width, data.end());

    auto [parent_phd, label_sum, n]            = PoissonHalfDeviance(data);
    auto [left_phd, label_sum_left, n_left]    = PoissonHalfDeviance(left_data);
    auto [right_phd, label_sum_right, n_right] = PoissonHalfDeviance(right_data);

    auto gain = parent_phd - ((n_left / n) * left_phd +
                              (n_right / n) * right_phd);  // gain in long form without proxy

    // edge cases
    if (n_left < params.min_samples_leaf or n_right < params.min_samples_leaf or
        label_sum < ObjectiveT::eps_ or label_sum_right < ObjectiveT::eps_ or
        label_sum_left < ObjectiveT::eps_)
      return -std::numeric_limits<DataT>::max();
    else
      return gain;
  }

  auto Entropy(std::vector<DataT> const& data)
  {  // sum((n_c/n_total)*(log(n_c/n_total)))
    DataT entropy(0);
    for (auto c = 0; c < params.n_classes; ++c) {
      IdxT sum(0);
      std::for_each(data.begin(), data.end(), [&](auto d) {
        if (d == DataT(c)) ++sum;
      });
      DataT class_proba = DataT(sum) / data.size();
      entropy += -class_proba * raft::log(class_proba ? class_proba : DataT(1)) /
                 raft::log(DataT(2));  // adding gain
    }
    return entropy;
  }

  auto EntropyGroundTruthGain(std::vector<DataT> const& data, std::size_t const split_bin_index)
  {
    auto bin_width = raft::ceildiv(params.n_rows, params.max_n_bins);
    std::vector<DataT> left_data(data.begin(), data.begin() + (split_bin_index + 1) * bin_width);
    std::vector<DataT> right_data(data.begin() + (split_bin_index + 1) * bin_width, data.end());

    auto parent_entropy = Entropy(data);
    auto left_entropy   = Entropy(left_data);
    auto right_entropy  = Entropy(right_data);
    DataT n             = data.size();
    DataT left_n        = left_data.size();
    DataT right_n       = right_data.size();

    auto gain = parent_entropy - ((left_n / n) * left_entropy + (right_n / n) * right_entropy);

    // edge cases
    if (left_n < params.min_samples_leaf or right_n < params.min_samples_leaf) {
      return -std::numeric_limits<DataT>::max();
    } else {
      return gain;
    }
  }

  auto GiniImpurity(std::vector<DataT> const& data)
  {  // sum((n_c/n_total)(1-(n_c/n_total)))
    DataT gini(0);
    for (auto c = 0; c < params.n_classes; ++c) {
      IdxT sum(0);
      std::for_each(data.begin(), data.end(), [&](auto d) {
        if (d == DataT(c)) ++sum;
      });
      DataT class_proba = DataT(sum) / data.size();
      gini += class_proba * (1 - class_proba);  // adding gain
    }
    return gini;
  }

  auto GiniGroundTruthGain(std::vector<DataT> const& data, std::size_t const split_bin_index)
  {
    auto bin_width = raft::ceildiv(params.n_rows, params.max_n_bins);
    std::vector<DataT> left_data(data.begin(), data.begin() + (split_bin_index + 1) * bin_width);
    std::vector<DataT> right_data(data.begin() + (split_bin_index + 1) * bin_width, data.end());

    auto parent_gini = GiniImpurity(data);
    auto left_gini   = GiniImpurity(left_data);
    auto right_gini  = GiniImpurity(right_data);
    DataT n          = data.size();
    DataT left_n     = left_data.size();
    DataT right_n    = right_data.size();

    auto gain = parent_gini - ((left_n / n) * left_gini + (right_n / n) * right_gini);

    // edge cases
    if (left_n < params.min_samples_leaf or right_n < params.min_samples_leaf) {
      return -std::numeric_limits<DataT>::max();
    } else {
      return gain;
    }
  }

  auto GroundTruthGain(std::vector<DataT> const& data, std::size_t const split_bin_index)
  {
    if constexpr (std::is_same<ObjectiveT, MSEObjectiveFunction<DataT, LabelT, IdxT>>::
                    value)  // mean squared error
    {
      return MSEGroundTruthGain(data, split_bin_index);
    } else if constexpr (std::is_same<ObjectiveT, PoissonObjectiveFunction<DataT, LabelT, IdxT>>::
                           value)  // poisson
    {
      return PoissonGroundTruthGain(data, split_bin_index);
    } else if constexpr (std::is_same<ObjectiveT,
                                      GammaObjectiveFunction<DataT, LabelT, IdxT>>::value)  // gamma
    {
      return GammaGroundTruthGain(data, split_bin_index);
    } else if constexpr (std::is_same<ObjectiveT,
                                      InverseGaussianObjectiveFunction<DataT, LabelT, IdxT>>::
                           value)  // inverse gaussian
    {
      return InverseGaussianGroundTruthGain(data, split_bin_index);
    } else if constexpr (std::is_same<ObjectiveT, EntropyObjectiveFunction<DataT, LabelT, IdxT>>::
                           value)  // entropy
    {
      return EntropyGroundTruthGain(data, split_bin_index);
    } else if constexpr (std::is_same<ObjectiveT,
                                      GiniObjectiveFunction<DataT, LabelT, IdxT>>::value)  // gini
    {
      return GiniGroundTruthGain(data, split_bin_index);
    }
    return DataT(0.0);
  }

  auto NumLeftOfBin(std::vector<BinT> const& cdf_hist, IdxT idx)
  {
    auto count{IdxT(0)};
    for (auto c = 0; c < params.n_classes; ++c) {
      if constexpr (std::is_same<BinT, CountBin>::value)  // countbin
      {
        count += static_cast<IdxT>(cdf_hist[params.max_n_bins * c + idx].x);
      } else  // aggregatebin
      {
        count += cdf_hist[params.max_n_bins * c + idx].count;
      }
    }
    return count;
  }

  void SetUp() override
  {
    srand(params.seed);
    params = ::testing::TestWithParam<ObjectiveTestParameters>::GetParam();
    ObjectiveT objective(params.n_classes, params.min_samples_leaf);

    auto data                 = GenRandomData();
    auto [cdf_hist, pdf_hist] = GenHist(data);
    auto split_bin_index      = RandUnder(params.max_n_bins);
    auto ground_truth_gain    = GroundTruthGain(data, split_bin_index);

    // Both Gini/Entropy (CountBin) and the regressor objectives (AggregateBin)
    // now take weighted denominators (W_total, W_left). Tests are unweighted,
    // so W == n; pass the unweighted sums through the weighted slots.
    auto W_total = static_cast<double>(NumLeftOfBin(cdf_hist, params.max_n_bins - 1));
    auto W_left  = static_cast<double>(NumLeftOfBin(cdf_hist, split_bin_index));
    DataT hypothesis_gain;
    hypothesis_gain = objective.GainPerSplit(&cdf_hist[0],
                                             split_bin_index,
                                             params.max_n_bins,
                                             NumLeftOfBin(cdf_hist, params.max_n_bins - 1),
                                             NumLeftOfBin(cdf_hist, split_bin_index),
                                             W_total,
                                             W_left);

    // The gain may actually be NaN. If so, a comparison between the result and
    // ground truth would yield false, even if they are both (correctly) NaNs.
    if (!std::isnan(ground_truth_gain) || !std::isnan(hypothesis_gain)) {
      ASSERT_NEAR(ground_truth_gain, hypothesis_gain, params.tolerance);
    }
  }
};

const std::vector<ObjectiveTestParameters> mse_objective_test_parameters = {
  {9507819643927052255LLU, 2048, 64, 1, 0, 0.00001},
  {9507819643927052259LLU, 2048, 128, 1, 1, 0.00001},
  {9507819643927052251LLU, 2048, 256, 1, 1, 0.00001},
  {9507819643927052258LLU, 2048, 512, 1, 5, 0.00001},
};

const std::vector<ObjectiveTestParameters> poisson_objective_test_parameters = {
  {9507819643927052255LLU, 2048, 64, 1, 0, 0.00001},
  {9507819643927052259LLU, 2048, 128, 1, 1, 0.00001},
  {9507819643927052251LLU, 2048, 256, 1, 1, 0.00001},
  {9507819643927052258LLU, 2048, 512, 1, 5, 0.00001},
};

const std::vector<ObjectiveTestParameters> gamma_objective_test_parameters = {
  {9507819643927052255LLU, 2048, 64, 1, 0, 0.00001},
  {9507819643927052259LLU, 2048, 128, 1, 1, 0.00001},
  {9507819643927052251LLU, 2048, 256, 1, 1, 0.00001},
  {9507819643927052258LLU, 2048, 512, 1, 5, 0.00001},
};

const std::vector<ObjectiveTestParameters> invgauss_objective_test_parameters = {
  {9507819643927052255LLU, 2048, 64, 1, 0, 0.00001},
  {9507819643927052259LLU, 2048, 128, 1, 1, 0.00001},
  {9507819643927052251LLU, 2048, 256, 1, 1, 0.00001},
  {9507819643927052258LLU, 2048, 512, 1, 5, 0.00001},
};

const std::vector<ObjectiveTestParameters> entropy_objective_test_parameters = {
  {9507819643927052255LLU, 2048, 64, 2, 0, 0.00001},
  {9507819643927052256LLU, 2048, 128, 10, 1, 0.00001},
  {9507819643927052257LLU, 2048, 256, 100, 1, 0.00001},
  {9507819643927052258LLU, 2048, 512, 100, 5, 0.00001},
};

const std::vector<ObjectiveTestParameters> gini_objective_test_parameters = {
  {9507819643927052255LLU, 2048, 64, 2, 0, 0.00001},
  {9507819643927052256LLU, 2048, 128, 10, 1, 0.00001},
  {9507819643927052257LLU, 2048, 256, 100, 1, 0.00001},
  {9507819643927052258LLU, 2048, 512, 100, 5, 0.00001},
};

// mse objective test
typedef ObjectiveTest<MSEObjectiveFunction<double, double, int>> MSEObjectiveTestD;
TEST_P(MSEObjectiveTestD, MSEObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        MSEObjectiveTestD,
                        ::testing::ValuesIn(mse_objective_test_parameters));
typedef ObjectiveTest<MSEObjectiveFunction<float, float, int>> MSEObjectiveTestF;
TEST_P(MSEObjectiveTestF, MSEObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        MSEObjectiveTestF,
                        ::testing::ValuesIn(mse_objective_test_parameters));

// poisson objective test
typedef ObjectiveTest<PoissonObjectiveFunction<double, double, int>> PoissonObjectiveTestD;
TEST_P(PoissonObjectiveTestD, poissonObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        PoissonObjectiveTestD,
                        ::testing::ValuesIn(poisson_objective_test_parameters));
typedef ObjectiveTest<PoissonObjectiveFunction<float, float, int>> PoissonObjectiveTestF;
TEST_P(PoissonObjectiveTestF, poissonObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        PoissonObjectiveTestF,
                        ::testing::ValuesIn(poisson_objective_test_parameters));

// gamma objective test
typedef ObjectiveTest<GammaObjectiveFunction<double, double, int>> GammaObjectiveTestD;
TEST_P(GammaObjectiveTestD, GammaObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        GammaObjectiveTestD,
                        ::testing::ValuesIn(gamma_objective_test_parameters));
typedef ObjectiveTest<GammaObjectiveFunction<float, float, int>> GammaObjectiveTestF;
TEST_P(GammaObjectiveTestF, GammaObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        GammaObjectiveTestF,
                        ::testing::ValuesIn(gamma_objective_test_parameters));

// InvGauss objective test
typedef ObjectiveTest<InverseGaussianObjectiveFunction<double, double, int>>
  InverseGaussianObjectiveTestD;
TEST_P(InverseGaussianObjectiveTestD, InverseGaussianObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        InverseGaussianObjectiveTestD,
                        ::testing::ValuesIn(invgauss_objective_test_parameters));
typedef ObjectiveTest<InverseGaussianObjectiveFunction<float, float, int>>
  InverseGaussianObjectiveTestF;
TEST_P(InverseGaussianObjectiveTestF, InverseGaussianObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        InverseGaussianObjectiveTestF,
                        ::testing::ValuesIn(invgauss_objective_test_parameters));

// entropy objective test
typedef ObjectiveTest<EntropyObjectiveFunction<double, int, int>> EntropyObjectiveTestD;
TEST_P(EntropyObjectiveTestD, entropyObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        EntropyObjectiveTestD,
                        ::testing::ValuesIn(entropy_objective_test_parameters));
typedef ObjectiveTest<EntropyObjectiveFunction<float, int, int>> EntropyObjectiveTestF;
TEST_P(EntropyObjectiveTestF, entropyObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        EntropyObjectiveTestF,
                        ::testing::ValuesIn(entropy_objective_test_parameters));

// gini objective test
typedef ObjectiveTest<GiniObjectiveFunction<double, int, int>> GiniObjectiveTestD;
TEST_P(GiniObjectiveTestD, giniObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        GiniObjectiveTestD,
                        ::testing::ValuesIn(gini_objective_test_parameters));
typedef ObjectiveTest<GiniObjectiveFunction<float, int, int>> GiniObjectiveTestF;
TEST_P(GiniObjectiveTestF, giniObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        GiniObjectiveTestF,
                        ::testing::ValuesIn(gini_objective_test_parameters));

#ifndef NDEBUG
// Feature sampling bias test
struct FeatureSamplingBiasTestParams {
  int n_features;
  int n_nodes;
  int k;                   // features sampled per node
  double tolerance_ratio;  // acceptable deviation from 1.0
};

class FeatureSamplingBiasTest : public ::testing::TestWithParam<FeatureSamplingBiasTestParams> {
 protected:
  void SetUp() override
  {
    params      = ::testing::TestWithParam<FeatureSamplingBiasTestParams>::GetParam();
    stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
    handle.reset(new raft::handle_t(rmm::cuda_stream_per_thread, stream_pool));
  }

  void TearDown() override
  {
    handle.reset();
    stream_pool.reset();
  }

  void TestFeatureSamplingBias()
  {
    auto stream = handle->get_stream();

    // Allocate device memory
    rmm::device_uvector<int> d_colids(params.n_nodes * params.k, stream);
    rmm::device_uvector<NodeWorkItem> d_work_items(params.n_nodes, stream);
    rmm::device_uvector<unsigned long long> d_counts(params.n_features, stream);

    // Initialize work items on host
    std::vector<NodeWorkItem> h_work_items(params.n_nodes);
    for (int i = 0; i < params.n_nodes; ++i) {
      h_work_items[i].idx             = i;
      h_work_items[i].depth           = 0;
      h_work_items[i].instances.begin = 0;
      h_work_items[i].instances.count = 100;  // arbitrary value
    }

    // Copy to device
    raft::update_device(d_work_items.data(), h_work_items.data(), params.n_nodes, stream);

    // Initialize counts to zero
    RAFT_CUDA_TRY(
      cudaMemsetAsync(d_counts.data(), 0, params.n_features * sizeof(unsigned long long), stream));

    // Calculate n_parallel_samples using the same formula as in builder.cuh
    const int BLOCK_THREADS          = 128;
    const int MAX_SAMPLES_PER_THREAD = 1;
    // Formula: log(1 - k/n) / log(1 - 1/n)
    // where k = params.k (features to sample), n = params.n_features (total features)
    int n_parallel_samples = std::ceil(raft::log(1 - double(params.k) / double(params.n_features)) /
                                       raft::log(1 - 1.0 / double(params.n_features)));

    // Verify that test conditions ensure excess_sample_with_replacement_kernel is used
    // (instead of falling back to algo_L_sample_kernel)
    ASSERT_GE(MAX_SAMPLES_PER_THREAD * BLOCK_THREADS, n_parallel_samples)
      << "Test parameters would trigger reservoir sampling instead of excess sampling. "
      << "n_parallel_samples=" << n_parallel_samples
      << ", max capacity=" << (MAX_SAMPLES_PER_THREAD * BLOCK_THREADS);

    // Run the sampling kernel with diagnostics enabled
    excess_sample_with_replacement_kernel<int, MAX_SAMPLES_PER_THREAD, BLOCK_THREADS>
      <<<params.n_nodes, BLOCK_THREADS, 0, stream>>>(d_colids.data(),
                                                     d_work_items.data(),
                                                     params.n_nodes,
                                                     0,   // treeid
                                                     42,  // seed
                                                     params.n_features,
                                                     params.k,
                                                     n_parallel_samples,
                                                     d_counts.data());

    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

    // Copy counts back to host for verification
    std::vector<unsigned long long> h_counts(params.n_features);
    raft::update_host(h_counts.data(), d_counts.data(), params.n_features, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

    // Copy colids back to host for duplicate checking
    std::vector<int> h_colids(params.n_nodes * params.k);
    raft::update_host(h_colids.data(), d_colids.data(), params.n_nodes * params.k, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

    // Verify that each node's sampled features are unique and valid
    for (int node = 0; node < params.n_nodes; ++node) {
      std::vector<bool> feature_seen(params.n_features, false);
      int unique_count = 0;

      for (int j = 0; j < params.k; ++j) {
        int feature_idx = h_colids[node * params.k + j];

        // Check feature index is within valid range
        EXPECT_GE(feature_idx, 0) << "Node " << node << " has invalid feature index " << feature_idx
                                  << " (< 0)";
        EXPECT_LT(feature_idx, params.n_features)
          << "Node " << node << " has invalid feature index " << feature_idx
          << " (>= n_features=" << params.n_features << ")";

        // Check for duplicates
        if (feature_idx >= 0 && feature_idx < params.n_features) {
          EXPECT_FALSE(feature_seen[feature_idx]) << "Node " << node << " has duplicate feature "
                                                  << feature_idx << " at positions in sampled set";
          if (!feature_seen[feature_idx]) {
            feature_seen[feature_idx] = true;
            unique_count++;
          }
        }
      }

      EXPECT_EQ(unique_count, params.k) << "Node " << node << " should have exactly " << params.k
                                        << " unique features, but got " << unique_count;
    }

    // Verify uniform sampling (no bias)
    unsigned long long total_samples = params.n_nodes * params.k;
    double expected_per_feature      = double(total_samples) / params.n_features;

    // Check for feature 0 under-sampling
    double feature_0_ratio = h_counts[0] / expected_per_feature;
    EXPECT_GT(feature_0_ratio, 1.0 - params.tolerance_ratio)
      << "Feature 0 is under-sampled! Ratio: " << feature_0_ratio << " (expected ~1.0)";

    // Check for feature n-1 over-sampling
    double feature_n1_ratio = h_counts[params.n_features - 1] / expected_per_feature;
    EXPECT_LT(feature_n1_ratio, 1.0 + params.tolerance_ratio)
      << "Feature n-1 is over-sampled! Ratio: " << feature_n1_ratio << " (expected ~1.0)";

    // Check all features are reasonably sampled
    for (int i = 0; i < params.n_features; ++i) {
      double ratio = h_counts[i] / expected_per_feature;
      EXPECT_GT(ratio, 1.0 - params.tolerance_ratio)
        << "Feature " << i << " under-sampled. Ratio: " << ratio;
      EXPECT_LT(ratio, 1.0 + params.tolerance_ratio)
        << "Feature " << i << " over-sampled. Ratio: " << ratio;
    }
  }

  std::shared_ptr<rmm::cuda_stream_pool> stream_pool;
  std::shared_ptr<raft::handle_t> handle;
  FeatureSamplingBiasTestParams params;
};

TEST_P(FeatureSamplingBiasTest, UniformSampling) { TestFeatureSamplingBias(); }

const std::vector<FeatureSamplingBiasTestParams> feature_sampling_bias_test_parameters = {
  {10, 1000, 5, 0.15},   // 10 features, 1000 nodes, sample 5, allow 15% deviation
  {20, 1000, 10, 0.15},  // 20 features, 1000 nodes, sample 10, allow 15% deviation
  {50, 2000, 25, 0.12},  // 50 features, 2000 nodes, sample 25, allow 12% deviation
};

INSTANTIATE_TEST_CASE_P(RfTests,
                        FeatureSamplingBiasTest,
                        ::testing::ValuesIn(feature_sampling_bias_test_parameters));
#endif  // NDEBUG

}  // end namespace DT
}  // end namespace ML
