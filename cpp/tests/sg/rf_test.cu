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

#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <tuple>
#include <type_traits>

namespace ML {

namespace DT {

template <typename T>
using ReturnValue = std::tuple<ML::DT::Quantiles<T, int>,
                               std::shared_ptr<rmm::device_uvector<T>>,
                               std::shared_ptr<rmm::device_uvector<int>>>;

template <typename T>
ReturnValue<T> computeQuantiles(
  const raft::handle_t& handle, const T* data, int max_n_bins, int n_rows, int n_cols);

template <>
ReturnValue<float> computeQuantiles<float>(
  const raft::handle_t& handle, const float* data, int max_n_bins, int n_rows, int n_cols);

template <>
ReturnValue<double> computeQuantiles<double>(
  const raft::handle_t& handle, const double* data, int max_n_bins, int n_rows, int n_cols);
}  // namespace DT

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
    thrust::device_vector<int> histogram(params.max_n_bins);
    thrust::host_vector<int> h_histogram(params.max_n_bins);

    raft::random::Rng r(8);
    r.normal(data.data().get(), data.size(), T(0.0), T(2.0), nullptr);
    auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
    raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);

    // computing the quantiles
    auto [quantiles, quantiles_array, n_bins_array] =
      DT::computeQuantiles(handle, data.data().get(), params.max_n_bins, params.n_rows, 1);

    int n_unique_bins;
    raft::copy(&n_unique_bins, quantiles.n_bins_array, 1, handle.get_stream());
    if (n_unique_bins < params.max_n_bins) {
      return;  // almost impossible that this happens, skip if so
    }

    auto d_quantiles = quantiles.quantiles_array;
    auto d_histogram = histogram.data().get();

    thrust::for_each(data.begin(), data.end(), [=] __device__(T x) {
      for (int j = 0; j < params.max_n_bins; j++) {
        if (x <= d_quantiles[j]) {
          atomicAdd(&d_histogram[j], 1);
          break;
        }
      }
    });

    h_histogram           = histogram;
    int max_items_per_bin = raft::ceildiv(params.n_rows, params.max_n_bins);
    int min_items_per_bin = max_items_per_bin - 1;
    int total_items       = 0;
    for (int b = 0; b < params.max_n_bins; b++) {
      ASSERT_TRUE(h_histogram[b] == max_items_per_bin or h_histogram[b] == min_items_per_bin)
        << "No. samples in bin[" << b << "] = " << h_histogram[b] << " Expected "
        << max_items_per_bin << " or " << min_items_per_bin << std::endl;
      total_items += h_histogram[b];
    }
    ASSERT_EQ(params.n_rows, total_items)
      << "Some samples from dataset are either missed of double counted in quantile bins"
      << std::endl;
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

    // calling computeQuantiles
    auto [quantiles, quantiles_array, n_bins_array] =
      DT::computeQuantiles(handle, data.data().get(), params.max_n_bins, params.n_rows, 1);
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

// ---- sample_weight fit-level gtests ----------------------------------------

namespace {

// Deterministic config shared by every equivalence test: bootstrap off so the
// duplicated-rows population (A) and the weighted population (B) match exactly.
RF_params WeightedEquivParams(CRITERION crit)
{
  return set_rf_params(/*max_depth*/ 8,
                       /*max_leaves*/ -1,
                       /*max_features*/ 1.0f,
                       /*max_n_bins*/ 128,
                       /*min_samples_leaf*/ 1,
                       /*min_samples_split*/ 2,
                       /*min_impurity_decrease*/ 0.0f,
                       /*bootstrap*/ false,
                       /*n_trees*/ 1,
                       /*max_samples*/ 1.0f,
                       /*seed*/ 42,
                       crit,
                       /*n_streams*/ 1,
                       128);
}

// Matched binning: max_n_bins > rows so every distinct value is its own
// edge and duplication cannot shift the grid, making dup-K == weight-K
// integer-exact (design 0.1 R6).
RF_params MatchedBinParams(CRITERION crit, int bins)
{
  return set_rf_params(/*max_depth*/ 8,
                       /*max_leaves*/ -1,
                       /*max_features*/ 1.0f,
                       /*max_n_bins*/ bins,
                       /*min_samples_leaf*/ 1,
                       /*min_samples_split*/ 2,
                       /*min_impurity_decrease*/ 0.0f,
                       /*bootstrap*/ false,
                       /*n_trees*/ 1,
                       /*max_samples*/ 1.0f,
                       /*seed*/ 42,
                       crit,
                       /*n_streams*/ 1,
                       bins);
}

// Continuous, per-feature-distinct host data (row-major) so exact bin ties are
// measure-zero. Small integer weights in [1, 3].
template <typename DataT>
void GenWeightedFixture(int n_rows,
                        int n_cols,
                        int n_classes,
                        std::vector<DataT>& X_rm,
                        std::vector<int>& y,
                        std::vector<int>& w)
{
  std::mt19937 rng(2026);
  std::uniform_real_distribution<double> ux(-5.0, 5.0);
  std::uniform_int_distribution<int> uw(1, 3);
  X_rm.resize(static_cast<std::size_t>(n_rows) * n_cols);
  y.resize(n_rows);
  w.resize(n_rows);
  for (int r = 0; r < n_rows; ++r) {
    for (int c = 0; c < n_cols; ++c)
      X_rm[static_cast<std::size_t>(r) * n_cols + c] = DataT(ux(rng) + 0.001 * c);
    y[r] = r % n_classes;
    w[r] = uw(rng);
  }
}

// Column-major device buffer (cuML RF fit expects Fortran order).
template <typename DataT>
thrust::device_vector<DataT> ToColMajorDevice(const std::vector<DataT>& X_rm,
                                              int n_rows,
                                              int n_cols)
{
  std::vector<DataT> cm(static_cast<std::size_t>(n_rows) * n_cols);
  for (int r = 0; r < n_rows; ++r)
    for (int c = 0; c < n_cols; ++c)
      cm[static_cast<std::size_t>(c) * n_rows + r] = X_rm[static_cast<std::size_t>(r) * n_cols + c];
  return thrust::device_vector<DataT>(cm.begin(), cm.end());
}

template <typename DataT, typename LabelT>
std::shared_ptr<RandomForestMetaData<DataT, LabelT>> FitClassifier(
  const raft::handle_t& handle,
  thrust::device_vector<DataT>& X_cm,
  thrust::device_vector<int>& y,
  int n_rows,
  int n_cols,
  int n_classes,
  RF_params rf_params,
  DataT* sample_weight)
{
  auto forest = std::make_shared<RandomForestMetaData<DataT, LabelT>>();
  fit(handle,
      forest.get(),
      X_cm.data().get(),
      n_rows,
      n_cols,
      y.data().get(),
      n_classes,
      rf_params,
      rapids_logger::level_enum::info,
      nullptr,
      sample_weight);
  return forest;
}

// Host tree walk: x[col] <= quesval goes left (decisiontree.cuh:387).
template <typename DataT, typename LabelT>
int RouteToLeaf(const std::vector<SparseTreeNode<DataT, LabelT>>& tree, const DataT* row)
{
  int nid = 0;
  while (!tree[nid].IsLeaf()) {
    nid = (row[tree[nid].ColumnId()] <= tree[nid].QueryValue()) ? tree[nid].LeftChildId()
                                                                : tree[nid].RightChildId();
  }
  return nid;
}

// Co-clustering equality: rows i,j share a leaf in A iff they share one in B.
// Leaf ids are NOT comparable across A and B (different node-array sizes).
template <typename DataT, typename LabelT>
void ExpectSameLeafPartition(const std::vector<SparseTreeNode<DataT, LabelT>>& a,
                             const std::vector<SparseTreeNode<DataT, LabelT>>& b,
                             const std::vector<DataT>& X_rm,
                             int n_rows,
                             int n_cols)
{
  std::vector<int> la(n_rows), lb(n_rows);
  for (int r = 0; r < n_rows; ++r) {
    const DataT* row = &X_rm[static_cast<std::size_t>(r) * n_cols];
    la[r]            = RouteToLeaf(a, row);
    lb[r]            = RouteToLeaf(b, row);
  }
  for (int i = 0; i < n_rows; ++i)
    for (int j = i + 1; j < n_rows; ++j)
      EXPECT_EQ(la[i] == la[j], lb[i] == lb[j])
        << "leaf-partition divergence at rows " << i << "," << j;
}

}  // namespace

template <typename DataT>
void RunClassifierSampleWeightEquivalence()
{
  const int n_rows = 80, n_cols = 4, n_classes = 3;
  std::vector<DataT> X_rm;
  std::vector<int> y, w;
  GenWeightedFixture<DataT>(n_rows, n_cols, n_classes, X_rm, y, w);

  // Duplicated dataset for the unweighted fit A.
  std::vector<DataT> Xd_rm;
  std::vector<int> yd;
  for (int r = 0; r < n_rows; ++r)
    for (int k = 0; k < w[r]; ++k) {
      yd.push_back(y[r]);
      for (int c = 0; c < n_cols; ++c)
        Xd_rm.push_back(X_rm[static_cast<std::size_t>(r) * n_cols + c]);
    }
  const int nd = static_cast<int>(yd.size());

  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
  raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);

  auto Xd_cm = ToColMajorDevice<DataT>(Xd_rm, nd, n_cols);
  auto X_cm  = ToColMajorDevice<DataT>(X_rm, n_rows, n_cols);
  thrust::device_vector<int> yd_dev(yd.begin(), yd.end());
  thrust::device_vector<int> y_dev(y.begin(), y.end());
  std::vector<DataT> wd(w.begin(), w.end());
  thrust::device_vector<DataT> w_dev(wd.begin(), wd.end());

  // Matched binning (see MatchedBinParams / design 0.1 R6): identical grid
  // for the duplicated-unweighted and weighted fits, so dup-K == weight-K is
  // integer-exact on both F and D. Default-bin topology is NOT asserted.
  const int bins = nd + 16;
  auto A         = FitClassifier<DataT, int>(
    handle, Xd_cm, yd_dev, nd, n_cols, n_classes, MatchedBinParams(GINI, bins), nullptr);
  auto B = FitClassifier<DataT, int>(handle,
                                     X_cm,
                                     y_dev,
                                     n_rows,
                                     n_cols,
                                     n_classes,
                                     MatchedBinParams(GINI, bins),
                                     w_dev.data().get());

  const auto& at  = A->trees[0]->sparsetree;
  const auto& bt  = B->trees[0]->sparsetree;
  const auto& wnc = B->trees[0]->weighted_node_count;

  // Identical tree structure under the matched grid.
  ASSERT_EQ(at.size(), bt.size());
  ASSERT_EQ(wnc.size(), bt.size());
  // Load-bearing kernel-correctness anchor: B's weighted count per node
  // equals A's duplicated instance count, integer-exact.
  for (std::size_t i = 0; i < bt.size(); ++i)
    EXPECT_EQ(double(at[i].InstanceCount()), wnc[i]);
  // Original rows induce an identical leaf partition.
  ExpectSameLeafPartition<DataT, int>(at, bt, X_rm, n_rows, n_cols);
}

TEST(RfTests, ClassifierFit_SampleWeightEquivalenceMatchedBinsF)
{
  RunClassifierSampleWeightEquivalence<float>();
}
TEST(RfTests, ClassifierFit_SampleWeightEquivalenceMatchedBinsD)
{
  RunClassifierSampleWeightEquivalence<double>();
}

template <typename DataT>
void RunRegressorSampleWeightEquivalence()
{
  const int n_rows = 80, n_cols = 4;
  std::vector<DataT> X_rm;
  std::vector<int> ylbl, w;
  GenWeightedFixture<DataT>(n_rows, n_cols, /*n_classes*/ 7, X_rm, ylbl, w);
  std::vector<DataT> y(n_rows);
  for (int r = 0; r < n_rows; ++r)
    y[r] = DataT(ylbl[r]) + DataT(0.5);

  std::vector<DataT> Xd_rm;
  std::vector<DataT> yd;
  for (int r = 0; r < n_rows; ++r)
    for (int k = 0; k < w[r]; ++k) {
      yd.push_back(y[r]);
      for (int c = 0; c < n_cols; ++c)
        Xd_rm.push_back(X_rm[static_cast<std::size_t>(r) * n_cols + c]);
    }
  const int nd = static_cast<int>(yd.size());

  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
  raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);

  auto Xd_cm = ToColMajorDevice<DataT>(Xd_rm, nd, n_cols);
  auto X_cm  = ToColMajorDevice<DataT>(X_rm, n_rows, n_cols);
  thrust::device_vector<DataT> yd_dev(yd.begin(), yd.end());
  thrust::device_vector<DataT> y_dev(y.begin(), y.end());
  std::vector<DataT> wd(w.begin(), w.end());
  thrust::device_vector<DataT> w_dev(wd.begin(), wd.end());

  // Matched binning (see classifier variant / design 0.1 R6): identical grid
  // for the duplicated-unweighted and weighted fits => integer-exact dup-K ==
  // weight-K on both float and double.
  const int bins = nd + 16;
  auto A         = std::make_shared<RandomForestMetaData<DataT, DataT>>();
  fit(handle,
      A.get(),
      Xd_cm.data().get(),
      nd,
      n_cols,
      yd_dev.data().get(),
      MatchedBinParams(MSE, bins),
      rapids_logger::level_enum::info,
      nullptr,
      nullptr);
  auto B = std::make_shared<RandomForestMetaData<DataT, DataT>>();
  fit(handle,
      B.get(),
      X_cm.data().get(),
      n_rows,
      n_cols,
      y_dev.data().get(),
      MatchedBinParams(MSE, bins),
      rapids_logger::level_enum::info,
      nullptr,
      w_dev.data().get());

  const auto& at  = A->trees[0]->sparsetree;
  const auto& bt  = B->trees[0]->sparsetree;
  const auto& wnc = B->trees[0]->weighted_node_count;

  ASSERT_EQ(at.size(), bt.size());
  ASSERT_EQ(wnc.size(), bt.size());
  for (std::size_t i = 0; i < bt.size(); ++i)
    EXPECT_EQ(double(at[i].InstanceCount()), wnc[i]);
  ExpectSameLeafPartition<DataT, DataT>(at, bt, X_rm, n_rows, n_cols);
}

TEST(RfTests, RegressorFit_SampleWeightEquivalenceMatchedBinsF)
{
  RunRegressorSampleWeightEquivalence<float>();
}
TEST(RfTests, RegressorFit_SampleWeightEquivalenceMatchedBinsD)
{
  RunRegressorSampleWeightEquivalence<double>();
}

// Unweighted backstop: sample_weight=nullptr must leave the side-vector empty.
TEST(RfTests, UnweightedWeightedNodeCountIsEmpty)
{
  const int n_rows = 60, n_cols = 4, n_classes = 2;
  std::vector<float> X_rm;
  std::vector<int> y, w;
  GenWeightedFixture<float>(n_rows, n_cols, n_classes, X_rm, y, w);

  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
  raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
  auto X_cm = ToColMajorDevice<float>(X_rm, n_rows, n_cols);
  thrust::device_vector<int> y_dev(y.begin(), y.end());

  auto clf = FitClassifier<float, int>(
    handle, X_cm, y_dev, n_rows, n_cols, n_classes, WeightedEquivParams(GINI), nullptr);
  for (const auto& tree : clf->trees)
    EXPECT_TRUE(tree->weighted_node_count.empty());

  std::vector<float> yr(n_rows);
  for (int r = 0; r < n_rows; ++r)
    yr[r] = float(y[r]) + 0.5f;
  thrust::device_vector<float> yr_dev(yr.begin(), yr.end());
  auto reg = std::make_shared<RandomForestMetaData<float, float>>();
  fit(handle,
      reg.get(),
      X_cm.data().get(),
      n_rows,
      n_cols,
      yr_dev.data().get(),
      WeightedEquivParams(MSE),
      rapids_logger::level_enum::info,
      nullptr,
      nullptr);
  for (const auto& tree : reg->trees)
    EXPECT_TRUE(tree->weighted_node_count.empty());
}

// Side-vector lockstep + child-sum invariant. The child-sum equality is
// algebraically true by construction (Push: right = parent - left); this test
// validates the index-lockstep with sparsetree and the unit-weight kernel.
TEST(RfTests, WeightedChildSumInvariant)
{
  const int n_rows = 80, n_cols = 4, n_classes = 3;
  std::vector<float> X_rm;
  std::vector<int> y, w;
  GenWeightedFixture<float>(n_rows, n_cols, n_classes, X_rm, y, w);

  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
  raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
  auto X_cm = ToColMajorDevice<float>(X_rm, n_rows, n_cols);
  thrust::device_vector<int> y_dev(y.begin(), y.end());

  std::vector<float> wnu(w.begin(), w.end());
  thrust::device_vector<float> wnu_dev(wnu.begin(), wnu.end());
  auto B          = FitClassifier<float, int>(handle,
                                     X_cm,
                                     y_dev,
                                     n_rows,
                                     n_cols,
                                     n_classes,
                                     WeightedEquivParams(GINI),
                                     wnu_dev.data().get());
  const auto& bt  = B->trees[0]->sparsetree;
  const auto& wnc = B->trees[0]->weighted_node_count;
  ASSERT_EQ(wnc.size(), bt.size());
  // Push sets right_wc = parent_wc - left_wc, so the child-sum is exact in
  // double (observed residual 0.0); abs_tol is a conservative FP margin.
  const double abs_tol = 1e-9;
  for (std::size_t nid = 0; nid < bt.size(); ++nid)
    if (!bt[nid].IsLeaf())
      EXPECT_NEAR(wnc[bt[nid].LeftChildId()] + wnc[bt[nid].RightChildId()], wnc[nid], abs_tol);

  std::vector<float> ones(n_rows, 1.0f);
  thrust::device_vector<float> ones_dev(ones.begin(), ones.end());
  auto U           = FitClassifier<float, int>(handle,
                                     X_cm,
                                     y_dev,
                                     n_rows,
                                     n_cols,
                                     n_classes,
                                     WeightedEquivParams(GINI),
                                     ones_dev.data().get());
  const auto& ut   = U->trees[0]->sparsetree;
  const auto& uwnc = U->trees[0]->weighted_node_count;
  ASSERT_EQ(uwnc.size(), ut.size());
  for (std::size_t nid = 0; nid < ut.size(); ++nid)
    EXPECT_EQ(uwnc[nid], double(ut[nid].InstanceCount()));
}

// Unweighted classification is bit-identical across runs (integer atomics are
// associative). Guards a nondeterminism regression only.
TEST(RfTests, ClassifierFit_UnweightedDeterminism)
{
  // n_rows > TPB_DEFAULT (128) so the root spawns multiple blocks and the
  // cross-block reduction path is exercised.
  const int n_rows = 1024, n_cols = 4, n_classes = 3;
  std::vector<float> X_rm;
  std::vector<int> y, w;
  GenWeightedFixture<float>(n_rows, n_cols, n_classes, X_rm, y, w);

  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
  raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
  auto X_cm = ToColMajorDevice<float>(X_rm, n_rows, n_cols);
  thrust::device_vector<int> y_dev(y.begin(), y.end());

  std::shared_ptr<RandomForestMetaData<float, int>> ref;
  for (int run = 0; run < 5; ++run) {
    auto f = FitClassifier<float, int>(
      handle, X_cm, y_dev, n_rows, n_cols, n_classes, WeightedEquivParams(GINI), nullptr);
    if (run == 0)
      ref = f;
    else
      EXPECT_EQ(ref->trees[0]->sparsetree, f->trees[0]->sparsetree);
  }
}

// Regressor mirror of ClassifierFit_UnweightedDeterminism: the ordered
// pool merge also makes the unweighted AggregateBin path deterministic
// (load-bearing side effect; the classifier test only covers CountBin).
TEST(RfTests, RegressorFit_UnweightedDeterminism)
{
  // n_rows > TPB_DEFAULT (128) so the root spawns multiple blocks and the
  // cross-block reduction path is exercised.
  const int n_rows = 1024, n_cols = 4;
  std::vector<float> X_rm;
  std::vector<int> ylbl, w;
  GenWeightedFixture<float>(n_rows, n_cols, /*n_classes*/ 7, X_rm, ylbl, w);
  std::vector<float> y(n_rows);
  for (int r = 0; r < n_rows; ++r)
    y[r] = float(ylbl[r]) + 0.5f;

  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
  raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
  auto X_cm = ToColMajorDevice<float>(X_rm, n_rows, n_cols);
  thrust::device_vector<float> y_dev(y.begin(), y.end());

  std::shared_ptr<RandomForestMetaData<float, float>> ref;
  for (int run = 0; run < 5; ++run) {
    auto f = std::make_shared<RandomForestMetaData<float, float>>();
    fit(handle,
        f.get(),
        X_cm.data().get(),
        n_rows,
        n_cols,
        y_dev.data().get(),
        WeightedEquivParams(MSE),
        rapids_logger::level_enum::info,
        nullptr,
        nullptr);
    if (run == 0)
      ref = f;
    else
      EXPECT_EQ(ref->trees[0]->sparsetree, f->trees[0]->sparsetree);
  }
}

// Weighted classification predictions agree across repeated fits. Observed
// agreement = 1.000000 at n=12000 (pool reduction deterministic for
// num_blocks <= POOL_SIZE); the assertion floor is documented at its site.
TEST(RfTests, ClassifierFit_WeightedApproxDeterminism)
{
  const int n_rows = 12000, n_cols = 6, n_classes = 4;
  std::vector<float> X_rm;
  std::vector<int> y, w;
  GenWeightedFixture<float>(n_rows, n_cols, n_classes, X_rm, y, w);

  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
  raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
  auto X_cm = ToColMajorDevice<float>(X_rm, n_rows, n_cols);
  thrust::device_vector<int> y_dev(y.begin(), y.end());
  std::vector<float> wf(w.begin(), w.end());
  thrust::device_vector<float> w_dev(wf.begin(), wf.end());

  auto leaves = [&](const std::shared_ptr<RandomForestMetaData<float, int>>& f) {
    std::vector<int> L(n_rows);
    for (int r = 0; r < n_rows; ++r)
      L[r] = RouteToLeaf(f->trees[0]->sparsetree, &X_rm[static_cast<std::size_t>(r) * n_cols]);
    return L;
  };
  auto ref = FitClassifier<float, int>(
    handle, X_cm, y_dev, n_rows, n_cols, n_classes, WeightedEquivParams(GINI), w_dev.data().get());
  auto Lref = leaves(ref);
  for (int run = 1; run < 5; ++run) {
    auto f      = FitClassifier<float, int>(handle,
                                       X_cm,
                                       y_dev,
                                       n_rows,
                                       n_cols,
                                       n_classes,
                                       WeightedEquivParams(GINI),
                                       w_dev.data().get());
    auto L      = leaves(f);
    int matched = 0;
    for (int r = 0; r < n_rows; ++r)
      matched += (L[r] == Lref[r]);
    // Observed agreement 1.0 over 4 fits at n=12000 (pool deterministic for
    // num_blocks <= POOL_SIZE); 0.9995 floor margins larger-size FP-atomic
    // ordering the design does not control.
    EXPECT_GE(double(matched) / n_rows, 0.9995);
  }
}

// Per-block smem budget exceeded surfaces as raft::logic_error, not CUDA OOM.
TEST(RfTests, BuilderSmemBudgetExceeded_Throws)
{
  const int n_rows = 6000, n_cols = 10, n_classes = 300;
  std::vector<float> X_rm;
  std::vector<int> y, w;
  GenWeightedFixture<float>(n_rows, n_cols, n_classes, X_rm, y, w);

  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
  raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
  auto X_cm = ToColMajorDevice<float>(X_rm, n_rows, n_cols);
  thrust::device_vector<int> y_dev(y.begin(), y.end());
  std::vector<float> wf(w.begin(), w.end());
  thrust::device_vector<float> w_dev(wf.begin(), wf.end());

  try {
    FitClassifier<float, int>(handle,
                              X_cm,
                              y_dev,
                              n_rows,
                              n_cols,
                              n_classes,
                              WeightedEquivParams(GINI),
                              w_dev.data().get());
    FAIL() << "expected a shared-memory budget exception";
  } catch (const raft::logic_error& e) {
    const std::string msg = e.what();
    EXPECT_TRUE(msg.find("shared-memory") != std::string::npos ||
                msg.find("max_n_bins") != std::string::npos ||
                msg.find("n_classes") != std::string::npos)
      << "unexpected message: " << msg;
  }
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
 protected:
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
        count += cdf_hist[params.max_n_bins * c + idx].x;
      } else  // aggregatebin
      {
        count += cdf_hist[params.max_n_bins * c + idx].count;
      }
    }
    return count;
  }

  void SetUp() override
  {
    // Guarded on the unweighted bin type: the weighted ObjectiveT still
    // instantiates this inherited virtual, so it must compile-discard there
    // (the 5-arg GainPerSplit / unweighted GenHist do not exist for it).
    if constexpr (std::is_same_v<BinT, CountBin> || std::is_same_v<BinT, AggregateBin>) {
      params = ::testing::TestWithParam<ObjectiveTestParameters>::GetParam();
      srand(params.seed);
      ObjectiveT objective(params.n_classes, params.min_samples_leaf);

      auto data                 = GenRandomData();
      auto [cdf_hist, pdf_hist] = GenHist(data);
      auto split_bin_index      = RandUnder(params.max_n_bins);
      auto ground_truth_gain    = GroundTruthGain(data, split_bin_index);

      auto hypothesis_gain = objective.GainPerSplit(&cdf_hist[0],
                                                    split_bin_index,
                                                    params.max_n_bins,
                                                    NumLeftOfBin(cdf_hist, params.max_n_bins - 1),
                                                    NumLeftOfBin(cdf_hist, split_bin_index));

      // The gain may actually be NaN. If so, a comparison between the result
      // and ground truth would yield false, even if both are (correctly) NaN.
      if (!std::isnan(ground_truth_gain) || !std::isnan(hypothesis_gain)) {
        ASSERT_NEAR(ground_truth_gain, hypothesis_gain, params.tolerance);
      }
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

// Derives from ObjectiveTest to reuse its unweighted ground-truth helpers,
// but owns SetUp (base's 5-arg GainPerSplit path is incompatible with the
// weighted 6-arg one) and seeds exactly one RNG, so there is no double-seed.
template <typename ObjectiveT>
class WeightedObjectiveTest : public ObjectiveTest<ObjectiveT> {
  using Base  = ObjectiveTest<ObjectiveT>;
  using DataT = typename Base::DataT;
  using IdxT  = typename ObjectiveT::IdxT;
  using BinT  = typename ObjectiveT::BinT;

  std::vector<DataT> GenWeightedData(std::mt19937& rng)
  {
    std::vector<DataT> data(this->params.n_rows);
    if constexpr (std::is_same_v<BinT, WeightedCountBin>) {
      std::uniform_int_distribution<int> uc(0, this->params.n_classes - 1);
      for (auto& d : data)
        d = DataT(uc(rng));
    } else {
      std::normal_distribution<double> nd(1.0, 2.0);
      for (auto& d : data) {
        double v;
        do {
          v = nd(rng);
        } while (v <= 0);
        d = DataT(v);
      }
    }
    return data;
  }

  std::vector<BinT> GenWeightedHist(const std::vector<DataT>& data,
                                    const std::vector<DataT>& weights)
  {
    std::vector<BinT> cdf, pdf;
    IdxT bin_width = raft::ceildiv(this->params.n_rows, this->params.max_n_bins);
    for (int c = 0; c < this->params.n_classes; ++c) {
      for (int b = 0; b < this->params.max_n_bins; ++b) {
        IdxT begin = b * bin_width;
        IdxT end   = std::min<IdxT>(begin + bin_width, this->params.n_rows);
        BinT bin;
        for (IdxT k = begin; k < end; ++k) {
          double w = double(weights[k]);
          if constexpr (std::is_same_v<BinT, WeightedCountBin>) {
            if (int(data[k]) == c) {
              bin.weighted_sum += w;
              bin.count += 1;
            }
          } else {
            bin.label_sum += w * double(data[k]);
            bin.weighted_count += w;
            bin.count += 1;
          }
        }
        auto cumulative = b > 0 ? cdf.back() : BinT();
        pdf.push_back(bin);
        cdf.push_back(bin);
        cdf.back() += cumulative;
      }
    }
    return cdf;
  }

  double NumWeightedLeftOfBin(const std::vector<BinT>& cdf, IdxT idx)
  {
    double w = 0.0;
    for (int c = 0; c < this->params.n_classes; ++c) {
      const BinT& bin = cdf[this->params.max_n_bins * c + idx];
      if constexpr (std::is_same_v<BinT, WeightedCountBin>)
        w += bin.weighted_sum;
      else
        w += bin.weighted_count;
    }
    return w;
  }

  // Weighted Gini reference: counts replaced by weighted sums.
  double WeightedGiniGroundTruthGain(const std::vector<DataT>& data,
                                     const std::vector<DataT>& weights,
                                     std::size_t split_bin)
  {
    IdxT bin_width  = raft::ceildiv(this->params.n_rows, this->params.max_n_bins);
    std::size_t cut = std::min<std::size_t>((split_bin + 1) * bin_width, data.size());
    auto impurity   = [&](std::size_t lo, std::size_t hi) {
      double tot = 0.0;
      std::vector<double> per(this->params.n_classes, 0.0);
      for (std::size_t k = lo; k < hi; ++k) {
        per[int(data[k])] += double(weights[k]);
        tot += double(weights[k]);
      }
      double g = 0.0;
      for (double p : per) {
        double pr = tot > 0 ? p / tot : 0.0;
        g += pr * (1 - pr);
      }
      return std::make_pair(g, tot);
    };
    auto [gp, wt] = impurity(0, data.size());
    auto [gl, wl] = impurity(0, cut);
    auto [gr, wr] = impurity(cut, data.size());
    if (wl <= 0 || wr <= 0) return -std::numeric_limits<double>::max();
    return gp - (wl / wt) * gl - (wr / wt) * gr;
  }

  // Weighted Entropy reference: counts replaced by weighted sums.
  double WeightedEntropyGroundTruthGain(const std::vector<DataT>& data,
                                        const std::vector<DataT>& weights,
                                        std::size_t split_bin)
  {
    IdxT bin_width  = raft::ceildiv(this->params.n_rows, this->params.max_n_bins);
    std::size_t cut = std::min<std::size_t>((split_bin + 1) * bin_width, data.size());
    auto entropy    = [&](std::size_t lo, std::size_t hi) {
      double tot = 0.0;
      std::vector<double> per(this->params.n_classes, 0.0);
      for (std::size_t k = lo; k < hi; ++k) {
        per[int(data[k])] += double(weights[k]);
        tot += double(weights[k]);
      }
      double h = 0.0;
      for (double p : per) {
        double pr = tot > 0 ? p / tot : 0.0;
        if (pr > 0) h += -pr * std::log(pr) / std::log(2.0);
      }
      return std::make_pair(h, tot);
    };
    auto [hp, wt] = entropy(0, data.size());
    auto [hl, wl] = entropy(0, cut);
    auto [hr, wr] = entropy(cut, data.size());
    if (wl <= 0 || wr <= 0) return -std::numeric_limits<double>::max();
    return hp - (wl / wt) * hl - (wr / wt) * hr;
  }

  // Weighted MSE reference, mirroring WeightedMSEObjectiveFunction exactly.
  double WeightedMSEGroundTruthGain(const std::vector<DataT>& data,
                                    const std::vector<DataT>& weights,
                                    std::size_t split_bin)
  {
    IdxT bin_width  = raft::ceildiv(this->params.n_rows, this->params.max_n_bins);
    std::size_t cut = std::min<std::size_t>((split_bin + 1) * bin_width, data.size());
    auto acc        = [&](std::size_t lo, std::size_t hi) {
      double sw = 0.0, swy = 0.0;
      for (std::size_t k = lo; k < hi; ++k) {
        sw += double(weights[k]);
        swy += double(weights[k]) * double(data[k]);
      }
      return std::make_pair(sw, swy);
    };
    auto [swT, swyT] = acc(0, data.size());
    auto [swL, swyL] = acc(0, cut);
    auto [swR, swyR] = acc(cut, data.size());
    if (swL <= 0 || swR <= 0) return -std::numeric_limits<double>::max();
    double parent = -swyT * swyT / swT;
    double left   = -swyL * swyL / swL;
    double right  = -swyR * swyR / swR;
    return (parent - (left + right)) * 0.5 / swT;
  }

 public:
  void SetUp() override
  {
    this->params = ::testing::TestWithParam<ObjectiveTestParameters>::GetParam();
    std::mt19937 rng(this->params.seed);
    ObjectiveT objective(this->params.n_classes, this->params.min_samples_leaf);

    auto data = GenWeightedData(rng);
    std::uniform_real_distribution<double> uw(0.5, 2.0);
    std::vector<DataT> weights(this->params.n_rows), ones(this->params.n_rows, DataT(1));
    for (auto& w : weights)
      w = DataT(uw(rng));

    std::uniform_int_distribution<int> split_dist(0, this->params.max_n_bins - 1);
    auto split_bin = split_dist(rng);

    // (1) Non-unit weighted hypothesis vs an independent weighted reference.
    // Poisson/Gamma/InverseGaussian reuse the WeightedMSE scaffold and are
    // covered by the (2) unit-weight anchor below, not a separate reference.
    auto cdf_w = GenWeightedHist(data, weights);
    IdxT len   = this->NumLeftOfBin(cdf_w, this->params.max_n_bins - 1);
    IdxT nLeft = this->NumLeftOfBin(cdf_w, split_bin);
    double wL  = NumWeightedLeftOfBin(cdf_w, split_bin);
    DataT hyp_w =
      objective.GainPerSplit(&cdf_w[0], split_bin, this->params.max_n_bins, len, nLeft, wL);
    if constexpr (std::is_same_v<ObjectiveT, WeightedGiniObjectiveFunction<DataT, int, int>>) {
      double gt = WeightedGiniGroundTruthGain(data, weights, split_bin);
      if (!std::isnan(gt) && !std::isnan(double(hyp_w)) &&
          gt != -std::numeric_limits<double>::max())
        ASSERT_NEAR(gt, double(hyp_w), this->params.tolerance);
    } else if constexpr (std::is_same_v<ObjectiveT,
                                        WeightedEntropyObjectiveFunction<DataT, int, int>>) {
      double gt = WeightedEntropyGroundTruthGain(data, weights, split_bin);
      if (!std::isnan(gt) && !std::isnan(double(hyp_w)) &&
          gt != -std::numeric_limits<double>::max())
        ASSERT_NEAR(gt, double(hyp_w), this->params.tolerance);
    } else if constexpr (std::is_same_v<ObjectiveT,
                                        WeightedMSEObjectiveFunction<DataT, DataT, int>>) {
      double gt = WeightedMSEGroundTruthGain(data, weights, split_bin);
      if (!std::isnan(gt) && !std::isnan(double(hyp_w)) &&
          gt != -std::numeric_limits<double>::max())
        ASSERT_NEAR(gt, double(hyp_w), this->params.tolerance);
    }

    // (2) unit-weight anchor: weighted GainPerSplit at w=1 must match the
    // existing unweighted reference for every objective.
    auto cdf_1  = GenWeightedHist(data, ones);
    IdxT len1   = this->NumLeftOfBin(cdf_1, this->params.max_n_bins - 1);
    IdxT nLeft1 = this->NumLeftOfBin(cdf_1, split_bin);
    double wL1  = NumWeightedLeftOfBin(cdf_1, split_bin);
    DataT hyp_1 =
      objective.GainPerSplit(&cdf_1[0], split_bin, this->params.max_n_bins, len1, nLeft1, wL1);
    DataT gt_u;
    if constexpr (std::is_same_v<ObjectiveT, WeightedGiniObjectiveFunction<DataT, int, int>>)
      gt_u = this->GiniGroundTruthGain(data, split_bin);
    else if constexpr (std::is_same_v<ObjectiveT,
                                      WeightedEntropyObjectiveFunction<DataT, int, int>>)
      gt_u = this->EntropyGroundTruthGain(data, split_bin);
    else if constexpr (std::is_same_v<ObjectiveT, WeightedMSEObjectiveFunction<DataT, DataT, int>>)
      gt_u = this->MSEGroundTruthGain(data, split_bin);
    else if constexpr (std::is_same_v<ObjectiveT,
                                      WeightedPoissonObjectiveFunction<DataT, DataT, int>>)
      gt_u = this->PoissonGroundTruthGain(data, split_bin);
    else if constexpr (std::is_same_v<ObjectiveT,
                                      WeightedGammaObjectiveFunction<DataT, DataT, int>>)
      gt_u = this->GammaGroundTruthGain(data, split_bin);
    else
      gt_u = this->InverseGaussianGroundTruthGain(data, split_bin);
    if (!std::isnan(double(gt_u)) && !std::isnan(double(hyp_1)) &&
        gt_u != -std::numeric_limits<DataT>::max())
      ASSERT_NEAR(double(gt_u), double(hyp_1), this->params.tolerance);
  }
};

typedef WeightedObjectiveTest<WeightedGiniObjectiveFunction<double, int, int>>
  WeightedGiniObjectiveTestD;
TEST_P(WeightedGiniObjectiveTestD, weightedGiniObjective) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        WeightedGiniObjectiveTestD,
                        ::testing::ValuesIn(gini_objective_test_parameters));
typedef WeightedObjectiveTest<WeightedGiniObjectiveFunction<float, int, int>>
  WeightedGiniObjectiveTestF;
TEST_P(WeightedGiniObjectiveTestF, weightedGiniObjective) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        WeightedGiniObjectiveTestF,
                        ::testing::ValuesIn(gini_objective_test_parameters));

typedef WeightedObjectiveTest<WeightedEntropyObjectiveFunction<double, int, int>>
  WeightedEntropyObjectiveTestD;
TEST_P(WeightedEntropyObjectiveTestD, weightedEntropyObjective) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        WeightedEntropyObjectiveTestD,
                        ::testing::ValuesIn(entropy_objective_test_parameters));
typedef WeightedObjectiveTest<WeightedEntropyObjectiveFunction<float, int, int>>
  WeightedEntropyObjectiveTestF;
TEST_P(WeightedEntropyObjectiveTestF, weightedEntropyObjective) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        WeightedEntropyObjectiveTestF,
                        ::testing::ValuesIn(entropy_objective_test_parameters));

typedef WeightedObjectiveTest<WeightedMSEObjectiveFunction<double, double, int>>
  WeightedMSEObjectiveTestD;
TEST_P(WeightedMSEObjectiveTestD, weightedMSEObjective) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        WeightedMSEObjectiveTestD,
                        ::testing::ValuesIn(mse_objective_test_parameters));
typedef WeightedObjectiveTest<WeightedMSEObjectiveFunction<float, float, int>>
  WeightedMSEObjectiveTestF;
TEST_P(WeightedMSEObjectiveTestF, weightedMSEObjective) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        WeightedMSEObjectiveTestF,
                        ::testing::ValuesIn(mse_objective_test_parameters));

typedef WeightedObjectiveTest<WeightedPoissonObjectiveFunction<double, double, int>>
  WeightedPoissonObjectiveTestD;
TEST_P(WeightedPoissonObjectiveTestD, weightedPoissonObjective) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        WeightedPoissonObjectiveTestD,
                        ::testing::ValuesIn(poisson_objective_test_parameters));
typedef WeightedObjectiveTest<WeightedPoissonObjectiveFunction<float, float, int>>
  WeightedPoissonObjectiveTestF;
TEST_P(WeightedPoissonObjectiveTestF, weightedPoissonObjective) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        WeightedPoissonObjectiveTestF,
                        ::testing::ValuesIn(poisson_objective_test_parameters));

typedef WeightedObjectiveTest<WeightedGammaObjectiveFunction<double, double, int>>
  WeightedGammaObjectiveTestD;
TEST_P(WeightedGammaObjectiveTestD, weightedGammaObjective) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        WeightedGammaObjectiveTestD,
                        ::testing::ValuesIn(gamma_objective_test_parameters));
typedef WeightedObjectiveTest<WeightedGammaObjectiveFunction<float, float, int>>
  WeightedGammaObjectiveTestF;
TEST_P(WeightedGammaObjectiveTestF, weightedGammaObjective) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        WeightedGammaObjectiveTestF,
                        ::testing::ValuesIn(gamma_objective_test_parameters));

typedef WeightedObjectiveTest<WeightedInverseGaussianObjectiveFunction<double, double, int>>
  WeightedInverseGaussianObjectiveTestD;
TEST_P(WeightedInverseGaussianObjectiveTestD, weightedInverseGaussianObjective) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        WeightedInverseGaussianObjectiveTestD,
                        ::testing::ValuesIn(invgauss_objective_test_parameters));
typedef WeightedObjectiveTest<WeightedInverseGaussianObjectiveFunction<float, float, int>>
  WeightedInverseGaussianObjectiveTestF;
TEST_P(WeightedInverseGaussianObjectiveTestF, weightedInverseGaussianObjective) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        WeightedInverseGaussianObjectiveTestF,
                        ::testing::ValuesIn(invgauss_objective_test_parameters));

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
