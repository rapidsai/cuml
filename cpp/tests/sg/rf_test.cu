/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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
#include <cuml/common/logger.hpp>
#include <cuml/datasets/make_blobs.hpp>
#include <cuml/ensemble/randomforest.hpp>
#include <cuml/fil/detail/raft_proto/device_type.hpp>
#include <cuml/fil/infer_kind.hpp>
#include <cuml/fil/tree_layout.hpp>
#include <cuml/fil/treelite_importer.hpp>
#include <cuml/tree/algo_helper.h>

#include <raft/core/handle.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cub/device/device_segmented_reduce.cuh>
#include <cuda/std/functional>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>
#include <thrust/transform.h>

#include <decisiontree/batched-levelalgo/kernels/builder_kernels.cuh>
#include <decisiontree/batched-levelalgo/quantiles.cuh>
#include <gtest/gtest.h>
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
std::shared_ptr<thrust::device_vector<LabelT>> FilPredict(
  const raft::handle_t& handle,
  RfTestParams params,
  DataT* X_transpose,
  RandomForestMetaData<DataT, LabelT>* forest)
{
  auto pred      = std::shared_ptr<thrust::device_vector<LabelT>>();
  auto workspace = std::shared_ptr<thrust::device_vector<DataT>>();  // Scratch space
  if constexpr (std::is_integral_v<LabelT>) {
    // For classifiers, allocate extra scratch space to store probabilities from FIL
    // We will perform argmax to convert probabilities into class outputs.
    pred      = std::make_shared<thrust::device_vector<LabelT>>(params.n_rows);
    workspace = std::make_shared<thrust::device_vector<DataT>>(params.n_rows * params.n_labels);
  } else {
    // For regressors, no need to post-process predictions from FIL
    static_assert(std::is_same_v<LabelT, DataT>,
                  "LabelT and DataT must be identical for regression task");
    pred      = std::make_shared<thrust::device_vector<LabelT>>(params.n_rows);
    workspace = pred;
  }
  TreeliteModelHandle model;
  build_treelite_forest(&model, forest, params.n_cols);

  auto fil_model = ML::fil::import_from_treelite_handle(model,
                                                        ML::fil::tree_layout::breadth_first,
                                                        128,
                                                        std::is_same_v<DataT, double>,
                                                        raft_proto::device_type::gpu,
                                                        handle.get_device(),
                                                        handle.get_next_usable_stream());
  handle.sync_stream();
  handle.sync_stream_pool();
  delete static_cast<treelite::Model*>(model);

  fil_model.predict(handle,
                    workspace->data().get(),
                    X_transpose,
                    params.n_rows,
                    raft_proto::device_type::gpu,
                    raft_proto::device_type::gpu,
                    ML::fil::infer_kind::default_kind,
                    1);
  handle.sync_stream();
  handle.sync_stream_pool();

  if constexpr (std::is_integral_v<LabelT>) {
    // Perform argmax to convert probabilities into class outputs
    auto offsets_it =
      thrust::make_transform_iterator(thrust::make_counting_iterator(0),
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
auto FilPredictProba(const raft::handle_t& handle,
                     RfTestParams params,
                     DataT* X_transpose,
                     RandomForestMetaData<DataT, LabelT>* forest)
{
  static_assert(std::is_integral_v<LabelT>, "Must be classification");

  std::size_t num_outputs = params.n_labels;
  auto pred = std::make_shared<thrust::device_vector<float>>(params.n_rows * num_outputs);
  TreeliteModelHandle model;
  build_treelite_forest(&model, forest, params.n_cols);

  auto fil_model = ML::fil::import_from_treelite_handle(model,
                                                        ML::fil::tree_layout::breadth_first,
                                                        128,
                                                        std::is_same_v<DataT, double>,
                                                        raft_proto::device_type::gpu,
                                                        handle.get_device(),
                                                        handle.get_next_usable_stream());
  handle.sync_stream();
  handle.sync_stream_pool();
  delete static_cast<treelite::Model*>(model);

  fil_model.predict(handle,
                    pred->data().get(),
                    X_transpose,
                    params.n_rows,
                    raft_proto::device_type::gpu,
                    raft_proto::device_type::gpu,
                    ML::fil::infer_kind::default_kind,
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

  // Compare fil against native rf predictions
  // Only for single precision models
  void TestFilPredict()
  {
    if constexpr (std::is_same_v<DataT, double>) {
      return;
    } else {
      auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(params.n_streams);
      raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
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

  // See if FIL overflows
  thrust::device_vector<float> pred(m);
  TreeliteModelHandle model;
  build_treelite_forest(&model, forest_ptr, n);

  auto fil_model = ML::fil::import_from_treelite_handle(model,
                                                        ML::fil::tree_layout::breadth_first,
                                                        128,
                                                        false,
                                                        raft_proto::device_type::gpu,
                                                        handle.get_device(),
                                                        handle.get_next_usable_stream());
  handle.sync_stream();
  handle.sync_stream_pool();
  delete static_cast<treelite::Model*>(model);

  fil_model.predict(handle,
                    pred.data().get(),
                    X.data().get(),
                    m,
                    raft_proto::device_type::gpu,
                    raft_proto::device_type::gpu,
                    ML::fil::infer_kind::default_kind,
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
    srand(params.seed);
    params = ::testing::TestWithParam<ObjectiveTestParameters>::GetParam();
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

}  // end namespace DT
}  // end namespace ML
