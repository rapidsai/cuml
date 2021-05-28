/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <raft/handle.hpp>

#include <decisiontree/memory.h>
#include <decisiontree/quantile/quantile.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <common/iota.cuh>
#include <decisiontree/batched-levelalgo/builder_base.cuh>
#include <decisiontree/batched-levelalgo/kernels.cuh>
#include <functional>

namespace ML {
namespace DecisionTree {

struct NodeSplitKernelTestParams {
  int min_samples_split;
  int min_samples_leaf;
  int expected_n_total_nodes;
  int expected_n_new_nodes;
};

struct NoOpParams {};

class BatchedLevelAlgoUnitTestFixture {
 protected:
  using DataT = float;
  using LabelT = float;
  using IdxT = int;
  using NodeT = Node<DataT, LabelT, IdxT>;
  using Traits = RegTraits<DataT, IdxT>;

  const int n_bins = 5;
  const IdxT n_row = 5;
  const IdxT n_col = 2;
  const IdxT max_batch = 8;

  void SetUp() {
    params.max_depth = 2;
    params.max_leaves = 8;
    params.max_features = 1.0f;
    params.n_bins = n_bins;
    params.split_algo = 1;
    params.min_samples_leaf = 0;
    params.min_samples_split = 0;
    params.bootstrap_features = false;
    params.split_criterion = CRITERION::MSE;
    params.min_impurity_decrease = 0.0f;
    params.max_batch_size = 8;
    params.use_experimental_backend = true;

    h_data = {-1.0f, 0.0f, 2.0f, 0.0f, -2.0f,
              0.0f,  1.0f, 0.0f, 3.0f, 0.0f};  // column-major
    h_labels = {-1.0f, 2.0f, 2.0f, 6.0f, -2.0f};
    // X0 + 2 * X1

    raft_handle = std::make_unique<raft::handle_t>();
    auto d_allocator = raft_handle->get_device_allocator();
    auto h_allocator = raft_handle->get_host_allocator();

    data = static_cast<DataT*>(
      d_allocator->allocate(sizeof(DataT) * n_row * n_col, 0));
    labels =
      static_cast<LabelT*>(d_allocator->allocate(sizeof(LabelT) * n_row, 0));
    row_ids =
      static_cast<IdxT*>(d_allocator->allocate(sizeof(IdxT) * n_row, 0));

    // Nodes that exist prior to the invocation of nodeSplitKernel()
    curr_nodes =
      static_cast<NodeT*>(d_allocator->allocate(sizeof(NodeT) * max_batch, 0));
    // Nodes that are created new by the invocation of nodeSplitKernel()
    new_nodes = static_cast<NodeT*>(
      d_allocator->allocate(sizeof(NodeT) * 2 * max_batch, 0));
    // Number of nodes and leaves that are created new by the invocation of
    // nodeSplitKernel()
    n_new_nodes = static_cast<IdxT*>(d_allocator->allocate(sizeof(IdxT), 0));
    n_new_leaves = static_cast<IdxT*>(d_allocator->allocate(sizeof(IdxT), 0));
    // New depth reached by the invocation of nodeSplitKernel()
    new_depth = static_cast<IdxT*>(d_allocator->allocate(sizeof(IdxT), 0));

    splits = static_cast<Traits::SplitT*>(
      d_allocator->allocate(sizeof(Traits::SplitT) * max_batch, 0));

    raft::update_device(data, h_data.data(), n_row * n_col, 0);
    raft::update_device(labels, h_labels.data(), n_row, 0);
    MLCommon::iota(row_ids, 0, 1, n_row, 0);

    // computing the quantiles
    d_quantiles =
      (DataT*)d_allocator->allocate(sizeof(DataT) * params.n_bins * n_col, 0);
    h_quantiles =
      (DataT*)h_allocator->allocate(sizeof(DataT) * params.n_bins * n_col, 0);
    computeQuantiles(d_quantiles, params.n_bins, data, n_row, n_col,
                     d_allocator, 0);
    raft::update_host(h_quantiles, d_quantiles, params.n_bins * n_col, 0);
    CUDA_CHECK(cudaStreamSynchronize(0));

    input.data = data;
    input.labels = labels;
    input.M = n_row;
    input.N = n_col;
    input.nSampledRows = n_row;
    input.nSampledCols = n_col;
    input.rowids = row_ids;
    input.nclasses = 0;  // not applicable for regression
    input.quantiles = d_quantiles;
  }

  void TearDown() {
    CUDA_CHECK(cudaStreamSynchronize(0));
    auto d_allocator = raft_handle->get_device_allocator();
    auto h_allocator = raft_handle->get_host_allocator();
    d_allocator->deallocate(data, sizeof(DataT) * n_row * n_col, 0);
    d_allocator->deallocate(labels, sizeof(LabelT) * n_row, 0);
    d_allocator->deallocate(row_ids, sizeof(IdxT) * n_row, 0);
    d_allocator->deallocate(curr_nodes, sizeof(NodeT) * max_batch, 0);
    d_allocator->deallocate(new_nodes, sizeof(NodeT) * 2 * max_batch, 0);
    d_allocator->deallocate(n_new_nodes, sizeof(IdxT), 0);
    d_allocator->deallocate(n_new_leaves, sizeof(IdxT), 0);
    d_allocator->deallocate(new_depth, sizeof(IdxT), 0);
    d_allocator->deallocate(splits, sizeof(Traits::SplitT) * max_batch, 0);
    d_allocator->deallocate(d_quantiles, sizeof(DataT) * n_bins * n_col, 0);
    CUDA_CHECK(cudaStreamSynchronize(0));
    raft_handle.reset();
    h_allocator->deallocate(h_quantiles, sizeof(DataT) * n_bins * n_col, 0);
  }

  DecisionTreeParams params;

  std::unique_ptr<raft::handle_t> raft_handle;

  std::vector<DataT> h_data;
  std::vector<LabelT> h_labels;

  DataT* h_quantiles;
  DataT* d_quantiles;
  Traits::InputT input;

  NodeT* curr_nodes;
  NodeT* new_nodes;
  IdxT* n_new_nodes;
  IdxT* n_new_leaves;
  IdxT* new_depth;
  Traits::SplitT* splits;

  DataT* data;
  DataT* labels;
  IdxT* row_ids;
};

class TestQuantiles : public ::testing::TestWithParam<NoOpParams>,
                      protected BatchedLevelAlgoUnitTestFixture {
 protected:
  void SetUp() override { BatchedLevelAlgoUnitTestFixture::SetUp(); }

  void TearDown() override { BatchedLevelAlgoUnitTestFixture::TearDown(); }
};

class TestNodeSplitKernel
  : public ::testing::TestWithParam<NodeSplitKernelTestParams>,
    protected BatchedLevelAlgoUnitTestFixture {
 protected:
  void SetUp() override { BatchedLevelAlgoUnitTestFixture::SetUp(); }

  void TearDown() override { BatchedLevelAlgoUnitTestFixture::TearDown(); }
};

class TestMetric : public ::testing::TestWithParam<CRITERION>,
                   protected BatchedLevelAlgoUnitTestFixture {
 protected:
  void SetUp() override { BatchedLevelAlgoUnitTestFixture::SetUp(); }

  void TearDown() override { BatchedLevelAlgoUnitTestFixture::TearDown(); }
};

TEST_P(TestQuantiles, Quantiles) {
  /* Ensure that quantiles are computed correctly */
  std::vector<DataT> expected_quantiles[]{{-2.0f, -1.0f, 0.0f, 2.0f},
                                          {0.0f, 1.0f, 3.0f}};
  for (int col = 0; col < n_col; col++) {
    std::vector<DataT> col_quantile(n_bins);
    std::copy(h_quantiles + n_bins * col, h_quantiles + n_bins * (col + 1),
              col_quantile.begin());
    auto last = std::unique(col_quantile.begin(), col_quantile.end());
    col_quantile.erase(last, col_quantile.end());
    EXPECT_EQ(col_quantile, expected_quantiles[col]);
  }
}

INSTANTIATE_TEST_SUITE_P(BatchedLevelAlgoUnitTest, TestQuantiles,
                         ::testing::Values(NoOpParams{}));

TEST_P(TestNodeSplitKernel, MinSamplesSplitLeaf) {
  auto test_params = GetParam();

  Builder<Traits> builder;
  auto smemSize = Traits::nodeSplitSmemSize(builder);

  IdxT h_n_total_nodes = 3;  // total number of nodes created so far
  IdxT h_n_new_nodes;        // number of nodes created in this round
  IdxT batchSize = 2;
  std::vector<NodeT> h_nodes{
    /* {
     *   SparseTreeNode{
     *     prediction, colid, quesval, best_metric_val, left_child_id },
     *   }, start, count, depth
     * } */
    {{1.40f, 0, -0.5f, 5.606667f, 1}, 0, 5, 0},
    {{-1.50f, IdxT(-1), DataT(0), DataT(0), NodeT::Leaf}, 0, 2, 1},
    {{3.333333f, IdxT(-1), DataT(0), DataT(0), NodeT::Leaf}, 1, 3, 1},
  };
  raft::update_device(curr_nodes, h_nodes.data() + 1, batchSize, 0);
  CUDA_CHECK(cudaMemsetAsync(n_new_nodes, 0, sizeof(IdxT), 0));
  CUDA_CHECK(cudaMemsetAsync(n_new_leaves, 0, sizeof(IdxT), 0));
  CUDA_CHECK(cudaMemsetAsync(new_depth, 0, sizeof(IdxT), 0));
  initSplit<DataT, IdxT, Traits::TPB_DEFAULT>(splits, batchSize, 0);

  /* { quesval, colid, best_metric_val, nLeft } */
  std::vector<Traits::SplitT> h_splits{{-1.5f, 0, 0.25f, 1},
                                       {2.0f, 1, 3.555556f, 2}};
  raft::update_device(splits, h_splits.data(), 2, 0);

  nodeSplitKernel<DataT, LabelT, IdxT, Traits::DevTraits, Traits::TPB_SPLIT>
    <<<batchSize, Traits::TPB_SPLIT, smemSize, 0>>>(
      params.max_depth, test_params.min_samples_leaf,
      test_params.min_samples_split, params.max_leaves,
      params.min_impurity_decrease, input, curr_nodes, new_nodes, n_new_nodes,
      splits, n_new_leaves, h_n_total_nodes, new_depth);
  CUDA_CHECK(cudaGetLastError());
  raft::update_host(&h_n_new_nodes, n_new_nodes, 1, 0);
  CUDA_CHECK(cudaStreamSynchronize(0));
  h_n_total_nodes += h_n_new_nodes;
  EXPECT_EQ(h_n_total_nodes, test_params.expected_n_total_nodes);
  EXPECT_EQ(h_n_new_nodes, test_params.expected_n_new_nodes);
}

const std::vector<NodeSplitKernelTestParams> min_samples_split_leaf_test_params{
  /* { min_samples_split, min_samples_leaf,
   *   expected_n_total_nodes, expected_n_new_nodes } */
  {0, 0, 7, 4}, {2, 0, 7, 4}, {3, 0, 5, 2}, {4, 0, 3, 0}, {5, 0, 3, 0},
  {0, 1, 7, 4}, {0, 2, 3, 0}, {0, 5, 3, 0}, {4, 2, 3, 0}, {5, 5, 3, 0}};

INSTANTIATE_TEST_SUITE_P(
  BatchedLevelAlgoUnitTest, TestNodeSplitKernel,
  ::testing::ValuesIn(min_samples_split_leaf_test_params));

TEST_P(TestMetric, RegressionMetricGain) {
  IdxT batchSize = 1;
  std::vector<NodeT> h_nodes{
    /* {
     *   SparseTreeNode{
     *     prediction, colid, quesval, best_metric_val, left_child_id },
     *   }, start, count, depth
     * } */
    {{1.40f, IdxT(-1), DataT(0), DataT(0), NodeT::Leaf}, 0, 5, 0}};
  raft::update_device(curr_nodes, h_nodes.data(), batchSize, 0);

  auto n_col_blks = 1;  // evaluate only one column (feature)

  IdxT nPredCounts = max_batch * n_bins * n_col_blks;

  auto d_allocator = raft_handle->get_device_allocator();

  // mutex array used for atomically updating best split
  int* mutex =
    static_cast<int*>(d_allocator->allocate(sizeof(int) * max_batch, 0));
  // threadblock arrival count
  int* done_count = static_cast<int*>(
    d_allocator->allocate(sizeof(int) * max_batch * n_col_blks, 0));
  DataT* pred = static_cast<DataT*>(
    d_allocator->allocate(2 * nPredCounts * sizeof(DataT), 0));
  IdxT* pred_count =
    static_cast<IdxT*>(d_allocator->allocate(nPredCounts * sizeof(IdxT), 0));

  WorkloadInfo<IdxT>* workload_info = static_cast<WorkloadInfo<IdxT>*>(
    d_allocator->allocate(sizeof(WorkloadInfo<IdxT>), 0));
  WorkloadInfo<IdxT> h_workload_info;

  // Just one threadBlock would be used
  h_workload_info.nodeid = 0;
  h_workload_info.offset_blockid = 0;
  h_workload_info.num_blocks = 1;

  raft::update_device(workload_info, &h_workload_info, 1, 0);

  CUDA_CHECK(cudaMemsetAsync(mutex, 0, sizeof(int) * max_batch, 0));
  CUDA_CHECK(
    cudaMemsetAsync(done_count, 0, sizeof(int) * max_batch * n_col_blks, 0));
  CUDA_CHECK(cudaMemsetAsync(pred, 0, 2 * sizeof(DataT) * nPredCounts, 0));
  CUDA_CHECK(cudaMemsetAsync(pred_count, 0, nPredCounts * sizeof(IdxT), 0));
  CUDA_CHECK(cudaMemsetAsync(n_new_leaves, 0, sizeof(IdxT), 0));
  initSplit<DataT, IdxT, Traits::TPB_DEFAULT>(splits, batchSize, 0);

  std::vector<Traits::SplitT> h_splits(1);

  CRITERION split_criterion = GetParam();

  dim3 grid(1, n_col_blks, 1);
  // Compute shared memory size
  size_t smemSize1 = (n_bins + 1) * sizeof(DataT) +  // pdf_spred
                     2 * n_bins * sizeof(DataT) +    // cdf_spred
                     n_bins * sizeof(int) +          // pdf_scount
                     n_bins * sizeof(int) +          // cdf_scount
                     n_bins * sizeof(DataT) +        // sbins
                     sizeof(int);                    // sDone
  // Room for alignment (see alignPointer in computeSplitRegressionkernels)
  smemSize1 += 6 * sizeof(DataT) + 3 * sizeof(int);
  // Calculate the shared memory needed for evalBestSplit
  size_t smemSize2 =
    raft::ceildiv(32, raft::WarpSize) * sizeof(Split<DataT, IdxT>);
  // Pick the max of two
  size_t smemSize = std::max(smemSize1, smemSize2);

  computeSplitRegressionKernel<DataT, DataT, IdxT, 32>
    <<<grid, 32, smemSize, 0>>>(
      pred, pred_count, n_bins, params.min_samples_leaf,
      params.min_impurity_decrease, input, curr_nodes, 0, done_count, mutex,
      splits, split_criterion, 0, workload_info, 1234ULL);

  raft::update_host(h_splits.data(), splits, 1, 0);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(0));

  // the split uses feature 0
  // rows 0, 4 go to the left side of the threshold
  // rows 1, 2, 3 go to the right side of the threshold
  EXPECT_EQ(h_splits[0].colid, 0);
  EXPECT_EQ(h_splits[0].nLeft, 2);
  for (int row_id : {0, 4}) {
    EXPECT_LE(h_data[0 * n_row + row_id], h_splits[0].quesval);
  }
  for (int row_id : {1, 2, 3}) {
    EXPECT_GT(h_data[0 * n_row + row_id], h_splits[0].quesval);
  }
  // Verify that the gain (reduction in MSE / MAE) is computed correctly
  std::function<float(const std::vector<DataT>&, const std::vector<IdxT>&)>
    metric;
  if (split_criterion == CRITERION::MSE) {
    metric = [](const std::vector<DataT>& y,
                const std::vector<IdxT>& idx) -> float {
      float y_mean = 0.0f;
      float mse = 0.0f;
      for (IdxT i : idx) {
        y_mean += y[i];
      }
      y_mean /= idx.size();
      for (IdxT i : idx) {
        mse += (y[i] - y_mean) * (y[i] - y_mean);
      }
      return mse / idx.size();
    };
  } else {
    EXPECT_EQ(split_criterion, CRITERION::MAE);
    metric = [](const std::vector<DataT>& y,
                const std::vector<IdxT>& idx) -> float {
      float y_mean = 0.0f;
      float mae = 0.0f;
      for (IdxT i : idx) {
        y_mean += y[i];
      }
      y_mean /= idx.size();
      for (IdxT i : idx) {
        mae += std::fabs(y[i] - y_mean);
      }
      return mae / idx.size();
    };
  }
  float expected_gain = metric(h_labels, {0, 1, 2, 3, 4}) -
                        2.0f / 5.0f * metric(h_labels, {0, 4}) -
                        3.0f / 5.0f * metric(h_labels, {1, 2, 3});

  EXPECT_FLOAT_EQ(h_splits[0].best_metric_val, expected_gain);

  d_allocator->deallocate(mutex, sizeof(int) * max_batch, 0);
  d_allocator->deallocate(done_count, sizeof(int) * max_batch * n_col_blks, 0);
  d_allocator->deallocate(pred, 2 * nPredCounts * sizeof(DataT), 0);
  d_allocator->deallocate(pred_count, nPredCounts * sizeof(IdxT), 0);
  d_allocator->deallocate(workload_info, sizeof(WorkloadInfo<IdxT>), 0);
}

INSTANTIATE_TEST_SUITE_P(BatchedLevelAlgoUnitTest, TestMetric,
                         ::testing::Values(CRITERION::MSE),
                         [](const auto& info) {
                           switch (info.param) {
                             case CRITERION::MSE:
                               return "MSE";
                             case CRITERION::MAE:
                               return "MAE";
                             default:
                               return "";
                           }
                         });

}  // namespace DecisionTree
}  // namespace ML
