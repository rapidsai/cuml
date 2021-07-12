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

#include <decisiontree/quantile/quantile.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <common/iota.cuh>
#include <decisiontree/batched-levelalgo/builder_base.cuh>
#include <decisiontree/batched-levelalgo/kernels.cuh>
#include <decisiontree/batched-levelalgo/metrics.cuh>
#include <functional>

namespace ML {
namespace DT {

/*
struct NodeSplitKernelTestParams {
  int min_samples_split;
  int min_samples_leaf;
  int expected_n_total_nodes;
  int expected_n_new_nodes;
};

struct NoOpParams {
};

class BatchedLevelAlgoUnitTestFixture {
 protected:
  using DataT      = float;
  using LabelT     = float;
  using IdxT       = int;
  using NodeT      = Node<DataT, LabelT, IdxT>;
  using SplitT     = Split<DataT, IdxT>;
  using InputT     = Input<DataT, LabelT, IdxT>;
  using ObjectiveT = MSEObjectiveFunction<DataT, LabelT, IdxT>;

  const int n_bins                 = 5;
  const IdxT n_row                 = 5;
  const IdxT n_col                 = 2;
  const IdxT max_batch             = 8;
  static constexpr int TPB_DEFAULT = 256;
  static constexpr int TPB_SPLIT   = 128;

  void SetUp()
  {
    params.max_depth             = 2;
    params.max_leaves            = 8;
    params.max_features          = 1.0f;
    params.n_bins                = n_bins;
    params.min_samples_leaf      = 0;
    params.min_samples_split     = 0;
    params.split_criterion       = CRITERION::MSE;
    params.min_impurity_decrease = 0.0f;
    params.max_batch_size        = 8;

    h_data   = {-1.0f, 0.0f, 2.0f, 0.0f, -2.0f, 0.0f, 1.0f, 0.0f, 3.0f, 0.0f};  // column-major
    h_labels = {-1.0f, 2.0f, 2.0f, 6.0f, -2.0f};
    // X0 + 2 * X1

    raft_handle      = std::make_unique<raft::handle_t>();
    auto d_allocator = raft_handle->get_device_allocator();

    data        = static_cast<DataT*>(d_allocator->allocate(sizeof(DataT) * n_row * n_col, 0));
    d_quantiles = static_cast<DataT*>(d_allocator->allocate(sizeof(DataT) * n_bins * n_col, 0));
    labels      = static_cast<LabelT*>(d_allocator->allocate(sizeof(LabelT) * n_row, 0));
    row_ids     = static_cast<IdxT*>(d_allocator->allocate(sizeof(IdxT) * n_row, 0));

    // Nodes that exist prior to the invocation of nodeSplitKernel()
    curr_nodes = static_cast<NodeT*>(d_allocator->allocate(sizeof(NodeT) * max_batch, 0));
    // Nodes that are created new by the invocation of nodeSplitKernel()
    new_nodes = static_cast<NodeT*>(d_allocator->allocate(sizeof(NodeT) * 2 * max_batch, 0));
    // Number of nodes and leaves that are created new by the invocation of
    // nodeSplitKernel()
    n_new_nodes  = static_cast<IdxT*>(d_allocator->allocate(sizeof(IdxT), 0));
    n_new_leaves = static_cast<IdxT*>(d_allocator->allocate(sizeof(IdxT), 0));
    // New depth reached by the invocation of nodeSplitKernel()
    new_depth = static_cast<IdxT*>(d_allocator->allocate(sizeof(IdxT), 0));

    splits = static_cast<SplitT*>(d_allocator->allocate(sizeof(SplitT) * max_batch, 0));

    raft::update_device(data, h_data.data(), n_row * n_col, 0);
    raft::update_device(labels, h_labels.data(), n_row, 0);
    computeQuantiles(d_quantiles, n_bins, data, n_row, n_col, d_allocator, nullptr);
    MLCommon::iota(row_ids, 0, 1, n_row, 0);

    CUDA_CHECK(cudaStreamSynchronize(0));

    input.data         = data;
    input.labels       = labels;
    input.M            = n_row;
    input.N            = n_col;
    input.nSampledRows = n_row;
    input.nSampledCols = n_col;
    input.rowids       = row_ids;
    input.numOutputs   = 1;
    input.quantiles    = d_quantiles;
  }

  void TearDown()
  {
    auto d_allocator = raft_handle->get_device_allocator();
    d_allocator->deallocate(data, sizeof(DataT) * n_row * n_col, 0);
    d_allocator->deallocate(d_quantiles, sizeof(DataT) * n_bins * n_col, 0);
    d_allocator->deallocate(labels, sizeof(LabelT) * n_row, 0);
    d_allocator->deallocate(row_ids, sizeof(IdxT) * n_row, 0);
    d_allocator->deallocate(curr_nodes, sizeof(NodeT) * max_batch, 0);
    d_allocator->deallocate(new_nodes, sizeof(NodeT) * 2 * max_batch, 0);
    d_allocator->deallocate(n_new_nodes, sizeof(IdxT), 0);
    d_allocator->deallocate(n_new_leaves, sizeof(IdxT), 0);
    d_allocator->deallocate(new_depth, sizeof(IdxT), 0);
    d_allocator->deallocate(splits, sizeof(SplitT) * max_batch, 0);
  }

  DecisionTreeParams params;

  std::unique_ptr<raft::handle_t> raft_handle;

  std::vector<DataT> h_data;
  std::vector<LabelT> h_labels;

  DataT* d_quantiles;
  InputT input;

  NodeT* curr_nodes;
  NodeT* new_nodes;
  IdxT* n_new_nodes;
  IdxT* n_new_leaves;
  IdxT* new_depth;
  SplitT* splits;

  DataT* data;
  DataT* labels;
  IdxT* row_ids;
};

class TestNodeSplitKernel : public ::testing::TestWithParam<NodeSplitKernelTestParams>,
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

TEST_P(TestNodeSplitKernel, MinSamplesSplitLeaf)
{
  auto test_params = GetParam();

  Builder<ObjectiveT> builder;
  builder.input = input;
  auto smemSize = builder.nodeSplitSmemSize();

  IdxT h_n_total_nodes = 3;  // total number of nodes created so far
  IdxT h_n_new_nodes;        // number of nodes created in this round
  IdxT batchSize = 2;
  std::vector<NodeT> h_nodes{
    {{1.40f, 0, -0.5f, 5.606667f, 1}, 0, 5, 0},
    {{-1.50f, IdxT(-1), DataT(0), DataT(0), NodeT::Leaf}, 0, 2, 1},
    {{3.333333f, IdxT(-1), DataT(0), DataT(0), NodeT::Leaf}, 1, 3, 1},
  };
  raft::update_device(curr_nodes, h_nodes.data() + 1, batchSize, 0);
  CUDA_CHECK(cudaMemsetAsync(n_new_nodes, 0, sizeof(IdxT), 0));
  CUDA_CHECK(cudaMemsetAsync(n_new_leaves, 0, sizeof(IdxT), 0));
  CUDA_CHECK(cudaMemsetAsync(new_depth, 0, sizeof(IdxT), 0));
  initSplit<DataT, IdxT, builder.TPB_DEFAULT>(splits, batchSize, 0);

  std::vector<SplitT> h_splits{{-1.5f, 0, 0.25f, 1}, {2.0f, 1, 3.555556f, 2}};
  raft::update_device(splits, h_splits.data(), 2, 0);

  nodeSplitKernel<DataT, LabelT, IdxT, ObjectiveT, builder.TPB_SPLIT>
    <<<batchSize, builder.TPB_SPLIT, smemSize, 0>>>(params.max_depth,
                                                    test_params.min_samples_leaf,
                                                    test_params.min_samples_split,
                                                    params.max_leaves,
                                                    params.min_impurity_decrease,
                                                    input,
                                                    curr_nodes,
                                                    new_nodes,
                                                    n_new_nodes,
                                                    splits,
                                                    n_new_leaves,
                                                    h_n_total_nodes,
                                                    new_depth);
  CUDA_CHECK(cudaGetLastError());
  raft::update_host(&h_n_new_nodes, n_new_nodes, 1, 0);
  CUDA_CHECK(cudaStreamSynchronize(0));
  h_n_total_nodes += h_n_new_nodes;
  EXPECT_EQ(h_n_total_nodes, test_params.expected_n_total_nodes);
  EXPECT_EQ(h_n_new_nodes, test_params.expected_n_new_nodes);
}

const std::vector<NodeSplitKernelTestParams> min_samples_split_leaf_test_params{
  {0, 0, 7, 4},
  {2, 0, 7, 4},
  {3, 0, 5, 2},
  {4, 0, 3, 0},
  {5, 0, 3, 0},
  {0, 1, 7, 4},
  {0, 2, 3, 0},
  {0, 5, 3, 0},
  {4, 2, 3, 0},
  {5, 5, 3, 0}};

INSTANTIATE_TEST_SUITE_P(BatchedLevelAlgoUnitTest,
                         TestNodeSplitKernel,
                         ::testing::ValuesIn(min_samples_split_leaf_test_params));

TEST_P(TestMetric, RegressionMetricGain)
{
  IdxT batchSize = 1;
  std::vector<NodeT> h_nodes{
                             {{1.40f, IdxT(-1), DataT(0), DataT(0), NodeT::Leaf}, 0, 5, 0}};
  raft::update_device(curr_nodes, h_nodes.data(), batchSize, 0);

  auto n_col_blks = 1;  // evaluate only one column (feature)

  IdxT nPredCounts = max_batch * n_bins * n_col_blks;

  auto d_allocator = raft_handle->get_device_allocator();

  // mutex array used for atomically updating best split
  int* mutex = static_cast<int*>(d_allocator->allocate(sizeof(int) * max_batch, 0));
  // threadblock arrival count
  int* done_count =
    static_cast<int*>(d_allocator->allocate(sizeof(int) * max_batch * n_col_blks, 0));
  ObjectiveT::BinT* hist = static_cast<ObjectiveT::BinT*>(
    d_allocator->allocate(2 * nPredCounts * sizeof(ObjectiveT::BinT), 0));

  WorkloadInfo<IdxT>* workload_info =
    static_cast<WorkloadInfo<IdxT>*>(d_allocator->allocate(sizeof(WorkloadInfo<IdxT>), 0));
  WorkloadInfo<IdxT> h_workload_info;

  // Just one threadBlock would be used
  h_workload_info.nodeid         = 0;
  h_workload_info.offset_blockid = 0;
  h_workload_info.num_blocks     = 1;

  raft::update_device(workload_info, &h_workload_info, 1, 0);
  CUDA_CHECK(cudaMemsetAsync(mutex, 0, sizeof(int) * max_batch, 0));
  CUDA_CHECK(cudaMemsetAsync(done_count, 0, sizeof(int) * max_batch * n_col_blks, 0));
  CUDA_CHECK(cudaMemsetAsync(hist, 0, 2 * sizeof(DataT) * nPredCounts, 0));
  CUDA_CHECK(cudaMemsetAsync(n_new_leaves, 0, sizeof(IdxT), 0));
  initSplit<DataT, IdxT, TPB_DEFAULT>(splits, batchSize, 0);

  std::vector<SplitT> h_splits(1);

  CRITERION split_criterion = GetParam();

  ObjectiveT obj(1, params.min_impurity_decrease, params.min_samples_leaf);
  size_t smemSize1 = n_bins * sizeof(ObjectiveT::BinT) +  // pdf_shist size
                     n_bins * sizeof(ObjectiveT::BinT) +  // cdf_shist size
                     n_bins * sizeof(DataT) +             // sbins size
                     sizeof(int);                         // sDone size
  // Extra room for alignment (see alignPointer in
  // computeSplitClassificationKernel)
  smemSize1 += sizeof(DataT) + 3 * sizeof(int);
  // Calculate the shared memory needed for evalBestSplit
  size_t smemSize2 = raft::ceildiv(TPB_DEFAULT, raft::WarpSize) * sizeof(SplitT);
  // Pick the max of two
  size_t smemSize = std::max(smemSize1, smemSize2);

  dim3 grid(1, n_col_blks, 1);
  computeSplitKernel<DataT, LabelT, IdxT, 32><<<grid, 32, smemSize, 0>>>(hist,
                                                                         n_bins,
                                                                         params.max_depth,
                                                                         params.min_samples_split,
                                                                         params.max_leaves,
                                                                         input,
                                                                         curr_nodes,
                                                                         0,
                                                                         done_count,
                                                                         mutex,
                                                                         splits,
                                                                         obj,
                                                                         0,
                                                                         workload_info,
                                                                         1234ULL);

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
  std::function<float(const std::vector<DataT>&, const std::vector<IdxT>&)> metric;
  if (split_criterion == CRITERION::MSE) {
    metric = [](const std::vector<DataT>& y, const std::vector<IdxT>& idx) -> float {
      float y_mean = 0.0f;
      float mse    = 0.0f;
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
    metric = [](const std::vector<DataT>& y, const std::vector<IdxT>& idx) -> float {
      float y_mean = 0.0f;
      float mae    = 0.0f;
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
  float expected_gain = metric(h_labels, {0, 1, 2, 3, 4}) - 2.0f / 5.0f * metric(h_labels, {0, 4}) -
                        3.0f / 5.0f * metric(h_labels, {1, 2, 3});

  EXPECT_FLOAT_EQ(h_splits[0].best_metric_val, expected_gain);

  d_allocator->deallocate(mutex, sizeof(int) * max_batch, 0);
  d_allocator->deallocate(done_count, sizeof(int) * max_batch * n_col_blks, 0);
  d_allocator->deallocate(hist, 2 * nPredCounts * sizeof(DataT), 0);
  d_allocator->deallocate(workload_info, sizeof(WorkloadInfo<IdxT>), 0);
}

INSTANTIATE_TEST_SUITE_P(BatchedLevelAlgoUnitTest,
                         TestMetric,
                         ::testing::Values(CRITERION::MSE),
                         [](const auto& info) {
                           switch (info.param) {
                             case CRITERION::MSE: return "MSE";
                             default: return "";
                           }
                         });
*/
}  // namespace DT
}  // namespace ML
