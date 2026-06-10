/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../prims/test_utils.h"
#include "test_opg_utils.h"

#include <cuml/ensemble/randomforest.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/handle.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

namespace ML {
namespace Test {
namespace opg {

enum class PartitionKind { Contiguous, Strided, Imbalanced, EmptyNonRootRanks };

struct RfMgTestParams {
  int n_rows;
  int n_cols;
  int n_trees;
  float max_features;
  int max_depth;
  int max_leaves;
  int max_n_bins;
  int min_samples_leaf;
  int min_samples_split;
  float min_impurity_decrease;
  int n_streams;
  int handle_n_streams;
  CRITERION split_criterion;
  int seed;
  int n_labels;
  bool double_precision;
  PartitionKind partition_kind;
};

template <typename T>
void hash_value(uint64_t& hash, const T& value)
{
  const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
  for (size_t i = 0; i < sizeof(T); ++i) {
    hash ^= bytes[i];
    hash *= 1099511628211ULL;
  }
}

template <typename T, typename L>
uint64_t hash_forest_structure(const RandomForestMetaData<T, L>& forest)
{
  uint64_t hash = 1469598103934665603ULL;
  hash_value(hash, forest.n_features);
  hash_value(hash, forest.rf_params.n_trees);
  hash_value(hash, forest.rf_params.bootstrap);
  hash_value(hash, forest.rf_params.max_samples);
  hash_value(hash, forest.rf_params.seed);
  hash_value(hash, forest.rf_params.tree_params.max_depth);
  hash_value(hash, forest.rf_params.tree_params.max_leaves);
  hash_value(hash, forest.rf_params.tree_params.max_n_bins);
  hash_value(hash, forest.rf_params.tree_params.min_samples_leaf);
  hash_value(hash, forest.rf_params.tree_params.min_samples_split);
  hash_value(hash, forest.rf_params.tree_params.min_impurity_decrease);
  hash_value(hash, forest.rf_params.tree_params.split_criterion);
  for (auto const& tree : forest.trees) {
    hash_value(hash, tree->treeid);
    hash_value(hash, tree->depth_counter);
    hash_value(hash, tree->leaf_counter);
    hash_value(hash, tree->num_outputs);
    hash_value(hash, tree->sparsetree.size());
    for (auto const& node : tree->sparsetree) {
      hash_value(hash, node.ColumnId());
      hash_value(hash, node.QueryValue());
      hash_value(hash, node.BestMetric());
      hash_value(hash, node.LeftChildId());
      hash_value(hash, node.InstanceCount());
      hash_value(hash, node.IsLeaf());
    }
  }
  return hash;
}

template <typename T, typename L>
uint64_t hash_forest_leaf_values(const RandomForestMetaData<T, L>& forest)
{
  uint64_t hash = 1469598103934665603ULL;
  for (auto const& tree : forest.trees) {
    hash_value(hash, tree->vector_leaf.size());
    for (auto const& leaf : tree->vector_leaf) {
      hash_value(hash, leaf);
    }
  }
  return hash;
}

template <typename T>
uint64_t hash_host_vector(std::vector<T> const& values)
{
  uint64_t hash = 1469598103934665603ULL;
  hash_value(hash, values.size());
  for (auto const& value : values) {
    hash_value(hash, value);
  }
  return hash;
}

std::vector<int> local_rows_for_rank(int n_rows, int rank, int size, PartitionKind kind)
{
  std::vector<int> rows;
  if (kind == PartitionKind::Strided) {
    for (int row = rank; row < n_rows; row += size) {
      rows.push_back(row);
    }
    return rows;
  }

  std::vector<int> counts(size, n_rows / size);
  for (int i = 0; i < n_rows % size; ++i) {
    counts[i]++;
  }
  if (kind == PartitionKind::Imbalanced && size > 1) {
    counts.assign(size, 0);
    counts[0]     = std::max(1, (n_rows * 3) / 4);
    int remaining = n_rows - counts[0];
    for (int i = 1; i < size; ++i) {
      counts[i] = remaining / (size - 1);
    }
    for (int i = 1; i <= remaining % (size - 1); ++i) {
      counts[i]++;
    }
  } else if (kind == PartitionKind::EmptyNonRootRanks && size > 1) {
    counts.assign(size, 0);
    counts[0] = n_rows;
  }

  int begin = std::accumulate(counts.begin(), counts.begin() + rank, 0);
  rows.resize(counts[rank]);
  std::iota(rows.begin(), rows.end(), begin);
  rows.erase(std::remove_if(rows.begin(), rows.end(), [=](int row) { return row >= n_rows; }),
             rows.end());
  return rows;
}

template <typename DataT, typename LabelT>
void make_local_dataset(RfMgTestParams const& params,
                        std::vector<int> const& rows,
                        std::vector<DataT>& X,
                        std::vector<LabelT>& y)
{
  X.resize(rows.size() * params.n_cols);
  y.resize(rows.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    int global_row = rows[i];
    DataT signal   = static_cast<DataT>((global_row % 97) - 48);
    for (int col = 0; col < params.n_cols; ++col) {
      DataT feature = signal * static_cast<DataT>(col + 1);
      feature += static_cast<DataT>(((global_row + 13 * col + params.seed) % 11) - 5) /
                 static_cast<DataT>(10);
      X[i * params.n_cols + col] = feature;
    }
    if constexpr (std::is_integral_v<LabelT>) {
      y[i] = (signal >= DataT(0)) ? 1 : 0;
      if (params.n_labels > 2 && global_row % 17 == 0) { y[i] = 2; }
    } else {
      y[i] = signal * DataT(0.5) + static_cast<DataT>((global_row % 7) - 3);
    }
  }
}

template <typename DataT>
void make_prediction_dataset(RfMgTestParams const& params, std::vector<DataT>& X)
{
  X.resize(params.n_rows * params.n_cols);
  for (int row = 0; row < params.n_rows; ++row) {
    for (int col = 0; col < params.n_cols; ++col) {
      X[row * params.n_cols + col] =
        static_cast<DataT>(((row * 5 + col * 11 + params.seed) % 101) - 50);
    }
  }
}

template <typename T, typename L>
void expect_global_tree_counts(RandomForestMetaData<T, L> const& forest, int n_rows)
{
  for (auto const& tree : forest.trees) {
    ASSERT_FALSE(tree->sparsetree.empty());
    EXPECT_EQ(tree->sparsetree.front().InstanceCount(), n_rows);
    for (auto const& node : tree->sparsetree) {
      if (!node.IsLeaf()) {
        auto left_count  = tree->sparsetree[node.LeftChildId()].InstanceCount();
        auto right_count = tree->sparsetree[node.RightChildId()].InstanceCount();
        EXPECT_EQ(left_count + right_count, node.InstanceCount());
      }
    }
  }
}

template <typename T, typename L>
void expect_tree_limits(RandomForestMetaData<T, L> const& forest, RfMgTestParams const& params)
{
  for (auto const& tree : forest.trees) {
    EXPECT_LE(tree->depth_counter, params.max_depth);
    if (params.max_leaves > 0) { EXPECT_LE(tree->leaf_counter, params.max_leaves); }
    for (auto const& node : tree->sparsetree) {
      if (!node.IsLeaf()) { EXPECT_GT(node.BestMetric(), params.min_impurity_decrease); }
    }
  }
}

void initialize_mpi_once()
{
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) { MPI_Init(nullptr, nullptr); }
}

template <typename DataT, typename LabelT>
class RfMgPropertyTestImpl {
 public:
  explicit RfMgPropertyTestImpl(RfMgTestParams const& params) : params(params)
  {
    initialize_mpi_once();
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_gpus = 0;
    RAFT_CUDA_TRY(cudaGetDeviceCount(&n_gpus));
    if (n_gpus < size) {
      ADD_FAILURE() << "Number of GPUs is smaller than MPI ranks: ngpus=" << n_gpus
                    << ", nranks=" << size;
      return;
    }
    RAFT_CUDA_TRY(cudaSetDevice(rank));

    auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(params.handle_n_streams);
    raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
    raft::comms::initialize_mpi_comms(&handle, MPI_COMM_WORLD);

    auto local_rows = local_rows_for_rank(params.n_rows, rank, size, params.partition_kind);
    std::vector<DataT> h_X;
    std::vector<LabelT> h_y;
    make_local_dataset<DataT, LabelT>(params, local_rows, h_X, h_y);

    rmm::device_uvector<DataT> X(h_X.size(), handle.get_stream());
    rmm::device_uvector<LabelT> y(h_y.size(), handle.get_stream());
    raft::update_device(X.data(), h_X.data(), h_X.size(), handle.get_stream());
    raft::update_device(y.data(), h_y.data(), h_y.size(), handle.get_stream());

    auto rf_params = set_rf_params(params.max_depth,
                                   params.max_leaves,
                                   params.max_features,
                                   params.max_n_bins,
                                   params.min_samples_leaf,
                                   params.min_samples_split,
                                   params.min_impurity_decrease,
                                   false,
                                   params.n_trees,
                                   1.0f,
                                   params.seed,
                                   params.split_criterion,
                                   params.n_streams,
                                   128);

    RandomForestMetaData<DataT, LabelT> forest;
    if constexpr (std::is_integral_v<LabelT>) {
      fit(handle,
          &forest,
          X.data(),
          static_cast<int>(local_rows.size()),
          params.n_cols,
          y.data(),
          params.n_labels,
          rf_params);
    } else {
      fit(handle,
          &forest,
          X.data(),
          static_cast<int>(local_rows.size()),
          params.n_cols,
          y.data(),
          rf_params);
    }

    expect_global_tree_counts(forest, params.n_rows);
    expect_tree_limits(forest, params);
    expect_identical_across_ranks(handle, hash_forest_structure(forest), "tree structure");
    expect_identical_across_ranks(handle, hash_forest_leaf_values(forest), "leaf values");
    expect_identical_predictions_across_ranks(handle, &forest);
  }

 private:
  void expect_identical_across_ranks(raft::handle_t const& handle,
                                     uint64_t local_hash,
                                     char const* label)
  {
    auto const& comm = handle.get_comms();
    std::vector<uint64_t> hashes(comm.get_size());
    MPI_Allgather(&local_hash, 1, MPI_UINT64_T, hashes.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD);
    for (auto hash : hashes) {
      EXPECT_EQ(hash, hashes.front()) << "Mismatched distributed RF " << label;
    }
  }

  void expect_identical_predictions_across_ranks(raft::handle_t const& handle,
                                                 RandomForestMetaData<DataT, LabelT>* forest)
  {
    std::vector<DataT> h_X;
    make_prediction_dataset(params, h_X);
    rmm::device_uvector<DataT> X(h_X.size(), handle.get_stream());
    rmm::device_uvector<DataT> X_transpose(h_X.size(), handle.get_stream());
    rmm::device_uvector<LabelT> predictions(params.n_rows, handle.get_stream());
    raft::update_device(X.data(), h_X.data(), h_X.size(), handle.get_stream());
    raft::linalg::transpose(
      handle, X.data(), X_transpose.data(), params.n_rows, params.n_cols, handle.get_stream());
    predict(handle, forest, X_transpose.data(), params.n_rows, params.n_cols, predictions.data());
    std::vector<LabelT> h_predictions(params.n_rows);
    raft::update_host(
      h_predictions.data(), predictions.data(), h_predictions.size(), handle.get_stream());
    handle.sync_stream();
    expect_identical_across_ranks(handle, hash_host_vector(h_predictions), "predictions");
  }

  RfMgTestParams params;
};

class RfMgPropertyTest : public ::testing::TestWithParam<RfMgTestParams> {
 public:
  void SetUp() override
  {
    auto params        = GetParam();
    bool is_regression = params.split_criterion != GINI && params.split_criterion != ENTROPY;
    if (params.double_precision) {
      if (is_regression) {
        RfMgPropertyTestImpl<double, double> test(params);
      } else {
        RfMgPropertyTestImpl<double, int> test(params);
      }
    } else {
      if (is_regression) {
        RfMgPropertyTestImpl<float, float> test(params);
      } else {
        RfMgPropertyTestImpl<float, int> test(params);
      }
    }
  }
};

TEST_P(RfMgPropertyTest, DistributedProperties) {}

std::vector<RfMgTestParams> inputs = {
  {128, 4, 1, 1.0f, 3, -1, 16, 1, 2, 0.0f, 1, 1, GINI, 7, 2, false, PartitionKind::Contiguous},
  {128, 4, 3, 0.5f, 4, 16, 32, 1, 2, 0.0f, 4, 4, ENTROPY, 11, 2, false, PartitionKind::Strided},
  {192, 6, 1, 1.0f, 5, -1, 32, 2, 4, 0.0f, 1, 1, MSE, 13, 2, false, PartitionKind::Imbalanced},
  {96, 3, 2, 1.0f, 4, 8, 8, 1, 2, 0.0f, 1, 1, GINI, 17, 2, true, PartitionKind::Imbalanced},
  {144, 5, 2, 0.8f, 4, -1, 16, 1, 2, 0.0f, 1, 1, GINI, 31, 3, false, PartitionKind::Strided},
  {160, 5, 1, 0.8f, 4, -1, 16, 1, 2, 0.0f, 1, 1, MSE, 19, 2, true, PartitionKind::Contiguous},
  {64,
   4,
   2,
   1.0f,
   4,
   -1,
   16,
   1,
   2,
   0.0f,
   3,
   3,
   GINI,
   23,
   2,
   false,
   PartitionKind::EmptyNonRootRanks},
  {80,
   5,
   2,
   0.8f,
   4,
   -1,
   16,
   1,
   2,
   0.0f,
   3,
   3,
   MSE,
   29,
   2,
   false,
   PartitionKind::EmptyNonRootRanks}};

INSTANTIATE_TEST_CASE_P(RfTests, RfMgPropertyTest, ::testing::ValuesIn(inputs));

}  // namespace opg
}  // namespace Test
}  // namespace ML

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MLCommon::Test::opg::MPIEnvironment());

  return RUN_ALL_TESTS();
}
