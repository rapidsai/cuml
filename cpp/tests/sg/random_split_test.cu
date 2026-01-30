/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file random_split_test.cu
 * @brief Unit tests for RandomObjectiveFunction used by Isolation Forest.
 *
 * Tests the RANDOM split criterion which enables random partitioning
 * without impurity optimization for anomaly detection.
 */

#include <cuml/tree/algo_helper.h>

#include <raft/core/handle.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <decisiontree/batched-levelalgo/bins.cuh>
#include <decisiontree/batched-levelalgo/objectives.cuh>
#include <decisiontree/batched-levelalgo/split.cuh>
#include <gtest/gtest.h>

#include <limits>
#include <vector>

namespace ML {
namespace DT {

class RandomSplitTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    handle = std::make_unique<raft::handle_t>();
    stream = handle->get_stream();
  }

  void TearDown() override { handle.reset(); }

  std::unique_ptr<raft::handle_t> handle;
  cudaStream_t stream;
};

/**
 * @brief Test that GainPerSplit returns constant positive gain for valid splits.
 */
TEST_F(RandomSplitTest, GainPerSplitReturnsConstantGain)
{
  using DataT  = float;
  using LabelT = float;
  using IdxT   = int;

  RandomObjectiveFunction<DataT, LabelT, IdxT> objective(1, 1);  // nclasses=1, min_samples_leaf=1

  // Create a mock histogram (not actually used for gain computation in random splits)
  AggregateBin hist[10];
  for (int i = 0; i < 10; i++) {
    hist[i].count     = i + 1;
    hist[i].label_sum = static_cast<double>(i + 1);
  }

  IdxT n_bins  = 10;
  IdxT len     = 55;  // total samples (sum of counts 1..10)
  IdxT nLeft_5 = 15;  // sum of counts 1..5

  // GainPerSplit should return 1.0 for valid splits
  DataT gain = objective.GainPerSplit(hist, 4, n_bins, len, nLeft_5);
  EXPECT_FLOAT_EQ(gain, 1.0f);
}

/**
 * @brief Test that GainPerSplit rejects splits violating min_samples_leaf.
 */
TEST_F(RandomSplitTest, GainPerSplitRejectsInvalidSplits)
{
  using DataT  = float;
  using LabelT = float;
  using IdxT   = int;

  // min_samples_leaf = 10
  RandomObjectiveFunction<DataT, LabelT, IdxT> objective(1, 10);

  AggregateBin hist[10];
  for (int i = 0; i < 10; i++) {
    hist[i].count     = 5;  // 5 samples per bin
    hist[i].label_sum = 5.0;
  }

  IdxT n_bins = 10;
  IdxT len    = 50;   // total samples
  IdxT nLeft  = 5;    // only 5 samples on left - violates min_samples_leaf=10
                      // nRight = 45, which is valid, but nLeft is too small

  DataT gain = objective.GainPerSplit(hist, 0, n_bins, len, nLeft);
  EXPECT_FLOAT_EQ(gain, -std::numeric_limits<DataT>::max());

  // Now test when right side violates
  nLeft = 45;   // valid
  // nRight = 5 - violates min_samples_leaf
  gain = objective.GainPerSplit(hist, 8, n_bins, len, nLeft);
  EXPECT_FLOAT_EQ(gain, -std::numeric_limits<DataT>::max());
}

/**
 * @brief Test that RandomObjectiveFunction can be instantiated with different nclasses.
 * Note: NumClasses() is a device-only function and cannot be tested from host code.
 * The actual NumClasses() always returns 1 for regression-style output.
 */
TEST_F(RandomSplitTest, ObjectiveCanBeInstantiated)
{
  using DataT  = float;
  using LabelT = float;
  using IdxT   = int;

  // Should compile and instantiate without errors
  RandomObjectiveFunction<DataT, LabelT, IdxT> objective1(1, 1);
  RandomObjectiveFunction<DataT, LabelT, IdxT> objective5(5, 1);
  RandomObjectiveFunction<DataT, LabelT, IdxT> objective10(10, 5);

  // Verify we can call host-callable functions
  AggregateBin hist[1];
  hist[0].count     = 10;
  hist[0].label_sum = 10.0;

  // GainPerSplit is HDI (host+device) so it can be called from host
  auto gain = objective1.GainPerSplit(hist, 0, 1, 20, 10);
  EXPECT_FLOAT_EQ(gain, 1.0f);
}

// Note: SetLeafVector and NumClasses are device-only (DI) functions.
// They are tested implicitly through integration tests that run on GPU.

/**
 * @brief Test that RANDOM criterion is properly defined in the enum.
 */
TEST_F(RandomSplitTest, RandomCriterionEnumExists)
{
  // Verify RANDOM is a valid criterion value
  CRITERION criterion = CRITERION::RANDOM;
  EXPECT_NE(criterion, CRITERION::CRITERION_END);
  EXPECT_NE(criterion, CRITERION::GINI);
  EXPECT_NE(criterion, CRITERION::MSE);
}

/**
 * @brief Test double precision version of RandomObjectiveFunction.
 */
TEST_F(RandomSplitTest, DoublePrecisionSupport)
{
  using DataT  = double;
  using LabelT = double;
  using IdxT   = int;

  RandomObjectiveFunction<DataT, LabelT, IdxT> objective(1, 1);

  AggregateBin hist[5];
  for (int i = 0; i < 5; i++) {
    hist[i].count     = 10;
    hist[i].label_sum = 10.0;
  }

  IdxT n_bins = 5;
  IdxT len    = 50;
  IdxT nLeft  = 20;

  DataT gain = objective.GainPerSplit(hist, 1, n_bins, len, nLeft);
  EXPECT_DOUBLE_EQ(gain, 1.0);
}

/**
 * @brief Test edge case: single sample node should not split.
 */
TEST_F(RandomSplitTest, SingleSampleNodeCannotSplit)
{
  using DataT  = float;
  using LabelT = float;
  using IdxT   = int;

  RandomObjectiveFunction<DataT, LabelT, IdxT> objective(1, 1);

  AggregateBin hist[1];
  hist[0].count     = 1;
  hist[0].label_sum = 1.0;

  IdxT n_bins = 1;
  IdxT len    = 1;     // Only 1 sample total
  IdxT nLeft  = 1;     // 1 on left
  // nRight = 0        // 0 on right - invalid

  DataT gain = objective.GainPerSplit(hist, 0, n_bins, len, nLeft);
  // Should reject because nRight < min_samples_leaf
  EXPECT_FLOAT_EQ(gain, -std::numeric_limits<DataT>::max());
}

/**
 * @brief Test that all valid splits have equal gain (no preference).
 */
TEST_F(RandomSplitTest, AllValidSplitsHaveEqualGain)
{
  using DataT  = float;
  using LabelT = float;
  using IdxT   = int;

  RandomObjectiveFunction<DataT, LabelT, IdxT> objective(1, 1);

  AggregateBin hist[10];
  for (int i = 0; i < 10; i++) {
    hist[i].count     = 10;
    hist[i].label_sum = 10.0;
  }

  IdxT n_bins = 10;
  IdxT len    = 100;

  // All splits with valid left/right counts should have same gain
  std::vector<DataT> gains;
  for (IdxT i = 0; i < n_bins - 1; i++) {
    IdxT nLeft = (i + 1) * 10;  // 10, 20, 30, ... 90
    DataT gain = objective.GainPerSplit(hist, i, n_bins, len, nLeft);
    if (gain > 0) { gains.push_back(gain); }
  }

  // All valid gains should be equal (1.0)
  for (const auto& g : gains) {
    EXPECT_FLOAT_EQ(g, 1.0f);
  }
}

}  // namespace DT
}  // namespace ML
