/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include "test_utils.h"
#include <gtest/gtest.h>
#include <iostream>
#include <metrics/scores.cuh>
#include <raft/cudart_utils.h>
#include <raft/interruptible.hpp>
#include <raft/random/rng.hpp>
#include <rmm/device_uvector.hpp>
#include <vector>

namespace MLCommon {
namespace Score {

class ScoreTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

typedef ScoreTest ScoreTestHighScore;
TEST(ScoreTestHighScore, Result)
{
  float y[5]     = {0.1, 0.2, 0.3, 0.4, 0.5};
  float y_hat[5] = {0.12, 0.22, 0.32, 0.42, 0.52};

  cudaStream_t stream = 0;
  RAFT_CUDA_TRY(cudaStreamCreate(&stream));

  rmm::device_uvector<float> d_y(5, stream);
  rmm::device_uvector<float> d_y_hat(5, stream);

  raft::update_device(d_y_hat.data(), y_hat, 5, stream);
  raft::update_device(d_y.data(), y, 5, stream);

  auto result = MLCommon::Score::r2_score(d_y.data(), d_y_hat.data(), 5, stream);
  ASSERT_TRUE(result == 0.98f);
  RAFT_CUDA_TRY(cudaStreamDestroy(stream));
}

typedef ScoreTest ScoreTestLowScore;
TEST(ScoreTestLowScore, Result)
{
  float y[5]     = {0.1, 0.2, 0.3, 0.4, 0.5};
  float y_hat[5] = {0.012, 0.022, 0.032, 0.042, 0.052};

  cudaStream_t stream = 0;
  RAFT_CUDA_TRY(cudaStreamCreate(&stream));

  rmm::device_uvector<float> d_y(5, stream);
  rmm::device_uvector<float> d_y_hat(5, stream);

  raft::update_device(d_y_hat.data(), y_hat, 5, stream);
  raft::update_device(d_y.data(), y, 5, stream);

  auto result = MLCommon::Score::r2_score(d_y.data(), d_y_hat.data(), 5, stream);

  std::cout << "Result: " << result - -3.4012f << std::endl;
  ASSERT_TRUE(result - -3.4012f < 0.00001);
  RAFT_CUDA_TRY(cudaStreamDestroy(stream));
}

// Tests for accuracy_score

struct AccuracyInputs {
  /**
   * Number of predictions.
   */
  int n;
  /**
   * Number of predictions w/ different values than their corresponding element in reference
   * predictions. Valid range [0, n]. changed_n in [0, n] will yield accuracy of (n - changed_n) /
   * n.
   */
  int changed_n;
  /**
   * Seed for randomly generated predictions.
   */
  unsigned long long int seed;
};

std::ostream& operator<<(::std::ostream& os, const AccuracyInputs& acc_inputs)
{
  os << "AccuracyInputs are {" << acc_inputs.n << ", " << acc_inputs.changed_n << ", "
     << acc_inputs.seed << "}" << std::endl;
  return os;
}

template <typename T>
__global__ void change_vals(T* predictions, T* ref_predictions, const int changed_n)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < changed_n) {
    predictions[tid] = ref_predictions[tid] + 1;  // change first changed_n predictions
  }
}

template <typename T>
class AccuracyTest : public ::testing::TestWithParam<AccuracyInputs> {
 protected:
  void SetUp() override
  {
    params = ::testing::TestWithParam<AccuracyInputs>::GetParam();
    ASSERT((params.changed_n <= params.n) && (params.changed_n >= 0), "Invalid params.");

    raft::random::Rng r(params.seed);
    RAFT_CUDA_TRY(cudaStreamCreate(&stream));

    rmm::device_uvector<T> predictions(params.n, stream);
    rmm::device_uvector<T> ref_predictions(params.n, stream);
    r.normal(ref_predictions.data(), params.n, (T)0.0, (T)1.0, stream);
    raft::copy_async(predictions.data(), ref_predictions.data(), params.n, stream);
    raft::interruptible::synchronize(stream);

    // Modify params.changed_n unique predictions to a different value. New value is irrelevant.
    if (params.changed_n > 0) {
      int threads = 64;
      int blocks  = raft::ceildiv(params.changed_n, threads);
      //@todo Could also generate params.changed_n unique random positions in [0, n) range, instead
      // of changing the first ones.
      change_vals<T><<<blocks, threads, 0, stream>>>(
        predictions.data(), ref_predictions.data(), params.changed_n);
      RAFT_CUDA_TRY(cudaGetLastError());
      raft::interruptible::synchronize(stream);
    }

    computed_accuracy = MLCommon::Score::accuracy_score<T>(
      predictions.data(), ref_predictions.data(), params.n, stream);
    ref_accuracy = (params.n - params.changed_n) * 1.0f / params.n;
    // std::cout << "computed_accuracy is " << computed_accuracy << " ref_accuracy is " <<
    // ref_accuracy << std::endl;
  }

  void TearDown() override
  {
    RAFT_CUDA_TRY(cudaStreamDestroy(stream));
    computed_accuracy = -1.0f;
    ref_accuracy      = -1.0f;
  }

  AccuracyInputs params;
  float computed_accuracy, ref_accuracy;
  cudaStream_t stream = 0;
};

const std::vector<AccuracyInputs> inputs = {
  {1, 1, 1234ULL},        // single element, wrong prediction
  {1, 0, 1234ULL},        // single element, perfect prediction
  {2, 1, 1234ULL},        // multiple elements, 0.5 accuracy
  {1000, 0, 1234ULL},     // multiple elements, perfect predictions
  {1000, 1000, 1234ULL},  // multiple elements, no correct predictions
  {1000, 80, 1234ULL},    // multiple elements, prediction mix
  {1000, 45, 1234ULL}     // multiple elements, prediction mix
};

typedef AccuracyTest<float> AccuracyTestF;
TEST_P(AccuracyTestF, Result) { ASSERT_TRUE(computed_accuracy == ref_accuracy); }

typedef AccuracyTest<double> AccuracyTestD;
TEST_P(AccuracyTestD, Result) { ASSERT_TRUE(computed_accuracy == ref_accuracy); }

INSTANTIATE_TEST_CASE_P(AccuracyTests, AccuracyTestF, ::testing::ValuesIn(inputs));
INSTANTIATE_TEST_CASE_P(AccuracyTests, AccuracyTestD, ::testing::ValuesIn(inputs));

// Tests for regression_metrics

template <typename T>
struct RegressionInputs {
  T tolerance;
  int n;                 // number of predictions
  bool hardcoded_preds;  // (hardcoded_preds) ? use predictions, ref_predictions : use randomly
                         // generated arrays.
  std::vector<T> predictions;
  std::vector<T> ref_predictions;
  T predictions_range[2];      // predictions in predictions_range if not hardcoded_preds
  T ref_predictions_range[2];  // predictions in ref_predictions_range if not hardcoded_preds
  unsigned long long int seed;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const RegressionInputs<T>& reg_inputs)
{
  os << "RegressionInputs are {" << reg_inputs.tolerance << ", " << reg_inputs.n << ", "
     << reg_inputs.hardcoded_preds << ", ";
  if (reg_inputs.hardcoded_preds) {
    os << "{";
    for (int i = 0; i < reg_inputs.n; i++)
      os << reg_inputs.predictions[i] << ", ";
    os << "}, {";
    for (int i = 0; i < reg_inputs.n; i++)
      os << reg_inputs.ref_predictions[i] << ", ";
    os << "}";
    os << "{" << reg_inputs.predictions_range[0] << ", " << reg_inputs.predictions_range[1]
       << "}, ";
    os << "{" << reg_inputs.ref_predictions_range[0] << ", " << reg_inputs.ref_predictions_range[1]
       << "}";
  } else {
    os << "{}, {}, {}, {}";
  }
  os << ", " << reg_inputs.seed;
  return os;
}

template <typename T>
void host_regression_computations(std::vector<T>& predictions,
                                  std::vector<T>& ref_predictions,
                                  const int n,
                                  std::vector<double>& regression_metrics)
{
  double abs_difference_sum = 0;
  double mse_sum            = 0;
  std::vector<double> abs_diffs(n);

  for (int i = 0; i < n; i++) {
    double abs_diff = raft::abs(predictions[i] - ref_predictions[i]);
    abs_difference_sum += abs_diff;
    mse_sum += pow(predictions[i] - ref_predictions[i], 2);
    abs_diffs[i] = abs_diff;
  }

  regression_metrics[0] = abs_difference_sum / n;
  regression_metrics[1] = mse_sum / n;

  std::sort(abs_diffs.begin(), abs_diffs.end());
  int middle = n / 2;
  if (n % 2 == 1) {
    regression_metrics[2] = abs_diffs[middle];
  } else {
    regression_metrics[2] = (abs_diffs[middle] + abs_diffs[middle - 1]) / 2;
  }
}

template <typename T>
class RegressionMetricsTest : public ::testing::TestWithParam<RegressionInputs<T>> {
 protected:
  void SetUp() override
  {
    params = ::testing::TestWithParam<RegressionInputs<T>>::GetParam();
    computed_regression_metrics.assign(3, -1.0);
    ref_regression_metrics.assign(3, -1.0);

    RAFT_CUDA_TRY(cudaStreamCreate(&stream));

    rmm::device_uvector<T> d_predictions(params.n, stream);
    rmm::device_uvector<T> d_ref_predictions(params.n, stream);

    if (params.hardcoded_preds) {
      raft::update_device(d_predictions.data(), params.predictions.data(), params.n, stream);
      raft::update_device(
        d_ref_predictions.data(), params.ref_predictions.data(), params.n, stream);
    } else {
      params.predictions.resize(params.n);
      params.ref_predictions.resize(params.n);
      raft::random::Rng r(params.seed);
      // randomly generate arrays
      r.uniform(d_predictions.data(),
                params.n,
                params.predictions_range[0],
                params.predictions_range[1],
                stream);
      r.uniform(d_ref_predictions.data(),
                params.n,
                params.ref_predictions_range[0],
                params.ref_predictions_range[1],
                stream);
      // copy to host to compute reference regression metrics
      raft::update_host(params.predictions.data(), d_predictions.data(), params.n, stream);
      raft::update_host(params.ref_predictions.data(), d_ref_predictions.data(), params.n, stream);
      raft::interruptible::synchronize(stream);
    }

    MLCommon::Score::regression_metrics(d_predictions.data(),
                                        d_ref_predictions.data(),
                                        params.n,
                                        stream,
                                        computed_regression_metrics[0],
                                        computed_regression_metrics[1],
                                        computed_regression_metrics[2]);

    host_regression_computations(
      params.predictions, params.ref_predictions, params.n, ref_regression_metrics);
    raft::interruptible::synchronize(stream);
  }

  void TearDown() override { RAFT_CUDA_TRY(cudaStreamDestroy(stream)); }

  RegressionInputs<T> params;
  std::vector<double> computed_regression_metrics;
  std::vector<double> ref_regression_metrics;
  cudaStream_t stream = 0;
};

const std::vector<RegressionInputs<float>> regression_inputs_float = {
  {0.00001f, 1, true, {10.2f}, {20.2f}, {}, {}, 1234ULL},                // single element
  {0.00001f, 2, true, {10.2f, 40.2f}, {20.2f, 80.2f}, {}, {}, 1234ULL},  // two elements, mean same
                                                                         // as median
  // next three inputs should result in identical regression metrics values
  {0.00001f,
   6,
   true,
   {10.5f, 20.5f, 30.5f, 40.5f, 50.5f, 60.5f},
   {20.5f, 40.5f, 55.5f, 80.5f, 100.5f, 120.5f},
   {},
   {},
   1234ULL},  // diffs all negative, reverse sorted
  {0.00001f,
   6,
   true,
   {20.5f, 40.5f, 55.5f, 80.5f, 100.5f, 120.5f},
   {10.5f, 20.5f, 30.5f, 40.5f, 50.5f, 60.5f},
   {},
   {},
   1234ULL},  // diffs all positive, already sorted
  {0.00001f,
   6,
   true,
   {40.5f, 55.5f, 20.5f, 120.5f, 100.5f, 80.5f},
   {20.5f, 30.5f, 10.5f, 60.5f, 50.5f, 40.5f},
   {},
   {},
   1234ULL},  // mix
  {0.00001f,
   6,
   true,
   {10.5f, 20.5f, 30.5f, 40.5f, 50.5f, 60.5f},
   {10.5f, 20.5f, 30.5f, 40.5f, 50.5f, 60.5f},
   {},
   {},
   1234ULL},  // identical predictions (0 error)
  {0.00001f,
   6,
   true,
   {10.5f, 20.5f, 30.5f, 40.5f, 50.5f, 60.5f},
   {20.5f, 30.5f, 40.5f, 50.5f, 60.5f, 70.5f},
   {},
   {},
   1234ULL},  // predictions[i] - ref_predictions[i] const for each i
  {0.00001f,
   2048,
   false,
   {},
   {},
   {-2048.0f, 2048.0f},
   {-2048.0f, 2048.0f},
   1234ULL},  // random mix, even number of elements
  {0.00001f,
   2049,
   false,
   {},
   {},
   {-2048.0f, 2048.0f},
   {-2048.0f, 2048.0f},
   1234ULL},  // random mix, odd number of elements
  {0.00001f,
   1024,
   false,
   {},
   {},
   {0.0f, 2048.0f},
   {8192.0f, 16384.0f},
   1234ULL},  // random mix, diffs are all negative
  {0.00001f,
   1024,
   false,
   {},
   {},
   {8192.0f, 16384.0f},
   {0.0f, 2048.0f},
   1234ULL}  // random mix, diffs are all positive
};

const std::vector<RegressionInputs<double>> regression_inputs_double = {
  {0.0000001, 1, true, {10.2}, {20.2}, {}, {}, 1234ULL},              // single element
  {0.0000001, 2, true, {10.2, 40.2}, {20.2, 80.2}, {}, {}, 1234ULL},  // two elements
  {0.0000001,
   6,
   true,
   {10.5, 20.5, 30.5, 40.5, 50.5, 60.5},
   {20.5, 40.5, 55.5, 80.5, 100.5, 120.5},
   {},
   {},
   1234ULL},  // diffs all negative, reverse sorted
  {0.0000001,
   6,
   true,
   {20.5, 40.5, 55.5, 80.5, 100.5, 120.5},
   {10.5, 20.5, 30.5, 40.5, 50.5, 60.5},
   {},
   {},
   1234ULL},  // diffs all positive, already sorted
  {0.0000001,
   6,
   true,
   {40.5, 55.5, 20.5, 120.5, 100.5, 80.5},
   {20.5, 30.5, 10.5, 60.5, 50.5, 40.5},
   {},
   {},
   1234ULL},  // mix
  {0.0000001,
   6,
   true,
   {10.5, 20.5, 30.5, 40.5, 50.5, 60.5},
   {10.5, 20.5, 30.5, 40.5, 50.5, 60.5},
   {},
   {},
   1234ULL},  // identical predictions (0 error)
  {0.0000001,
   6,
   true,
   {10.5, 20.5, 30.5, 40.5, 50.5, 60.5},
   {20.5, 30.5, 40.5, 50.5, 60.5, 70.5},
   {},
   {},
   1234ULL},  // predictions[i] - ref_predictions[i] const for each i
  {0.0000001,
   2048,
   false,
   {},
   {},
   {-2048.0, 2048.0},
   {-2048.0, 2048.0},
   1234ULL},  // random mix, even number of elements
  {0.0000001,
   2049,
   false,
   {},
   {},
   {-2048.0, 2048.0},
   {-2048.0, 2048.0},
   1234ULL},  // random mix, odd number of elements
  {0.0000001, 1024, false, {}, {}, {0, 2048}, {8192.0, 16384.0}, 1234ULL},  // random mix, diffs are
                                                                            // all negative
  {0.0000001, 1024, false, {}, {}, {8192.0, 16384.0}, {0.0, 2048}, 1234ULL}  // random mix, diffs
                                                                             // are all positive
};

typedef RegressionMetricsTest<float> RegressionMetricsTestF;
TEST_P(RegressionMetricsTestF, Result)
{
  for (int i = 0; i < 3; i++) {
    ASSERT_TRUE(match(computed_regression_metrics[i],
                      ref_regression_metrics[i],
                      raft::CompareApprox<float>(params.tolerance)));
  }
}

typedef RegressionMetricsTest<double> RegressionMetricsTestD;
TEST_P(RegressionMetricsTestD, Result)
{
  for (int i = 0; i < 3; i++) {
    ASSERT_TRUE(match(computed_regression_metrics[i],
                      ref_regression_metrics[i],
                      raft::CompareApprox<double>(params.tolerance)));
  }
}

INSTANTIATE_TEST_CASE_P(RegressionMetricsTests,
                        RegressionMetricsTestF,
                        ::testing::ValuesIn(regression_inputs_float));
INSTANTIATE_TEST_CASE_P(RegressionMetricsTests,
                        RegressionMetricsTestD,
                        ::testing::ValuesIn(regression_inputs_double));

}  // end namespace Score
}  // end namespace MLCommon
