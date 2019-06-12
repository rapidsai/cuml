/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "score/scores.h"
#include <gtest/gtest.h>
#include "random/rng.h"
#include "test_utils.h"
#include <iostream>

namespace MLCommon {
namespace Score {

class ScoreTest : public ::testing::Test {
protected:
  void SetUp() override {}

  void TearDown() override {}

};

typedef ScoreTest ScoreTestHighScore;
TEST(ScoreTestHighScore, Result) {

    float y[5] = {0.1, 0.2, 0.3, 0.4, 0.5};
    float y_hat[5] = {0.12, 0.22, 0.32, 0.42, 0.52};

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    float *d_y;
    MLCommon::allocate(d_y, 5);

    float *d_y_hat;
    MLCommon::allocate(d_y_hat, 5);

    MLCommon::updateDevice(d_y_hat, y_hat, 5, stream);
    MLCommon::updateDevice(d_y, y, 5, stream);

    float result = MLCommon::Score::r2_score(d_y, d_y_hat, 5, stream);
    ASSERT_TRUE(result == 0.98f);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

typedef ScoreTest ScoreTestLowScore;
TEST(ScoreTestLowScore, Result) {

    float y[5] = {0.1, 0.2, 0.3, 0.4, 0.5};
    float y_hat[5] = {0.012, 0.022, 0.032, 0.042, 0.052};

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    float *d_y;
    MLCommon::allocate(d_y, 5);

    float *d_y_hat;
    MLCommon::allocate(d_y_hat, 5);

    MLCommon::updateDevice(d_y_hat, y_hat, 5, stream);
    MLCommon::updateDevice(d_y, y, 5, stream);

    float result = MLCommon::Score::r2_score(d_y, d_y_hat, 5, stream);

    std::cout << "Result: " << result - -3.4012f << std::endl;
    ASSERT_TRUE(result - -3.4012f < 0.00001);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

struct AccuracyInputs {
	/**
	 * Number of predictions.
	 */
	int n;
	/**
         * Number of predictions w/ different values than their corresponding element in reference predictions.
	 * Valid range [0, n]. changed_n in [0, n] will yield accuracy of (n - changed_n) / n.
	 */
	int changed_n;
	/**
	 * Seed for randomly generated predictions.
	 */
	unsigned long long int seed;
};

std::ostream &operator<<(::std::ostream &os, const AccuracyInputs &acc_inputs) {
  os << "AccuracyInputs are {" << acc_inputs.n << ", " << acc_inputs.changed_n << ", " << acc_inputs.seed << "}" << std::endl;
  return os;
}

template<typename T>
__global__ void change_vals(T * predictions, T * ref_predictions, const int changed_n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < changed_n) {
    predictions[tid] = ref_predictions[tid] + 1; // change first changed_n predictions
  }
}

template<typename T>
class AccuracyTest : public ::testing::TestWithParam<AccuracyInputs> {
protected:

  void SetUp() override {
        params = ::testing::TestWithParam<AccuracyInputs>::GetParam();
    ASSERT((params.changed_n <= params.n) && (params.changed_n >= 0), "Invalid params.");

        Random::Rng r(params.seed);
        CUDA_CHECK(cudaStreamCreate(&stream));
        std::shared_ptr<deviceAllocator> d_allocator(new defaultDeviceAllocator);

    allocate(predictions, params.n);
    allocate(ref_predictions, params.n);
    r.normal(ref_predictions, params.n, (T) 0.0, (T) 1.0, stream);
    copyAsync(predictions, ref_predictions, params.n, stream);
          CUDA_CHECK(cudaStreamSynchronize(stream));

    //Modify params.changed_n unique predictions to a different value. New value is irrelevant.
    if (params.changed_n > 0) {
       int threads = 64;
      int blocks = ceildiv(params.changed_n, threads);
      //@todo Could also generate params.changed_n unique random positions in [0, n) range, instead of changing the first ones.
        change_vals<T><<<blocks, threads, 0, stream>>>(predictions, ref_predictions, params.changed_n);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    computed_accuracy = MLCommon::Score::accuracy_score<T>(predictions, ref_predictions, params.n,  d_allocator, stream);
    ref_accuracy = (params.n - params.changed_n) * 1.0f/ params.n;
    //std::cout << "computed_accuracy is " << computed_accuracy << " ref_accuracy is " << ref_accuracy << std::endl;
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(predictions));
    CUDA_CHECK(cudaFree(ref_predictions));
    CUDA_CHECK(cudaStreamDestroy(stream));
    computed_accuracy = -1.0f;
    ref_accuracy = -1.0f;
  }

  AccuracyInputs params;
  T * predictions, * ref_predictions;
  float computed_accuracy, ref_accuracy;
  cudaStream_t stream;
};

const std::vector<AccuracyInputs> inputs = {
  {1, 1, 1234ULL}, // single element, wrong prediction
  {1, 0, 1234ULL}, // single element, perfect prediction
  {2, 1, 1234ULL}, // multiple elements, 0.5 accuracy
  {1000, 0, 1234ULL}, // multiple elements, perfect predictions
  {1000, 1000, 1234ULL}, // multiple elements, no correct predictions
  {1000, 80, 1234ULL}, // multiple elements, prediction mix
  {1000, 45, 1234ULL} // multiple elements, prediction mix
};


typedef AccuracyTest<float> AccuracyTestF;
TEST_P(AccuracyTestF, Result) {
  ASSERT_TRUE(computed_accuracy == ref_accuracy);
}

typedef AccuracyTest<double> AccuracyTestD;
TEST_P(AccuracyTestD, Result) {
  ASSERT_TRUE(computed_accuracy == ref_accuracy);
}

INSTANTIATE_TEST_CASE_P(AccuracyTests, AccuracyTestF, ::testing::ValuesIn(inputs));

INSTANTIATE_TEST_CASE_P(AccuracyTests, AccuracyTestD, ::testing::ValuesIn(inputs));

} // end namespace Score
} // end namespace MLCommon

