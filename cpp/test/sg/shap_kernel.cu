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

#include <gtest/gtest.h>
#include <cuml/explainer/kernel_shap.hpp>
#include <raft/mr/device/allocator.hpp>

#include <raft/cudart_utils.h>
#include <raft/cuda_utils.cuh>
#include <raft/handle.hpp>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <test_utils.h>

namespace MLCommon {
class raft::mr::device::allocator;
}

namespace ML {

namespace Explainer {

struct MakeKSHAPDatasetInputs {
  int nrows_exact;
  int nrows_sampled;
  int ncols;
  int nrows_background;
  int max_samples;
  uint64_t seed;
};

template <typename T>
class MakeKSHAPDatasetTest : public ::testing::TestWithParam<MakeKSHAPDatasetInputs> {
 protected:
  void SetUp() override
  {
    int i, j;

    params  = ::testing::TestWithParam<MakeKSHAPDatasetInputs>::GetParam();
    nrows_X = params.nrows_exact + params.nrows_sampled;

    raft::allocate(background, params.nrows_background * params.ncols);
    raft::allocate(observation, params.ncols);
    raft::allocate(nsamples, params.nrows_sampled / 2);

    raft::allocate(X, nrows_X * params.ncols);
    raft::allocate(dataset, nrows_X * params.nrows_background * params.ncols);

    thrust::device_ptr<T> b_ptr   = thrust::device_pointer_cast(background);
    thrust::device_ptr<T> o_ptr   = thrust::device_pointer_cast(observation);
    thrust::device_ptr<int> n_ptr = thrust::device_pointer_cast(nsamples);

    thrust::device_ptr<float> X_ptr = thrust::device_pointer_cast(X);
    thrust::device_ptr<T> d_ptr     = thrust::device_pointer_cast(dataset);

    // Initialize arrays:

    // Aassign a sentinel value to the observation to check easily later
    T sent_value = nrows_X * params.nrows_background * params.ncols * 100;
    for (i = 0; i < params.ncols; i++) {
      o_ptr[i] = sent_value;
    }

    // Initialize background array with different odd value per row, makes
    // it easier to debug if something goes wrong.
    for (i = 0; i < params.nrows_background; i++) {
      for (j = 0; j < params.ncols; j++) {
        b_ptr[i * params.ncols + j] = (i * 2) + 1;
      }
    }

    // Initialize the exact part of X. We create 2 `1` values per row for the test
    thrust::fill(thrust::device, X_ptr, &X_ptr[nrows_X * params.ncols - 1], 0);
    for (i = 0; i < params.nrows_exact; i++) {
      for (j = i; j < i + 2; j++) {
        X_ptr[i * params.ncols + j] = (float)1.0;
      }
    }

    // Initialize the number of samples per row, we initialize each even row to
    // max samples and each odd row to max_samples - 1
    for (i = 0; i < params.nrows_sampled / 2; i++) {
      n_ptr[i] = params.max_samples - i % 2;
    }

    kernel_dataset(handle,
                   X,
                   nrows_X,
                   params.ncols,
                   background,
                   params.nrows_background,
                   dataset,
                   observation,
                   nsamples,
                   params.nrows_sampled,
                   params.max_samples,
                   params.seed);

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    int counter;

    // Check the generated part of X by sampling. The first nrows_exact
    // correspond to the exact part generated before, so we just test after that.
    test_sampled_X = true;
    j              = 0;
    for (i = params.nrows_exact * params.ncols; i < nrows_X * params.ncols / 2;
         i += 2 * params.ncols) {
      // check that number of samples is the number indicated by nsamples.
      counter        = thrust::count(&X_ptr[i], &X_ptr[i + params.ncols], 1);
      test_sampled_X = (test_sampled_X && (counter == n_ptr[j]));

      // check that number of samples of the next line is the compliment,
      // i.e. ncols - nsamples[j]
      counter        = thrust::count(&X_ptr[i + params.ncols], &X_ptr[i + 2 * params.ncols], 1);
      test_sampled_X = (test_sampled_X && (counter == (params.ncols - n_ptr[j])));

      j++;
    }

    // Check for the exact part of the generated dataset.
    test_scatter_exact = true;
    for (i = 0; i < params.nrows_exact; i++) {
      for (j = i * params.nrows_background * params.ncols;
           j < (i + 1) * params.nrows_background * params.ncols;
           j += params.ncols) {
        counter = thrust::count(&d_ptr[j], &d_ptr[j + params.ncols], sent_value);

        // Check that indeed we have two observation entries ber row
        test_scatter_exact = test_scatter_exact && (counter == 2);
        if (not test_scatter_exact) {
          std::cout << "test_scatter_exact counter failed with: " << counter
                    << ", expected value was 2." << std::endl;
          break;
        }
      }
      if (not test_scatter_exact) { break; }
    }

    // Check for the sampled part of the generated dataset
    test_scatter_sampled = true;

    // compliment_ctr is a helper counter to help check nrows_dataset per entry in
    // nsamples without complicating indexing since sampled part starts at nrows_sampled
    int compliment_ctr = 0;
    for (i = params.nrows_exact; i < params.nrows_exact + params.nrows_sampled / 2; i++) {
      // First set of dataset observations must correspond to nsamples[i]
      for (j = (i + compliment_ctr) * params.nrows_background * params.ncols;
           j < (i + compliment_ctr + 1) * params.nrows_background * params.ncols;
           j += params.ncols) {
        counter              = thrust::count(&d_ptr[j], &d_ptr[j + params.ncols], sent_value);
        test_scatter_sampled = test_scatter_sampled && (counter == n_ptr[i - params.nrows_exact]);
      }

      // The next set of samples must correspond to the compliment: ncols - nsamples[i]
      compliment_ctr++;
      for (j = (i + compliment_ctr) * params.nrows_background * params.ncols;
           j < (i + compliment_ctr + 1) * params.nrows_background * params.ncols;
           j += params.ncols) {
        // Check that number of observation entries corresponds to nsamples.
        counter = thrust::count(&d_ptr[j], &d_ptr[j + params.ncols], sent_value);
        test_scatter_sampled =
          test_scatter_sampled && (counter == params.ncols - n_ptr[i - params.nrows_exact]);
      }
    }
  }

  void TearDown() override
  {
    CUDA_CHECK(cudaFree(background));
    CUDA_CHECK(cudaFree(observation));
    CUDA_CHECK(cudaFree(X));
    CUDA_CHECK(cudaFree(dataset));
    CUDA_CHECK(cudaFree(nsamples));
  }

 protected:
  MakeKSHAPDatasetInputs params;
  T* background;
  T* observation;
  float* X;
  T* dataset;
  int* nsamples;
  int nrows_X;
  bool test_sampled_X;
  bool test_scatter_exact;
  bool test_scatter_sampled;
  std::shared_ptr<raft::mr::device::allocator> allocator;
  raft::handle_t handle;
  cudaStream_t stream;
};

const std::vector<MakeKSHAPDatasetInputs> inputsf = {{10, 10, 12, 2, 3, 1234ULL},
                                                     {10, 0, 12, 2, 3, 1234ULL},
                                                     {100, 50, 200, 10, 10, 1234ULL},
                                                     {100, 0, 200, 10, 10, 1234ULL},
                                                     {0, 10, 12, 2, 3, 1234ULL},
                                                     {0, 50, 200, 10, 10, 1234ULL}

};

typedef MakeKSHAPDatasetTest<float> MakeKSHAPDatasetTestF;
TEST_P(MakeKSHAPDatasetTestF, Result)
{
  ASSERT_TRUE(test_sampled_X);
  // todo (dgd): re-enable assertions
  // disabled due to a sporadic cuda 10.1 fail (by one value in one case!)
  // will be re-enabled soon after 0.17 release
  // ASSERT_TRUE(test_scatter_exact);
  // ASSERT_TRUE(test_scatter_sampled);
}
INSTANTIATE_TEST_CASE_P(MakeKSHAPDatasetTests, MakeKSHAPDatasetTestF, ::testing::ValuesIn(inputsf));

const std::vector<MakeKSHAPDatasetInputs> inputsd = {{10, 10, 12, 2, 3, 1234ULL},
                                                     {10, 0, 12, 2, 3, 1234ULL},
                                                     {100, 50, 200, 10, 10, 1234ULL},
                                                     {100, 0, 200, 10, 10, 1234ULL},
                                                     {0, 10, 12, 2, 3, 1234ULL},
                                                     {0, 50, 200, 10, 10, 1234ULL}};

typedef MakeKSHAPDatasetTest<double> MakeKSHAPDatasetTestD;
TEST_P(MakeKSHAPDatasetTestD, Result)
{
  ASSERT_TRUE(test_sampled_X);
  // todo (dgd): re-enable assertions
  // disabled due to a sporadic cuda 10.1 fail (by one value in one case!)
  // will be re-enabled soon after 0.17 release
  // ASSERT_TRUE(test_scatter_exact);
  // ASSERT_TRUE(test_scatter_sampled);
}
INSTANTIATE_TEST_CASE_P(MakeKSHAPDatasetTests, MakeKSHAPDatasetTestD, ::testing::ValuesIn(inputsd));

}  // end namespace Explainer
}  // end namespace ML
