/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include "cuML.hpp"

#include <cuda_utils.h>
#include <gtest/gtest.h>
#include <vector>

#include "linalg/cublas_wrappers.h"
#include "linalg/transpose.h"

#include "datasets/make_blobs.hpp"
#include "dbscan/dbscan.hpp"
#include "metrics/metrics.hpp"

#include "ml_utils.h"
#include "test_utils.h"

#include "common/cuml_allocator.hpp"
#include "common/device_buffer.hpp"

namespace ML {

using namespace MLCommon;
using namespace Datasets;
using namespace Metrics;
using namespace std;

template <typename T>
struct DbscanInputs {
  int n_row;
  int n_col;
  int n_centers;
  T cluster_std;
  T eps;
  int min_pts;
  size_t max_bytes_per_batch;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const DbscanInputs<T>& dims) {
  return os;
}

template <typename T>
class DbscanTest : public ::testing::TestWithParam<DbscanInputs<T>> {
 protected:
  void basicTest() {
    cumlHandle handle;

    params = ::testing::TestWithParam<DbscanInputs<T>>::GetParam();

    device_buffer<T> out(handle.getDeviceAllocator(), handle.getStream(),
                         params.n_row * params.n_col);
    device_buffer<int> l(handle.getDeviceAllocator(), handle.getStream(),
                         params.n_row);

    make_blobs(handle, out.data(), l.data(), params.n_row, params.n_col,
               params.n_centers, nullptr, nullptr, params.cluster_std, true,
               -10.0f, 10.0f, 1234ULL, false);

    allocate(labels, params.n_row);
    allocate(labels_ref, params.n_row);

    MLCommon::copy(labels_ref, l.data(), params.n_row, handle.getStream());

    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));

    dbscanFit(handle, out.data(), params.n_row, params.n_col, params.eps,
              params.min_pts, labels, params.max_bytes_per_batch, false);

    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));

    score = adjustedRandIndex(handle, labels_ref, labels, params.n_row, 0,
                              params.n_centers - 1);

    if (score < 1.0) {
      std::cout << "y: "
                << arr2Str(labels_ref, 25, "labels_ref", handle.getStream())
                << std::endl;
      std::cout << "y_hat: "
                << arr2Str(labels, 25, "labels", handle.getStream())
                << std::endl;
    }
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {
    CUDA_CHECK(cudaFree(labels));
    CUDA_CHECK(cudaFree(labels_ref));
  }

 protected:
  DbscanInputs<T> params;
  int *labels, *labels_ref;

  double score;
};

const std::vector<DbscanInputs<float>> inputsf2 = {
  {50000, 16, 5, 0.01, 2, 2, (size_t)13e8, 1234ULL},
  {500, 16, 5, 0.01, 2, 2, (size_t)100, 1234ULL},
  {1000, 1000, 10, 0.01, 2, 2, (size_t)13e8, 1234ULL},
  {50000, 16, 5, 0.01, 2, 2, (size_t)13e8, 1234ULL},
  {20000, 10000, 10, 0.01, 2, 2, (size_t)13e8, 1234ULL},
  {20000, 100, 5000, 0.01, 2, 2, (size_t)13e8, 1234ULL}};

const std::vector<DbscanInputs<double>> inputsd2 = {
  {50000, 16, 5, 0.01, 2, 2, (size_t)13e9, 1234ULL},
  {500, 16, 5, 0.01, 2, 2, (size_t)100, 1234ULL},
  {1000, 1000, 10, 0.01, 2, 2, (size_t)13e9, 1234ULL},
  {100, 10000, 10, 0.01, 2, 2, (size_t)13e9, 1234ULL},
  {20000, 10000, 10, 0.01, 2, 2, (size_t)13e9, 1234ULL},
  {20000, 100, 5000, 0.01, 2, 2, (size_t)13e9, 1234ULL}};

typedef DbscanTest<float> DbscanTestF;
TEST_P(DbscanTestF, Result) { ASSERT_TRUE(score == 1.0); }

typedef DbscanTest<double> DbscanTestD;
TEST_P(DbscanTestD, Result) { ASSERT_TRUE(score == 1.0); }

INSTANTIATE_TEST_CASE_P(DbscanTests, DbscanTestF,
                        ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(DbscanTests, DbscanTestD,
                        ::testing::ValuesIn(inputsd2));

}  // end namespace ML
