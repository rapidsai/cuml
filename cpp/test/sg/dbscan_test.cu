/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

#include <cuda_utils.h>
#include <gtest/gtest.h>
#include <vector>

#include <cuml/cluster/dbscan.hpp>
#include <cuml/common/cuml_allocator.hpp>
#include <cuml/cuml.hpp>
#include <cuml/datasets/make_blobs.hpp>
#include <cuml/metrics/metrics.hpp>

#include "linalg/cublas_wrappers.h"
#include "linalg/transpose.h"

#include "ml_utils.h"
#include "test_utils.h"

#include "common/device_buffer.hpp"

namespace ML {

using namespace MLCommon;
using namespace Datasets;
using namespace Metrics;
using namespace std;

template <typename T, typename IdxT>
struct DbscanInputs {
  IdxT n_row;
  IdxT n_col;
  IdxT n_centers;
  T cluster_std;
  T eps;
  int min_pts;
  size_t max_bytes_per_batch;
  unsigned long long int seed;
};

template <typename T, typename IdxT>
::std::ostream& operator<<(::std::ostream& os,
                           const DbscanInputs<T, IdxT>& dims) {
  return os;
}

template <typename T, typename IdxT>
class DbscanTest : public ::testing::TestWithParam<DbscanInputs<T, IdxT>> {
 protected:
  void basicTest() {
    cumlHandle handle;

    params = ::testing::TestWithParam<DbscanInputs<T, IdxT>>::GetParam();

    device_buffer<T> out(handle.getDeviceAllocator(), handle.getStream(),
                         params.n_row * params.n_col);
    device_buffer<IdxT> l(handle.getDeviceAllocator(), handle.getStream(),
                          params.n_row);

    make_blobs(handle, out.data(), l.data(), params.n_row, params.n_col,
               params.n_centers, nullptr, nullptr, params.cluster_std, true,
               -10.0f, 10.0f, 1234ULL);

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

      std::cout << "Score = " << score << std::endl;
    }
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {
    CUDA_CHECK(cudaFree(labels));
    CUDA_CHECK(cudaFree(labels_ref));
  }

 protected:
  DbscanInputs<T, IdxT> params;
  IdxT *labels, *labels_ref;

  double score;
};

const std::vector<DbscanInputs<float, int>> inputsf2 = {
  {50000, 16, 5, 0.01, 2, 2, (size_t)13e3, 1234ULL},
  {500, 16, 5, 0.01, 2, 2, (size_t)100, 1234ULL},
  {1000, 1000, 10, 0.01, 2, 2, (size_t)13e3, 1234ULL},
  {50000, 16, 5l, 0.01, 2, 2, (size_t)13e3, 1234ULL},
  {20000, 10000, 10, 0.01, 2, 2, (size_t)13e3, 1234ULL},
  {20000, 100, 5000, 0.01, 2, 2, (size_t)13e3, 1234ULL}};

const std::vector<DbscanInputs<float, int64_t>> inputsf3 = {
  {50000, 16, 5, 0.01, 2, 2, (size_t)9e3, 1234ULL},
  {500, 16, 5, 0.01, 2, 2, (size_t)100, 1234ULL},
  {1000, 1000, 10, 0.01, 2, 2, (size_t)9e3, 1234ULL},
  {50000, 16, 5l, 0.01, 2, 2, (size_t)9e3, 1234ULL},
  {20000, 10000, 10, 0.01, 2, 2, (size_t)9e3, 1234ULL},
  {20000, 100, 5000, 0.01, 2, 2, (size_t)9e3, 1234ULL}};

const std::vector<DbscanInputs<double, int>> inputsd2 = {
  {50000, 16, 5, 0.01, 2, 2, (size_t)13e3, 1234ULL},
  {500, 16, 5, 0.01, 2, 2, (size_t)100, 1234ULL},
  {1000, 1000, 10, 0.01, 2, 2, (size_t)13e3, 1234ULL},
  {100, 10000, 10, 0.01, 2, 2, (size_t)13e3, 1234ULL},
  {20000, 10000, 10, 0.01, 2, 2, (size_t)13e3, 1234ULL},
  {20000, 100, 5000, 0.01, 2, 2, (size_t)13e3, 1234ULL}};

const std::vector<DbscanInputs<double, int64_t>> inputsd3 = {
  {50000, 16, 5, 0.01, 2, 2, (size_t)9e3, 1234ULL},
  {500, 16, 5, 0.01, 2, 2, (size_t)100, 1234ULL},
  {1000, 1000, 10, 0.01, 2, 2, (size_t)9e3, 1234ULL},
  {100, 10000, 10, 0.01, 2, 2, (size_t)9e3, 1234ULL},
  {20000, 10000, 10, 0.01, 2, 2, (size_t)9e3, 1234ULL},
  {20000, 100, 5000, 0.01, 2, 2, (size_t)9e3, 1234ULL}};

typedef DbscanTest<float, int> DbscanTestF_Int;
TEST_P(DbscanTestF_Int, Result) { ASSERT_TRUE(score == 1.0); }

typedef DbscanTest<float, int64_t> DbscanTestF_Int64;
TEST_P(DbscanTestF_Int64, Result) { ASSERT_TRUE(score == 1.0); }

typedef DbscanTest<double, int> DbscanTestD_Int;
TEST_P(DbscanTestD_Int, Result) { ASSERT_TRUE(score == 1.0); }

typedef DbscanTest<double, int64_t> DbscanTestD_Int64;
TEST_P(DbscanTestD_Int64, Result) { ASSERT_TRUE(score == 1.0); }

INSTANTIATE_TEST_CASE_P(DbscanTests, DbscanTestF_Int,
                        ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(DbscanTests, DbscanTestF_Int64,
                        ::testing::ValuesIn(inputsf3));

INSTANTIATE_TEST_CASE_P(DbscanTests, DbscanTestD_Int,
                        ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_CASE_P(DbscanTests, DbscanTestD_Int64,
                        ::testing::ValuesIn(inputsd3));

}  // end namespace ML
