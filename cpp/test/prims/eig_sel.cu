/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#if CUDART_VERSION >= 10010

#include <common/cudart_utils.h>
#include <gtest/gtest.h>
#include <cuda_utils.cuh>
#include <linalg/eig.cuh>
#include <random/rng.cuh>
#include "test_utils.h"

namespace raft {
namespace linalg {

template <typename T>
struct EigSelInputs {
  T tolerance;
  int len;
  int n_row;
  int n_col;
  unsigned long long int seed;
  int n;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os, const EigSelInputs<T> &dims) {
  return os;
}

template <typename T>
class EigSelTest : public ::testing::TestWithParam<EigSelInputs<T>> {
 protected:
  void SetUp() override {
    raft::handle_t handle;
    stream = handle.get_stream();

    params = ::testing::TestWithParam<EigSelInputs<T>>::GetParam();
    int len = params.len;

    raft::allocate(cov_matrix, len);
    T cov_matrix_h[] = {1.0,  0.9, 0.81, 0.729, 0.9,   1.0,  0.9, 0.81,
                        0.81, 0.9, 1.0,  0.9,   0.729, 0.81, 0.9, 1.0};
    ASSERT(len == 16, "This test only works with 4x4 matrices!");
    raft::update_device(cov_matrix, cov_matrix_h, len, stream);

    raft::allocate(eig_vectors, 12);
    raft::allocate(eig_vals, params.n_col);

    T eig_vectors_ref_h[] = {-0.5123, 0.4874,  0.4874, -0.5123, 0.6498, 0.2789,
                             -0.2789, -0.6498, 0.4874, 0.5123,  0.5123, 0.4874};
    T eig_vals_ref_h[] = {0.1024, 0.3096, 3.5266, 3.5266};

    raft::allocate(eig_vectors_ref, 12);
    raft::allocate(eig_vals_ref, params.n_col);

    raft::update_device(eig_vectors_ref, eig_vectors_ref_h, 12, stream);
    raft::update_device(eig_vals_ref, eig_vals_ref_h, 4, stream);

    eigSelDC(handle, cov_matrix, params.n_row, params.n_col, 3, eig_vectors,
             eig_vals, EigVecMemUsage::OVERWRITE_INPUT, stream);
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(cov_matrix));
    CUDA_CHECK(cudaFree(eig_vectors));
    CUDA_CHECK(cudaFree(eig_vals));
    CUDA_CHECK(cudaFree(eig_vectors_ref));
    CUDA_CHECK(cudaFree(eig_vals_ref));
  }

 protected:
  EigSelInputs<T> params;
  T *cov_matrix, *eig_vectors, *eig_vectors_ref, *eig_vals, *eig_vals_ref;

  cudaStream_t stream;
};

const std::vector<EigSelInputs<float>> inputsf2 = {
  {0.001f, 4 * 4, 4, 4, 1234ULL, 256}};

const std::vector<EigSelInputs<double>> inputsd2 = {
  {0.001, 4 * 4, 4, 4, 1234ULL, 256}};

typedef EigSelTest<float> EigSelTestValF;
TEST_P(EigSelTestValF, Result) {
  ASSERT_TRUE(
    raft::devArrMatch(eig_vals_ref, eig_vals, params.n_col,
                      raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef EigSelTest<double> EigSelTestValD;
TEST_P(EigSelTestValD, Result) {
  ASSERT_TRUE(
    raft::devArrMatch(eig_vals_ref, eig_vals, params.n_col,
                      raft::CompareApproxAbs<double>(params.tolerance)));
}

typedef EigSelTest<float> EigSelTestVecF;
TEST_P(EigSelTestVecF, Result) {
  ASSERT_TRUE(
    raft::devArrMatch(eig_vectors_ref, eig_vectors, 12,
                      raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef EigSelTest<double> EigSelTestVecD;
TEST_P(EigSelTestVecD, Result) {
  ASSERT_TRUE(
    raft::devArrMatch(eig_vectors_ref, eig_vectors, 12,
                      raft::CompareApproxAbs<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(EigSelTest, EigSelTestValF,
                        ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(EigSelTest, EigSelTestValD,
                        ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_CASE_P(EigSelTest, EigSelTestVecF,
                        ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(EigSelTest, EigSelTestVecD,
                        ::testing::ValuesIn(inputsd2));

}  // end namespace linalg
}  // end namespace raft

#endif
