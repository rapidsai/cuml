/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

#include <common/cudart_utils.h>
#include <gtest/gtest.h>
#include <cuda_utils.cuh>
#include <linalg/eig.cuh>
#include <random/rng.cuh>
#include "test_utils.h"

namespace raft {
namespace linalg {

template <typename T>
struct EigInputs {
  T tolerance;
  int len;
  int n_row;
  int n_col;
  unsigned long long int seed;
  int n;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os, const EigInputs<T> &dims) {
  return os;
}

template <typename T>
class EigTest : public ::testing::TestWithParam<EigInputs<T>> {
 protected:
  void SetUp() override {
    raft::handle_t handle;
    stream = handle.get_stream();

    params = ::testing::TestWithParam<EigInputs<T>>::GetParam();
    raft::random::Rng r(params.seed);
    int len = params.len;

    raft::allocate(cov_matrix, len);
    T cov_matrix_h[] = {1.0,  0.9, 0.81, 0.729, 0.9,   1.0,  0.9, 0.81,
                        0.81, 0.9, 1.0,  0.9,   0.729, 0.81, 0.9, 1.0};
    ASSERT(len == 16, "This test only works with 4x4 matrices!");
    raft::update_device(cov_matrix, cov_matrix_h, len, stream);

    raft::allocate(eig_vectors, len);
    raft::allocate(eig_vals, params.n_col);
    raft::allocate(eig_vectors_jacobi, len);
    raft::allocate(eig_vals_jacobi, params.n_col);

    T eig_vectors_ref_h[] = {0.2790, -0.6498, 0.6498, -0.2789, -0.5123, 0.4874,
                             0.4874, -0.5123, 0.6498, 0.2789,  -0.2789, -0.6498,
                             0.4874, 0.5123,  0.5123, 0.4874};
    T eig_vals_ref_h[] = {0.0614, 0.1024, 0.3096, 3.5266};

    raft::allocate(eig_vectors_ref, len);
    raft::allocate(eig_vals_ref, params.n_col);

    raft::update_device(eig_vectors_ref, eig_vectors_ref_h, len, stream);
    raft::update_device(eig_vals_ref, eig_vals_ref_h, params.n_col, stream);

    eigDC(handle, cov_matrix, params.n_row, params.n_col, eig_vectors, eig_vals,
          stream);

    T tol = 1.e-7;
    int sweeps = 15;
    eigJacobi(handle, cov_matrix, params.n_row, params.n_col,
              eig_vectors_jacobi, eig_vals_jacobi, stream, tol, sweeps);

    // test code for comparing two methods
    len = params.n * params.n;
    raft::allocate(cov_matrix_large, len);
    raft::allocate(eig_vectors_large, len);
    raft::allocate(eig_vectors_jacobi_large, len);
    raft::allocate(eig_vals_large, params.n);
    raft::allocate(eig_vals_jacobi_large, params.n);

    r.uniform(cov_matrix_large, len, T(-1.0), T(1.0), stream);

    eigDC(handle, cov_matrix_large, params.n, params.n, eig_vectors_large,
          eig_vals_large, stream);
    eigJacobi(handle, cov_matrix_large, params.n, params.n,
              eig_vectors_jacobi_large, eig_vals_jacobi_large, stream, tol,
              sweeps);
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(cov_matrix));
    CUDA_CHECK(cudaFree(eig_vectors));
    CUDA_CHECK(cudaFree(eig_vectors_jacobi));
    CUDA_CHECK(cudaFree(eig_vals));
    CUDA_CHECK(cudaFree(eig_vals_jacobi));
    CUDA_CHECK(cudaFree(eig_vectors_ref));
    CUDA_CHECK(cudaFree(eig_vals_ref));
  }

 protected:
  EigInputs<T> params;
  T *cov_matrix, *eig_vectors, *eig_vectors_jacobi, *eig_vectors_ref, *eig_vals,
    *eig_vals_jacobi, *eig_vals_ref;

  T *cov_matrix_large, *eig_vectors_large, *eig_vectors_jacobi_large,
    *eig_vals_large, *eig_vals_jacobi_large;

  cudaStream_t stream;
};

const std::vector<EigInputs<float>> inputsf2 = {
  {0.001f, 4 * 4, 4, 4, 1234ULL, 256}};

const std::vector<EigInputs<double>> inputsd2 = {
  {0.001, 4 * 4, 4, 4, 1234ULL, 256}};

typedef EigTest<float> EigTestValF;
TEST_P(EigTestValF, Result) {
  ASSERT_TRUE(
    raft::devArrMatch(eig_vals_ref, eig_vals, params.n_col,
                      raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef EigTest<double> EigTestValD;
TEST_P(EigTestValD, Result) {
  ASSERT_TRUE(
    raft::devArrMatch(eig_vals_ref, eig_vals, params.n_col,
                      raft::CompareApproxAbs<double>(params.tolerance)));
}

typedef EigTest<float> EigTestVecF;
TEST_P(EigTestVecF, Result) {
  ASSERT_TRUE(
    raft::devArrMatch(eig_vectors_ref, eig_vectors, params.len,
                      raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef EigTest<double> EigTestVecD;
TEST_P(EigTestVecD, Result) {
  ASSERT_TRUE(
    raft::devArrMatch(eig_vectors_ref, eig_vectors, params.len,
                      raft::CompareApproxAbs<double>(params.tolerance)));
}

typedef EigTest<float> EigTestValJacobiF;
TEST_P(EigTestValJacobiF, Result) {
  ASSERT_TRUE(
    raft::devArrMatch(eig_vals_ref, eig_vals_jacobi, params.n_col,
                      raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef EigTest<double> EigTestValJacobiD;
TEST_P(EigTestValJacobiD, Result) {
  ASSERT_TRUE(
    raft::devArrMatch(eig_vals_ref, eig_vals_jacobi, params.n_col,
                      raft::CompareApproxAbs<double>(params.tolerance)));
}

typedef EigTest<float> EigTestVecJacobiF;
TEST_P(EigTestVecJacobiF, Result) {
  ASSERT_TRUE(
    raft::devArrMatch(eig_vectors_ref, eig_vectors_jacobi, params.len,
                      raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef EigTest<double> EigTestVecJacobiD;
TEST_P(EigTestVecJacobiD, Result) {
  ASSERT_TRUE(
    raft::devArrMatch(eig_vectors_ref, eig_vectors_jacobi, params.len,
                      raft::CompareApproxAbs<double>(params.tolerance)));
}

typedef EigTest<float> EigTestVecCompareF;
TEST_P(EigTestVecCompareF, Result) {
  ASSERT_TRUE(raft::devArrMatch(
    eig_vectors_large, eig_vectors_jacobi_large, (params.n * params.n),
    raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef EigTest<double> EigTestVecCompareD;
TEST_P(EigTestVecCompareD, Result) {
  ASSERT_TRUE(raft::devArrMatch(
    eig_vectors_large, eig_vectors_jacobi_large, (params.n * params.n),
    raft::CompareApproxAbs<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(EigTests, EigTestValF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(EigTests, EigTestValD, ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_CASE_P(EigTests, EigTestVecF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(EigTests, EigTestVecD, ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_CASE_P(EigTests, EigTestValJacobiF,
                        ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(EigTests, EigTestValJacobiD,
                        ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_CASE_P(EigTests, EigTestVecJacobiF,
                        ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(EigTests, EigTestVecJacobiD,
                        ::testing::ValuesIn(inputsd2));

}  // namespace linalg
}  // namespace raft
