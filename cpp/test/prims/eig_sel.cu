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

#include <gtest/gtest.h>
#include "cuda_utils.h"
#include "linalg/eig.h"
#include "random/rng.h"
#include "test_utils.h"

namespace MLCommon {
namespace LinAlg {

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
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUDA_CHECK(cudaStreamCreate(&stream));
    std::shared_ptr<deviceAllocator> allocator(new defaultDeviceAllocator);

    params = ::testing::TestWithParam<EigSelInputs<T>>::GetParam();
    Random::Rng r(params.seed);
    int len = params.len;

    allocate(cov_matrix, len);
    T cov_matrix_h[] = {1.0,  0.9, 0.81, 0.729, 0.9,   1.0,  0.9, 0.81,
                        0.81, 0.9, 1.0,  0.9,   0.729, 0.81, 0.9, 1.0};
    ASSERT(len == 16, "This test only works with 4x4 matrices!");
    updateDevice(cov_matrix, cov_matrix_h, len, stream);

    allocate(eig_vectors, len);
    allocate(eig_vals, params.n_col);
    allocate(eig_vectors_jacobi, len);
    allocate(eig_vals_jacobi, params.n_col);

    T eig_vectors_ref_h[] = {0.2790, -0.6498, 0.6498, -0.2789, -0.5123, 0.4874,
                             0.4874, -0.5123, 0.6498, 0.2789,  -0.2789, -0.6498,
                             0.4874, 0.5123,  0.5123, 0.4874};
    T eig_vals_ref_h[] = {0.0614, 0.1024, 0.3096, 3.5266};

    allocate(eig_vectors_ref, len);
    allocate(eig_vals_ref, params.n_col);

    updateDevice(eig_vectors_ref, eig_vectors_ref_h, len, stream);
    updateDevice(eig_vals_ref, eig_vals_ref_h, params.n_col, stream);

    eigDC(cov_matrix, params.n_row, params.n_col, eig_vectors, eig_vals,
          cusolverH, stream, allocator);

    T tol = 1.e-7;
    int sweeps = 15;
    eigJacobi(cov_matrix, params.n_row, params.n_col, eig_vectors_jacobi,
              eig_vals_jacobi, tol, sweeps, cusolverH, stream, allocator);

    // test code for comparing two methods
    len = params.n * params.n;
    allocate(cov_matrix_large, len);
    allocate(eig_vectors_large, len);
    allocate(eig_vectors_jacobi_large, len);
    allocate(eig_vals_large, params.n);
    allocate(eig_vals_jacobi_large, params.n);

    r.uniform(cov_matrix_large, len, T(-1.0), T(1.0), stream);

    eigDC(cov_matrix_large, params.n, params.n, eig_vectors_large,
          eig_vals_large, cusolverH, stream, allocator);
    eigJacobi(cov_matrix_large, params.n, params.n, eig_vectors_jacobi_large,
              eig_vals_jacobi_large, tol, sweeps, cusolverH, stream, allocator);
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(cov_matrix));
    CUDA_CHECK(cudaFree(eig_vectors));
    CUDA_CHECK(cudaFree(eig_vectors_jacobi));
    CUDA_CHECK(cudaFree(eig_vals));
    CUDA_CHECK(cudaFree(eig_vals_jacobi));
    CUDA_CHECK(cudaFree(eig_vectors_ref));
    CUDA_CHECK(cudaFree(eig_vals_ref));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

 protected:
  EigSelInputs<T> params;
  T *cov_matrix, *eig_vectors, *eig_vectors_jacobi, *eig_vectors_ref, *eig_vals,
    *eig_vals_jacobi, *eig_vals_ref;

  T *cov_matrix_large, *eig_vectors_large, *eig_vectors_jacobi_large,
    *eig_vals_large, *eig_vals_jacobi_large;

  cusolverDnHandle_t cusolverH = NULL;
  cudaStream_t stream;
};

const std::vector<EigSelInputs<float>> inputsf2 = {
  {0.001f, 4 * 4, 4, 4, 1234ULL, 256}};

const std::vector<EigSelInputs<double>> inputsd2 = {
  {0.001, 4 * 4, 4, 4, 1234ULL, 256}};

typedef EigSelTest<float> EigSelTestValF;
TEST_P(EigSelTestValF, Result) {
  ASSERT_TRUE(devArrMatch(eig_vals_ref, eig_vals, params.n_col,
                          CompareApproxAbs<float>(params.tolerance)));
}

typedef EigSelTest<double> EigSelTestValD;
TEST_P(EigSelTestValD, Result) {
  ASSERT_TRUE(devArrMatch(eig_vals_ref, eig_vals, params.n_col,
                          CompareApproxAbs<double>(params.tolerance)));
}

typedef EigSelTest<float> EigSelTestVecF;
TEST_P(EigSelTestVecF, Result) {
  ASSERT_TRUE(devArrMatch(eig_vectors_ref, eig_vectors, params.len,
                          CompareApproxAbs<float>(params.tolerance)));
}

typedef EigSelTest<double> EigSelTestVecD;
TEST_P(EigSelTestVecD, Result) {
  ASSERT_TRUE(devArrMatch(eig_vectors_ref, eig_vectors, params.len,
                          CompareApproxAbs<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(EigSelTest, EigSelTestValF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(EigSelTest, EigSelTestValD, ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_CASE_P(EigSelTest, EigSelTestVecF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(EigSelTest, EigSelTestVecD, ::testing::ValuesIn(inputsd2));

}  // end namespace LinAlg
}  // end namespace MLCommon
