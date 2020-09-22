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
#include <linalg/svd.cuh>
#include <matrix/matrix.cuh>
#include <random/rng.cuh>
#include "test_utils.h"

namespace MLCommon {
namespace LinAlg {

template <typename T>
struct SvdInputs {
  T tolerance;
  int len;
  int n_row;
  int n_col;
  unsigned long long int seed;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os, const SvdInputs<T> &dims) {
  return os;
}

template <typename T>
class SvdTest : public ::testing::TestWithParam<SvdInputs<T>> {
 protected:
  void SetUp() override {
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUBLAS_CHECK(cublasCreate(&cublasH));
    allocator.reset(new raft::mr::device::default_allocator);

    params = ::testing::TestWithParam<SvdInputs<T>>::GetParam();
    Random::Rng r(params.seed);
    int len = params.len;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    allocate(data, len);

    ASSERT(params.n_row == 3, "This test only supports nrows=3!");
    ASSERT(params.len == 6, "This test only supports len=6!");
    T data_h[] = {1.0, 4.0, 2.0, 2.0, 5.0, 1.0};
    updateDevice(data, data_h, len, stream);

    int left_evl = params.n_row * params.n_col;
    int right_evl = params.n_col * params.n_col;

    allocate(left_eig_vectors_qr, left_evl);
    allocate(right_eig_vectors_trans_qr, right_evl);
    allocate(sing_vals_qr, params.n_col);

    // allocate(left_eig_vectors_jacobi, left_evl);
    // allocate(right_eig_vectors_trans_jacobi, right_evl);
    // allocate(sing_vals_jacobi, params.n_col);

    T left_eig_vectors_ref_h[] = {-0.308219, -0.906133, -0.289695,
                                  0.488195,  0.110706,  -0.865685};

    T right_eig_vectors_ref_h[] = {-0.638636, -0.769509, -0.769509, 0.638636};

    T sing_vals_ref_h[] = {7.065283, 1.040081};

    allocate(left_eig_vectors_ref, left_evl);
    allocate(right_eig_vectors_ref, right_evl);
    allocate(sing_vals_ref, params.n_col);

    updateDevice(left_eig_vectors_ref, left_eig_vectors_ref_h, left_evl,
                 stream);
    updateDevice(right_eig_vectors_ref, right_eig_vectors_ref_h, right_evl,
                 stream);
    updateDevice(sing_vals_ref, sing_vals_ref_h, params.n_col, stream);

    svdQR(data, params.n_row, params.n_col, sing_vals_qr, left_eig_vectors_qr,
          right_eig_vectors_trans_qr, true, true, true, cusolverH, cublasH,
          allocator, stream);
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(left_eig_vectors_qr));
    CUDA_CHECK(cudaFree(right_eig_vectors_trans_qr));
    CUDA_CHECK(cudaFree(sing_vals_qr));
    CUDA_CHECK(cudaFree(left_eig_vectors_ref));
    CUDA_CHECK(cudaFree(right_eig_vectors_ref));
    CUDA_CHECK(cudaFree(sing_vals_ref));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUBLAS_CHECK(cublasDestroy(cublasH));
  }

 protected:
  SvdInputs<T> params;
  T *data, *left_eig_vectors_qr, *right_eig_vectors_trans_qr, *sing_vals_qr,
    *left_eig_vectors_ref, *right_eig_vectors_ref, *sing_vals_ref;
  cusolverDnHandle_t cusolverH = NULL;
  cublasHandle_t cublasH;
  std::shared_ptr<deviceAllocator> allocator;
};

const std::vector<SvdInputs<float>> inputsf2 = {
  {0.00001f, 3 * 2, 3, 2, 1234ULL}};

const std::vector<SvdInputs<double>> inputsd2 = {
  {0.00001, 3 * 2, 3, 2, 1234ULL}};

typedef SvdTest<float> SvdTestValF;
TEST_P(SvdTestValF, Result) {
  ASSERT_TRUE(devArrMatch(sing_vals_ref, sing_vals_qr, params.n_col,
                          CompareApproxAbs<float>(params.tolerance)));
}

typedef SvdTest<double> SvdTestValD;
TEST_P(SvdTestValD, Result) {
  ASSERT_TRUE(devArrMatch(sing_vals_ref, sing_vals_qr, params.n_col,
                          CompareApproxAbs<double>(params.tolerance)));
}

typedef SvdTest<float> SvdTestLeftVecF;
TEST_P(SvdTestLeftVecF, Result) {
  ASSERT_TRUE(devArrMatch(left_eig_vectors_ref, left_eig_vectors_qr,
                          params.n_row * params.n_col,
                          CompareApproxAbs<float>(params.tolerance)));
}

typedef SvdTest<double> SvdTestLeftVecD;
TEST_P(SvdTestLeftVecD, Result) {
  ASSERT_TRUE(devArrMatch(left_eig_vectors_ref, left_eig_vectors_qr,
                          params.n_row * params.n_col,
                          CompareApproxAbs<double>(params.tolerance)));
}

typedef SvdTest<float> SvdTestRightVecF;
TEST_P(SvdTestRightVecF, Result) {
  ASSERT_TRUE(devArrMatch(right_eig_vectors_ref, right_eig_vectors_trans_qr,
                          params.n_col * params.n_col,
                          CompareApproxAbs<float>(params.tolerance)));
}

typedef SvdTest<double> SvdTestRightVecD;
TEST_P(SvdTestRightVecD, Result) {
  ASSERT_TRUE(devArrMatch(right_eig_vectors_ref, right_eig_vectors_trans_qr,
                          params.n_col * params.n_col,
                          CompareApproxAbs<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(SvdTests, SvdTestValF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(SvdTests, SvdTestValD, ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_CASE_P(SvdTests, SvdTestLeftVecF,
                        ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(SvdTests, SvdTestLeftVecD,
                        ::testing::ValuesIn(inputsd2));

// INSTANTIATE_TEST_CASE_P(SvdTests, SvdTestRightVecF,
// ::testing::ValuesIn(inputsf2));

// INSTANTIATE_TEST_CASE_P(SvdTests, SvdTestRightVecD,
//::testing::ValuesIn(inputsd2));

}  // end namespace LinAlg
}  // end namespace MLCommon
