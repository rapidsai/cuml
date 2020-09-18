/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <functions/hinge.cuh>
#include <random/rng.cuh>
#include "test_utils.h"

namespace MLCommon {
namespace Functions {

template <typename T>
struct HingeLossInputs {
  T tolerance;
  T n_rows;
  T n_cols;
  int len;
};

template <typename T>
class HingeLossTest : public ::testing::TestWithParam<HingeLossInputs<T>> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<HingeLossInputs<T>>::GetParam();
    int len = params.len;
    int n_rows = params.n_rows;
    int n_cols = params.n_cols;

    T *labels, *coef;

    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    allocator.reset(new raft::mr::device::default_allocator);

    allocate(in, len);
    allocate(out, 1);
    allocate(out_lasso, 1);
    allocate(out_ridge, 1);
    allocate(out_elasticnet, 1);
    allocate(out_grad, n_cols);
    allocate(out_lasso_grad, n_cols);
    allocate(out_ridge_grad, n_cols);
    allocate(out_elasticnet_grad, n_cols);
    allocate(out_ref, 1);
    allocate(out_lasso_ref, 1);
    allocate(out_ridge_ref, 1);
    allocate(out_elasticnet_ref, 1);
    allocate(out_grad_ref, n_cols);
    allocate(out_lasso_grad_ref, n_cols);
    allocate(out_ridge_grad_ref, n_cols);
    allocate(out_elasticnet_grad_ref, n_cols);

    allocate(labels, params.n_rows);
    allocate(coef, params.n_cols);

    T h_in[len] = {0.1, 0.35, -0.9, -1.4, 2.0, 3.1};
    updateDevice(in, h_in, len, stream);

    T h_labels[n_rows] = {0.3, 2.0, -1.1};
    updateDevice(labels, h_labels, n_rows, stream);

    T h_coef[n_cols] = {0.35, -0.24};
    updateDevice(coef, h_coef, n_cols, stream);

    T h_out_ref[1] = {2.6037};
    updateDevice(out_ref, h_out_ref, 1, stream);

    T h_out_lasso_ref[1] = {2.9577};
    updateDevice(out_lasso_ref, h_out_lasso_ref, 1, stream);

    T h_out_ridge_ref[1] = {2.71176};
    updateDevice(out_ridge_ref, h_out_ridge_ref, 1, stream);

    T h_out_elasticnet_ref[1] = {2.83473};
    updateDevice(out_elasticnet_ref, h_out_elasticnet_ref, 1, stream);

    T h_out_grad_ref[n_cols] = {-0.24333, -1.1933};
    updateDevice(out_grad_ref, h_out_grad_ref, n_cols, stream);

    T h_out_lasso_grad_ref[n_cols] = {0.3566, -1.7933};
    updateDevice(out_lasso_grad_ref, h_out_lasso_grad_ref, n_cols, stream);

    T h_out_ridge_grad_ref[n_cols] = {0.1766, -1.4813};
    updateDevice(out_ridge_grad_ref, h_out_ridge_grad_ref, n_cols, stream);

    T h_out_elasticnet_grad_ref[n_cols] = {0.2666, -1.63733};
    updateDevice(out_elasticnet_grad_ref, h_out_elasticnet_grad_ref, n_cols,
                 stream);

    T alpha = 0.6;
    T l1_ratio = 0.5;

    hingeLoss(in, params.n_rows, params.n_cols, labels, coef, out,
              penalty::NONE, alpha, l1_ratio, cublas_handle, allocator, stream);

    updateDevice(in, h_in, len, stream);

    hingeLossGrads(in, params.n_rows, params.n_cols, labels, coef, out_grad,
                   penalty::NONE, alpha, l1_ratio, cublas_handle, allocator,
                   stream);

    updateDevice(in, h_in, len, stream);

    hingeLoss(in, params.n_rows, params.n_cols, labels, coef, out_lasso,
              penalty::L1, alpha, l1_ratio, cublas_handle, allocator, stream);

    updateDevice(in, h_in, len, stream);

    hingeLossGrads(in, params.n_rows, params.n_cols, labels, coef,
                   out_lasso_grad, penalty::L1, alpha, l1_ratio, cublas_handle,
                   allocator, stream);

    updateDevice(in, h_in, len, stream);

    hingeLoss(in, params.n_rows, params.n_cols, labels, coef, out_ridge,
              penalty::L2, alpha, l1_ratio, cublas_handle, allocator, stream);

    hingeLossGrads(in, params.n_rows, params.n_cols, labels, coef,
                   out_ridge_grad, penalty::L2, alpha, l1_ratio, cublas_handle,
                   allocator, stream);

    updateDevice(in, h_in, len, stream);

    hingeLoss(in, params.n_rows, params.n_cols, labels, coef, out_elasticnet,
              penalty::ELASTICNET, alpha, l1_ratio, cublas_handle, allocator,
              stream);

    hingeLossGrads(in, params.n_rows, params.n_cols, labels, coef,
                   out_elasticnet_grad, penalty::ELASTICNET, alpha, l1_ratio,
                   cublas_handle, allocator, stream);

    updateDevice(in, h_in, len, stream);

    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(labels));
    CUDA_CHECK(cudaFree(coef));
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(in));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(out_lasso));
    CUDA_CHECK(cudaFree(out_ridge));
    CUDA_CHECK(cudaFree(out_elasticnet));
    CUDA_CHECK(cudaFree(out_grad));
    CUDA_CHECK(cudaFree(out_lasso_grad));
    CUDA_CHECK(cudaFree(out_ridge_grad));
    CUDA_CHECK(cudaFree(out_elasticnet_grad));
    CUDA_CHECK(cudaFree(out_ref));
    CUDA_CHECK(cudaFree(out_lasso_ref));
    CUDA_CHECK(cudaFree(out_ridge_ref));
    CUDA_CHECK(cudaFree(out_elasticnet_ref));
    CUDA_CHECK(cudaFree(out_grad_ref));
    CUDA_CHECK(cudaFree(out_lasso_grad_ref));
    CUDA_CHECK(cudaFree(out_ridge_grad_ref));
    CUDA_CHECK(cudaFree(out_elasticnet_grad_ref));
  }

 protected:
  HingeLossInputs<T> params;
  T *in;
  T *out, *out_lasso, *out_ridge, *out_elasticnet;
  T *out_ref, *out_lasso_ref, *out_ridge_ref, *out_elasticnet_ref;
  T *out_grad, *out_lasso_grad, *out_ridge_grad, *out_elasticnet_grad;
  T *out_grad_ref, *out_lasso_grad_ref, *out_ridge_grad_ref,
    *out_elasticnet_grad_ref;
  std::shared_ptr<deviceAllocator> allocator;
};

const std::vector<HingeLossInputs<float>> inputsf = {{0.01f, 3, 2, 6}};

const std::vector<HingeLossInputs<double>> inputsd = {{0.01, 3, 2, 6}};

typedef HingeLossTest<float> HingeLossTestF;
TEST_P(HingeLossTestF, Result) {
  ASSERT_TRUE(
    devArrMatch(out_ref, out, 1, CompareApprox<float>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_lasso_ref, out_lasso, 1,
                          CompareApprox<float>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_ridge_ref, out_ridge, 1,
                          CompareApprox<float>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_elasticnet_ref, out_elasticnet, 1,
                          CompareApprox<float>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_grad_ref, out_grad, params.n_cols,
                          CompareApprox<float>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_lasso_grad_ref, out_lasso_grad, params.n_cols,
                          CompareApprox<float>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_ridge_grad_ref, out_ridge_grad, params.n_cols,
                          CompareApprox<float>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_elasticnet_grad_ref, out_elasticnet_grad,
                          params.n_cols,
                          CompareApprox<float>(params.tolerance)));
}

typedef HingeLossTest<double> HingeLossTestD;
TEST_P(HingeLossTestD, Result) {
  ASSERT_TRUE(
    devArrMatch(out_ref, out, 1, CompareApprox<double>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_lasso_ref, out_lasso, 1,
                          CompareApprox<double>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_ridge_ref, out_ridge, 1,
                          CompareApprox<double>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_elasticnet_ref, out_elasticnet, 1,
                          CompareApprox<double>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_grad_ref, out_grad, params.n_cols,
                          CompareApprox<double>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_lasso_grad_ref, out_lasso_grad, params.n_cols,
                          CompareApprox<double>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_ridge_grad_ref, out_ridge_grad, params.n_cols,
                          CompareApprox<double>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_elasticnet_grad_ref, out_elasticnet_grad,
                          params.n_cols,
                          CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(HingeLossTests, HingeLossTestF,
                        ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(HingeLossTests, HingeLossTestD,
                        ::testing::ValuesIn(inputsd));

}  // end namespace Functions
}  // end namespace MLCommon
