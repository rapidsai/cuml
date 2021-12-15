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
#include <raft/cudart_utils.h>
#include <functions/penalty.cuh>
#include <raft/random/rng.hpp>
#include "test_utils.h"

namespace MLCommon {
namespace Functions {

template <typename T>
struct PenaltyInputs {
  T tolerance;
  int len;
};

template <typename T>
class PenaltyTest : public ::testing::TestWithParam<PenaltyInputs<T>> {
 protected:
  void SetUp() override
  {
    params  = ::testing::TestWithParam<PenaltyInputs<T>>::GetParam();
    int len = params.len;

    cudaStream_t stream = 0;
    CUDA_CHECK(cudaStreamCreate(&stream));

    raft::allocate(in, len, stream);
    raft::allocate(out_lasso, 1, stream);
    raft::allocate(out_ridge, 1, stream);
    raft::allocate(out_elasticnet, 1, stream);
    raft::allocate(out_lasso_grad, len, stream);
    raft::allocate(out_ridge_grad, len, stream);
    raft::allocate(out_elasticnet_grad, len, stream);
    raft::allocate(out_lasso_ref, 1, stream);
    raft::allocate(out_ridge_ref, 1, stream);
    raft::allocate(out_elasticnet_ref, 1, stream);
    raft::allocate(out_lasso_grad_ref, len, stream);
    raft::allocate(out_ridge_grad_ref, len, stream);
    raft::allocate(out_elasticnet_grad_ref, len, stream);

    T h_in[len] = {0.1, 0.35, -0.9, -1.4};
    raft::update_device(in, h_in, len, stream);

    T h_out_lasso_ref[1] = {1.65};
    raft::update_device(out_lasso_ref, h_out_lasso_ref, 1, stream);

    T h_out_ridge_ref[1] = {1.741499};
    raft::update_device(out_ridge_ref, h_out_ridge_ref, 1, stream);

    T h_out_elasticnet_ref[1] = {1.695749};
    raft::update_device(out_elasticnet_ref, h_out_elasticnet_ref, 1, stream);

    T h_out_lasso_grad_ref[len] = {0.6, 0.6, -0.6, -0.6};
    raft::update_device(out_lasso_grad_ref, h_out_lasso_grad_ref, len, stream);

    T h_out_ridge_grad_ref[len] = {0.12, 0.42, -1.08, -1.68};
    raft::update_device(out_ridge_grad_ref, h_out_ridge_grad_ref, len, stream);

    T h_out_elasticnet_grad_ref[len] = {0.36, 0.51, -0.84, -1.14};
    raft::update_device(out_elasticnet_grad_ref, h_out_elasticnet_grad_ref, len, stream);

    T alpha    = 0.6;
    T l1_ratio = 0.5;

    lasso(out_lasso, in, len, alpha, stream);
    ridge(out_ridge, in, len, alpha, stream);
    elasticnet(out_elasticnet, in, len, alpha, l1_ratio, stream);
    lassoGrad(out_lasso_grad, in, len, alpha, stream);
    ridgeGrad(out_ridge_grad, in, len, alpha, stream);
    elasticnetGrad(out_elasticnet_grad, in, len, alpha, l1_ratio, stream);
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void TearDown() override
  {
    CUDA_CHECK(cudaFree(in));
    CUDA_CHECK(cudaFree(out_lasso));
    CUDA_CHECK(cudaFree(out_ridge));
    CUDA_CHECK(cudaFree(out_elasticnet));
    CUDA_CHECK(cudaFree(out_lasso_grad));
    CUDA_CHECK(cudaFree(out_ridge_grad));
    CUDA_CHECK(cudaFree(out_elasticnet_grad));
    CUDA_CHECK(cudaFree(out_lasso_ref));
    CUDA_CHECK(cudaFree(out_ridge_ref));
    CUDA_CHECK(cudaFree(out_elasticnet_ref));
    CUDA_CHECK(cudaFree(out_lasso_grad_ref));
    CUDA_CHECK(cudaFree(out_ridge_grad_ref));
    CUDA_CHECK(cudaFree(out_elasticnet_grad_ref));
  }

 protected:
  PenaltyInputs<T> params;
  T* in;
  T *out_lasso, *out_ridge, *out_elasticnet;
  T *out_lasso_ref, *out_ridge_ref, *out_elasticnet_ref;
  T *out_lasso_grad, *out_ridge_grad, *out_elasticnet_grad;
  T *out_lasso_grad_ref, *out_ridge_grad_ref, *out_elasticnet_grad_ref;
};

const std::vector<PenaltyInputs<float>> inputsf = {{0.01f, 4}};

const std::vector<PenaltyInputs<double>> inputsd = {{0.01, 4}};

typedef PenaltyTest<float> PenaltyTestF;
TEST_P(PenaltyTestF, Result)
{
  ASSERT_TRUE(
    devArrMatch(out_lasso_ref, out_lasso, 1, raft::CompareApprox<float>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(
    out_lasso_grad_ref, out_lasso_grad, params.len, raft::CompareApprox<float>(params.tolerance)));

  ASSERT_TRUE(
    devArrMatch(out_ridge_ref, out_ridge, 1, raft::CompareApprox<float>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(
    out_ridge_grad_ref, out_ridge_grad, params.len, raft::CompareApprox<float>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(
    out_elasticnet_ref, out_elasticnet, 1, raft::CompareApprox<float>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_elasticnet_grad_ref,
                          out_elasticnet_grad,
                          params.len,
                          raft::CompareApprox<float>(params.tolerance)));
}

typedef PenaltyTest<double> PenaltyTestD;
TEST_P(PenaltyTestD, Result)
{
  ASSERT_TRUE(
    devArrMatch(out_lasso_ref, out_lasso, 1, raft::CompareApprox<double>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(
    out_lasso_grad_ref, out_lasso_grad, params.len, raft::CompareApprox<double>(params.tolerance)));

  ASSERT_TRUE(
    devArrMatch(out_ridge_ref, out_ridge, 1, raft::CompareApprox<double>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(
    out_ridge_grad_ref, out_ridge_grad, params.len, raft::CompareApprox<double>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(
    out_elasticnet_ref, out_elasticnet, 1, raft::CompareApprox<double>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_elasticnet_grad_ref,
                          out_elasticnet_grad,
                          params.len,
                          raft::CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(PenaltyTests, PenaltyTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(PenaltyTests, PenaltyTestD, ::testing::ValuesIn(inputsd));

}  // end namespace Functions
}  // end namespace MLCommon
