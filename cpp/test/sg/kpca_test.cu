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

#include <gtest/gtest.h>
#include <raft/cudart_utils.h>
#include <raft/linalg/cublas_wrappers.h>
#include <test_utils.h>
#include <cuml/decomposition/params.hpp>
#include <kpca/kpca.cuh>
#include <raft/cuda_utils.cuh>
#include <raft/random/rng.cuh>
#include <vector>

namespace ML {

using namespace MLCommon;

template <typename T>
struct KPcaInputs {
  T tolerance;
  int n_rows;
  int n_cols;
  int n_components;
  int algo;
  MLCommon::Matrix::KernelParams kernel;
  std::vector<T> data_h;
  std::vector<T> alphas_ref_h;
  std::vector<T> lambdas_ref_h;
  std::vector<T> trans_data_ref_h;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const KPcaInputs<T>& dims) {
  return os;
}

template <typename T>
class KPcaTest : public ::testing::TestWithParam<KPcaInputs<T>> {
 protected:
  void basicTest() {
    params = ::testing::TestWithParam<KPcaInputs<T>>::GetParam();
    raft::allocate(data, params.n_rows * params.n_cols);
    raft::allocate(trans_data, params.n_rows * params.n_rows);
    raft::allocate(alphas, params.n_rows * params.n_rows);
    raft::allocate(lambdas, params.n_rows);
    raft::allocate(trans_data_ref, params.n_rows * params.n_rows);
    raft::allocate(alphas_ref, params.n_rows * params.n_rows);
    raft::allocate(lambdas_ref, params.n_rows);
    raft::update_device(data, params.data_h.data(), params.n_rows * params.n_cols, stream);
    raft::update_device(trans_data_ref, params.trans_data_ref_h.data(), params.n_rows * params.n_rows, stream);
    raft::update_device(lambdas_ref, params.lambdas_ref_h.data(), params.n_rows, stream);
    raft::update_device(alphas_ref, params.alphas_ref_h.data(), params.n_rows * params.n_rows, stream);

    paramsKPCA prms;
    prms.n_rows = params.n_rows;
    prms.n_cols = params.n_cols;
    prms.n_components = params.n_components;
    prms.kernel = params.kernel;
    if (params.algo == 0)
      prms.algorithm = solver::COV_EIG_DQ;
    else
      prms.algorithm = solver::COV_EIG_JACOBI;
    
    kpcaFit(handle, data, alphas, lambdas, prms, stream);
    kpcaTransform(handle, data, alphas, lambdas, trans_data, prms, stream);
  }

  void SetUp() override {
    CUDA_CHECK(cudaStreamCreate(&stream));
    handle.set_stream(stream);
    basicTest();
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(trans_data));
    CUDA_CHECK(cudaFree(alphas));
    CUDA_CHECK(cudaFree(lambdas));
    CUDA_CHECK(cudaFree(trans_data_ref));
    CUDA_CHECK(cudaFree(alphas_ref));
    CUDA_CHECK(cudaFree(lambdas_ref));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

 protected:
  KPcaInputs<T> params;
  T *data, *trans_data, *data_back, *alphas, *lambdas, 
    *trans_data_ref, *alphas_ref, *lambdas_ref;
  
  raft::handle_t handle;
  cudaStream_t stream;
};


float tolerance = 0.01f;
int n_rows = 3;
int n_cols = 2;
int n_components = 3;
int algo = 1;
std::vector<float> data_h = {1.0, 2.0, 5.0, 4.0, 2.0, 1.0};

MLCommon::Matrix::KernelParams lin_kern = {Matrix::LINEAR, 0, 0, 0};
std::vector<float> lin_alpha_ref_h = {-0.6525, -0.0987, 0.7513, -0.4907, 0.8105, -0.3197};
std::vector<float> lin_lambda_ref_h = {12.6759, 0.6574};
std::vector<float> lin_trans_data_ref_h = {-2.32318647,-0.35170213, 2.6748886, -0.39794495, 0.65716145,-0.25921649};
KPcaInputs<float> linear_inputs = {tolerance, n_rows, n_cols, n_components, algo, lin_kern
                                  , data_h, lin_alpha_ref_h, lin_lambda_ref_h, lin_trans_data_ref_h};

MLCommon::Matrix::KernelParams poly_kern = {Matrix::POLYNOMIAL, 3, 1.0/2.0f, 1};
std::vector<float> poly_alpha_ref_h = {-0.5430, -0.2565, 0.7995, -0.6097, 0.7751, -0.1653};
std::vector<float> poly_lambda_ref_h = {1790.3207, 210.3639};
std::vector<float> poly_trans_data_ref_h = {-22.9760, -10.8554, 33.8314, -8.8438, 11.2426, -2.3987};
KPcaInputs<float> poly_inputs = {tolerance, n_rows, n_cols, n_components, algo, poly_kern
                              , data_h, poly_alpha_ref_h, poly_lambda_ref_h, poly_trans_data_ref_h};

MLCommon::Matrix::KernelParams rbf_kern = {Matrix::RBF, 0, 1.0/2.0f, 0};
std::vector<float> rbf_alpha_ref_h = {-0.4341, -0.3818, 0.8159, -0.6915, 0.7217, -0.0301};
std::vector<float> rbf_lambda_ref_h = {1.0230, 0.9177};
std::vector<float> rbf_trans_data_ref_h = {-0.4391, -0.3862, 0.8253, -0.6624, 0.6914, -0.0289};
KPcaInputs<float> rbf_inputs = {tolerance, n_rows, n_cols, n_components, algo, rbf_kern
                              , data_h, rbf_alpha_ref_h, rbf_lambda_ref_h, rbf_trans_data_ref_h};

const std::vector<KPcaInputs<float>> inputs_f = {linear_inputs, poly_inputs, rbf_inputs};


typedef KPcaTest<float> KPcaTestLambdasF;
TEST_P(KPcaTestLambdasF, Result) {
  ASSERT_TRUE(devArrMatch(lambdas, lambdas_ref, params.n_cols,
                          raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef KPcaTest<float> KPcaTestAlphasF;
TEST_P(KPcaTestAlphasF, Result) {
  ASSERT_TRUE(devArrMatch(alphas, alphas_ref,
                          (params.n_rows * params.n_cols),
                          raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef KPcaTest<float> KPcaTestTransDataF;
TEST_P(KPcaTestTransDataF, Result) {
  ASSERT_TRUE(devArrMatch(trans_data, trans_data_ref,
                          (params.n_rows * params.n_cols),
                          raft::CompareApproxAbs<float>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(KPcaTests, KPcaTestLambdasF, ::testing::ValuesIn(inputs_f));

INSTANTIATE_TEST_CASE_P(KPcaTests, KPcaTestAlphasF,
                        ::testing::ValuesIn(inputs_f));

INSTANTIATE_TEST_CASE_P(KPcaTests, KPcaTestTransDataF,
                        ::testing::ValuesIn(inputs_f));

}  // end namespace ML
