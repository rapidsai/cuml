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
  int len;
  int n_row;
  int n_col;
  int len2;
  int n_row2;
  int n_col2;
  unsigned long long int seed;
  int algo;
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
    raft::random::Rng r(params.seed, raft::random::GenTaps);
    int len = params.len;

    std::cout << "basicTest - 1 \n";
    std::cout << "test params - tolerance: " << params.tolerance << " \n";
    std::cout << "test params - length: " << params.len << " \n";
    std::cout << "test params - nrow: " << params.n_row << " \n";
    std::cout << "test params - n_col: " << params.n_col << " \n";
    std::cout << "test params - length2: " << params.len2 << " \n";
    std::cout << "test params - nrow2: " << params.n_row2 << " \n";
    std::cout << "test params - n_col2: " << params.n_col2 << " \n";
    std::cout << "test params - seed: " << params.seed << " \n";
    std::cout << "test params - algo: " << params.algo << " \n";

    raft::allocate(data, len);
    raft::allocate(data_back, len);
    raft::allocate(trans_data, len);      //  transformed data

    std::vector<T> data_h = {1.0, 2.0, 5.0, 4.0, 2.0, 1.0};
    data_h.resize(len);
    raft::update_device(data, data_h.data(), len, stream);

    //  could change this to n_row * min(n_row, n_col) once truncation is working
    int len_alphas = params.n_row * params.n_row;  
    raft::allocate(alphas, len_alphas);
    raft::allocate(lambdas, params.n_row);

    //  unscaled eigenvectors - sklearn does not re-scale them until the transform step
    std::vector<T> alphas_ref_h = {-0.6525, -0.0987, 0.7513, -0.4907, 0.8105, -0.3197};
    alphas_ref_h.resize(len_alphas);
    std::vector<T> lambdas_ref_h = {12.6759, 0.6574};
    lambdas_ref_h.resize(params.n_row);
    std::vector<T> trans_data_ref_h = {-2.32318647,-0.35170213, 2.6748886, -0.39794495, 0.65716145,-0.25921649};
    trans_data_ref_h.resize(len);

    raft::allocate(alphas_ref, len_alphas);
    raft::allocate(lambdas_ref, params.n_row);
    raft::allocate(trans_data_ref, len);

    raft::update_device(alphas_ref, alphas_ref_h.data(), len_alphas, stream);
    raft::update_device(lambdas_ref, lambdas_ref_h.data(), params.n_row, stream);
    raft::update_device(trans_data_ref, trans_data_ref_h.data(), len, stream);

    //  standard PCA params work for now
    //  the only irrelevant one is "whiten"
    paramsKPCA prms;
    prms.n_cols = params.n_col;
    prms.n_rows = params.n_row;
    prms.n_components = params.n_col;
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
    CUDA_CHECK(cudaFree(data_back));
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

const std::vector<KPcaInputs<float>> inputsf2 = {
  {0.01f, 3 * 2, 3, 2, 1024 * 128, 1024, 128, 1234ULL, 0},
  {0.01f, 3 * 2, 3, 2, 256 * 32, 256, 32, 1234ULL, 1}};

const std::vector<KPcaInputs<double>> inputsd2 = {
  {0.01, 3 * 2, 3, 2, 1024 * 128, 1024, 128, 1234ULL, 0},
  {0.01, 3 * 2, 3, 2, 256 * 32, 256, 32, 1234ULL, 1}};


typedef KPcaTest<float> KPcaTestValF;
TEST_P(KPcaTestValF, Result) {
  ASSERT_TRUE(devArrMatch(lambdas, lambdas_ref, params.n_col,
                          raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef KPcaTest<double> KPcaTestValD;
TEST_P(KPcaTestValD, Result) {
  ASSERT_TRUE(devArrMatch(lambdas, lambdas_ref, params.n_col,
                          raft::CompareApproxAbs<double>(params.tolerance)));
}

typedef KPcaTest<float> KPcaTestLeftVecF;
TEST_P(KPcaTestLeftVecF, Result) {
  ASSERT_TRUE(devArrMatch(alphas, alphas_ref,
                          (params.n_col * params.n_col),
                          raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef KPcaTest<double> KPcaTestLeftVecD;
TEST_P(KPcaTestLeftVecD, Result) {
  ASSERT_TRUE(devArrMatch(alphas, alphas_ref,
                          (params.n_col * params.n_col),
                          raft::CompareApproxAbs<double>(params.tolerance)));
}

typedef KPcaTest<float> KPcaTestTransDataF;
TEST_P(KPcaTestTransDataF, Result) {
  ASSERT_TRUE(devArrMatch(trans_data, trans_data_ref,
                          (params.n_row * params.n_col),
                          raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef KPcaTest<double> KPcaTestTransDataD;
TEST_P(KPcaTestTransDataD, Result) {
  ASSERT_TRUE(devArrMatch(trans_data, trans_data_ref,
                          (params.n_row * params.n_col),
                          raft::CompareApproxAbs<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(KPcaTests, KPcaTestValF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(KPcaTests, KPcaTestValD, ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_CASE_P(KPcaTests, KPcaTestLeftVecF,
                        ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(KPcaTests, KPcaTestLeftVecD,
                        ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_CASE_P(KPcaTests, KPcaTestTransDataF,
                        ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(KPcaTests, KPcaTestTransDataD,
                        ::testing::ValuesIn(inputsd2));

}  // end namespace ML
