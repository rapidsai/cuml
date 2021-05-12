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
    raft::allocate(trans_data_ref, len);  //  ground truth transformed data

    std::vector<T> data_h = {1.0, 2.0, 5.0, 4.0, 2.0, 1.0};
    data_h.resize(len);

    raft::update_device(data, data_h.data(), len, stream);

    std::vector<T> trans_data_ref_h = {-2.3231, -0.3517, 2.6748,
                                       -0.3979, 0.6571,  -0.2592};
    trans_data_ref_h.resize(len);
    raft::update_device(trans_data_ref, trans_data_ref_h.data(), len, stream);

    int len_comp = params.n_row * params.n_row;
    raft::allocate(components, len_comp);
    raft::allocate(explained_vars, params.n_row);
    raft::allocate(explained_var_ratio, params.n_row);
    raft::allocate(singular_vals, params.n_row);
    raft::allocate(mean, params.n_row);
    raft::allocate(noise_vars, 1);

    std::vector<T> components_ref_h = {-0.6525, -0.0987, 0.7513, -0.4907, 0.8105, -0.3197};


    components_ref_h.resize(len_comp);
    std::vector<T> explained_vars_ref_h = {12.6759, 0.6574};
    explained_vars_ref_h.resize(params.n_row);

    raft::allocate(components_ref, len_comp);
    raft::allocate(explained_vars_ref, params.n_row);

    raft::update_device(components_ref, components_ref_h.data(), len_comp,
                        stream);
    raft::update_device(explained_vars_ref, explained_vars_ref_h.data(),
                        params.n_row, stream);

    paramsPCA prms;
    prms.n_cols = params.n_col;
    prms.n_rows = params.n_row;
    prms.n_components = params.n_col;
    prms.whiten = false;
    if (params.algo == 0)
      prms.algorithm = solver::COV_EIG_DQ;
    else
      prms.algorithm = solver::COV_EIG_JACOBI;

    std::cout << "basicTest - 5 \n";
    kpcaFit(handle, data, components, explained_vars, explained_var_ratio,
            singular_vals, mean, noise_vars, prms, stream);
    std::cout << "basicTest - 6 \n";
    kpcaTransform(handle, data, components, trans_data, explained_vars, mean,
                  prms, stream);
  }

  void SetUp() override {
    CUDA_CHECK(cudaStreamCreate(&stream));
    handle.set_stream(stream);
    basicTest();
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(components));
    CUDA_CHECK(cudaFree(trans_data));
    CUDA_CHECK(cudaFree(data_back));
    CUDA_CHECK(cudaFree(trans_data_ref));
    CUDA_CHECK(cudaFree(explained_vars));
    CUDA_CHECK(cudaFree(explained_var_ratio));
    CUDA_CHECK(cudaFree(singular_vals));
    CUDA_CHECK(cudaFree(mean));
    CUDA_CHECK(cudaFree(noise_vars));
    CUDA_CHECK(cudaFree(components_ref));
    CUDA_CHECK(cudaFree(explained_vars_ref));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

 protected:
  KPcaInputs<T> params;
  T *data, *trans_data, *data_back, *components, *explained_vars,
    *explained_var_ratio, *singular_vals, *mean, *noise_vars, *trans_data_ref,
    *components_ref, *explained_vars_ref;

  raft::handle_t handle;
  cudaStream_t stream;
};

const std::vector<KPcaInputs<float>> inputsf2 = {
  {0.01f, 3 * 2, 3, 2, 1024 * 128, 1024, 128, 1234ULL, 0},
  {0.01f, 3 * 2, 3, 2, 256 * 32, 256, 32, 1234ULL, 1}};

typedef KPcaTest<float> KPcaTestValF;
TEST_P(KPcaTestValF, Result) {
  ASSERT_TRUE(devArrMatch(explained_vars, explained_vars_ref, params.n_col,
                          raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef KPcaTest<float> KPcaTestLeftVecF;
TEST_P(KPcaTestLeftVecF, Result) {
  ASSERT_TRUE(devArrMatch(components, components_ref,
                          (params.n_col * params.n_col),
                          raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef KPcaTest<float> KPcaTestTransDataF;
TEST_P(KPcaTestTransDataF, Result) {
  ASSERT_TRUE(devArrMatch(trans_data, trans_data_ref,
                          (params.n_row * params.n_col),
                          raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef KPcaTest<float> KPcaTestDataVecSmallF;
TEST_P(KPcaTestDataVecSmallF, Result) {
  ASSERT_TRUE(devArrMatch(data, data_back, (params.n_col * params.n_col),
                          raft::CompareApproxAbs<float>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(KPcaTests, KPcaTestValF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(KPcaTests, KPcaTestLeftVecF,
                        ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(KPcaTests, KPcaTestDataVecSmallF,
                        ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(KPcaTests, KPcaTestTransDataF,
                        ::testing::ValuesIn(inputsf2));

}  // end namespace ML
