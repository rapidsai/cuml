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
#include <pca/pca.cuh>
#include <raft/cuda_utils.cuh>
#include <raft/random/rng.cuh>
#include <vector>

namespace ML {

using namespace MLCommon;

template <typename T>
struct PcaInputs {
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
::std::ostream& operator<<(::std::ostream& os, const PcaInputs<T>& dims) {
  return os;
}

template <typename T>
class PcaTest : public ::testing::TestWithParam<PcaInputs<T>> {
 protected:
  void basicTest() {
    params = ::testing::TestWithParam<PcaInputs<T>>::GetParam();
    raft::random::Rng r(params.seed, raft::random::GenTaps);
    int len = params.len;

    raft::allocate(data, len);
    raft::allocate(data_back, len);
    raft::allocate(trans_data, len);
    raft::allocate(trans_data_ref, len);

    std::vector<T> data_h = {1.0, 2.0, 5.0, 4.0, 2.0, 1.0};
    data_h.resize(len);
    raft::update_device(data, data_h.data(), len, stream);

    std::vector<T> trans_data_ref_h = {-2.3231, -0.3517, 2.6748,
                                       -0.3979, 0.6571,  -0.2592};
    trans_data_ref_h.resize(len);
    raft::update_device(trans_data_ref, trans_data_ref_h.data(), len, stream);

    int len_comp = params.n_col * params.n_col;
    raft::allocate(components, len_comp);
    raft::allocate(explained_vars, params.n_col);
    raft::allocate(explained_var_ratio, params.n_col);
    raft::allocate(singular_vals, params.n_col);
    raft::allocate(mean, params.n_col);
    raft::allocate(noise_vars, 1);

    std::vector<T> components_ref_h = {0.8163, 0.5776, -0.5776, 0.8163};
    components_ref_h.resize(len_comp);
    std::vector<T> explained_vars_ref_h = {6.338, 0.3287};
    explained_vars_ref_h.resize(params.n_col);

    raft::allocate(components_ref, len_comp);
    raft::allocate(explained_vars_ref, params.n_col);

    raft::update_device(components_ref, components_ref_h.data(), len_comp,
                        stream);
    raft::update_device(explained_vars_ref, explained_vars_ref_h.data(),
                        params.n_col, stream);

    paramsPCA prms;
    prms.n_cols = params.n_col;
    prms.n_rows = params.n_row;
    prms.n_components = params.n_col;
    prms.whiten = false;
    if (params.algo == 0)
      prms.algorithm = solver::COV_EIG_DQ;
    else
      prms.algorithm = solver::COV_EIG_JACOBI;

    pcaFit(handle, data, components, explained_vars, explained_var_ratio,
           singular_vals, mean, noise_vars, prms, stream);
    pcaTransform(handle, data, components, trans_data, singular_vals, mean,
                 prms, stream);
    pcaInverseTransform(handle, trans_data, components, singular_vals, mean,
                        data_back, prms, stream);
  }

  void advancedTest() {
    params = ::testing::TestWithParam<PcaInputs<T>>::GetParam();
    raft::random::Rng r(params.seed, raft::random::GenTaps);
    int len = params.len2;

    paramsPCA prms;
    prms.n_cols = params.n_col2;
    prms.n_rows = params.n_row2;
    prms.n_components = params.n_col2;
    prms.whiten = false;
    if (params.algo == 0)
      prms.algorithm = solver::COV_EIG_DQ;
    else if (params.algo == 1)
      prms.algorithm = solver::COV_EIG_JACOBI;

    raft::allocate(data2, len);
    r.uniform(data2, len, T(-1.0), T(1.0), stream);
    raft::allocate(data2_trans, prms.n_rows * prms.n_components);

    int len_comp = params.n_col2 * prms.n_components;
    raft::allocate(components2, len_comp);
    raft::allocate(explained_vars2, prms.n_components);
    raft::allocate(explained_var_ratio2, prms.n_components);
    raft::allocate(singular_vals2, prms.n_components);
    raft::allocate(mean2, prms.n_cols);
    raft::allocate(noise_vars2, 1);

    pcaFitTransform(handle, data2, data2_trans, components2, explained_vars2,
                    explained_var_ratio2, singular_vals2, mean2, noise_vars2,
                    prms, stream);

    raft::allocate(data2_back, len);
    pcaInverseTransform(handle, data2_trans, components2, singular_vals2, mean2,
                        data2_back, prms, stream);
  }

  void SetUp() override {
    CUDA_CHECK(cudaStreamCreate(&stream));
    handle.set_stream(stream);
    basicTest();
    advancedTest();
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
    CUDA_CHECK(cudaFree(data2));
    CUDA_CHECK(cudaFree(data2_trans));
    CUDA_CHECK(cudaFree(data2_back));
    CUDA_CHECK(cudaFree(components2));
    CUDA_CHECK(cudaFree(explained_vars2));
    CUDA_CHECK(cudaFree(explained_var_ratio2));
    CUDA_CHECK(cudaFree(singular_vals2));
    CUDA_CHECK(cudaFree(mean2));
    CUDA_CHECK(cudaFree(noise_vars2));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

 protected:
  PcaInputs<T> params;
  T *data, *trans_data, *data_back, *components, *explained_vars,
    *explained_var_ratio, *singular_vals, *mean, *noise_vars, *trans_data_ref,
    *components_ref, *explained_vars_ref;

  T *data2, *data2_trans, *data2_back, *components2, *explained_vars2,
    *explained_var_ratio2, *singular_vals2, *mean2, *noise_vars2;
  raft::handle_t handle;
  cudaStream_t stream;
};

const std::vector<PcaInputs<float>> inputsf2 = {
  {0.01f, 3 * 2, 3, 2, 1024 * 128, 1024, 128, 1234ULL, 0},
  {0.01f, 3 * 2, 3, 2, 256 * 32, 256, 32, 1234ULL, 1}};

const std::vector<PcaInputs<double>> inputsd2 = {
  {0.01, 3 * 2, 3, 2, 1024 * 128, 1024, 128, 1234ULL, 0},
  {0.01, 3 * 2, 3, 2, 256 * 32, 256, 32, 1234ULL, 1}};

typedef PcaTest<float> PcaTestValF;
TEST_P(PcaTestValF, Result) {
  ASSERT_TRUE(devArrMatch(explained_vars, explained_vars_ref, params.n_col,
                          raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef PcaTest<double> PcaTestValD;
TEST_P(PcaTestValD, Result) {
  ASSERT_TRUE(devArrMatch(explained_vars, explained_vars_ref, params.n_col,
                          raft::CompareApproxAbs<double>(params.tolerance)));
}

typedef PcaTest<float> PcaTestLeftVecF;
TEST_P(PcaTestLeftVecF, Result) {
  ASSERT_TRUE(devArrMatch(components, components_ref,
                          (params.n_col * params.n_col),
                          raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef PcaTest<double> PcaTestLeftVecD;
TEST_P(PcaTestLeftVecD, Result) {
  ASSERT_TRUE(devArrMatch(components, components_ref,
                          (params.n_col * params.n_col),
                          raft::CompareApproxAbs<double>(params.tolerance)));
}

typedef PcaTest<float> PcaTestTransDataF;
TEST_P(PcaTestTransDataF, Result) {
  ASSERT_TRUE(devArrMatch(trans_data, trans_data_ref,
                          (params.n_row * params.n_col),
                          raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef PcaTest<double> PcaTestTransDataD;
TEST_P(PcaTestTransDataD, Result) {
  ASSERT_TRUE(devArrMatch(trans_data, trans_data_ref,
                          (params.n_row * params.n_col),
                          raft::CompareApproxAbs<double>(params.tolerance)));
}

typedef PcaTest<float> PcaTestDataVecSmallF;
TEST_P(PcaTestDataVecSmallF, Result) {
  ASSERT_TRUE(devArrMatch(data, data_back, (params.n_col * params.n_col),
                          raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef PcaTest<double> PcaTestDataVecSmallD;
TEST_P(PcaTestDataVecSmallD, Result) {
  ASSERT_TRUE(devArrMatch(data, data_back, (params.n_col * params.n_col),
                          raft::CompareApproxAbs<double>(params.tolerance)));
}

// FIXME: These tests are disabled due to driver 418+ making them fail:
// https://github.com/rapidsai/cuml/issues/379
typedef PcaTest<float> PcaTestDataVecF;
TEST_P(PcaTestDataVecF, Result) {
  ASSERT_TRUE(devArrMatch(data2, data2_back, (params.n_col2 * params.n_col2),
                          raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef PcaTest<double> PcaTestDataVecD;
TEST_P(PcaTestDataVecD, Result) {
  ASSERT_TRUE(
    raft::devArrMatch(data2, data2_back, (params.n_col2 * params.n_col2),
                      raft::CompareApproxAbs<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestValF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestValD, ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestLeftVecF,
                        ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestLeftVecD,
                        ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestDataVecSmallF,
                        ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestDataVecSmallD,
                        ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestTransDataF,
                        ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestTransDataD,
                        ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestDataVecF,
                        ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestDataVecD,
                        ::testing::ValuesIn(inputsd2));

}  // end namespace ML
