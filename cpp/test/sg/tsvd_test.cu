/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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
#include <test_utils.h>
#include <cuml/decomposition/params.hpp>
#include <raft/random/rng.hpp>
#include <tsvd/tsvd.cuh>
#include <vector>

namespace ML {

using namespace MLCommon;

template <typename T>
struct TsvdInputs {
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
::std::ostream& operator<<(::std::ostream& os, const TsvdInputs<T>& dims)
{
  return os;
}

template <typename T>
class TsvdTest : public ::testing::TestWithParam<TsvdInputs<T>> {
 protected:
  void basicTest()
  {
    params = ::testing::TestWithParam<TsvdInputs<T>>::GetParam();
    raft::random::Rng r(params.seed, raft::random::GenTaps);
    int len = params.len;

    raft::allocate(data, len, stream);

    std::vector<T> data_h = {1.0, 2.0, 4.0, 2.0, 4.0, 5.0, 5.0, 4.0, 2.0, 1.0, 6.0, 4.0};
    data_h.resize(len);
    raft::update_device(data, data_h.data(), len, stream);

    int len_comp = params.n_col * params.n_col;
    raft::allocate(components, len_comp, stream);
    raft::allocate(singular_vals, params.n_col, stream);

    std::vector<T> components_ref_h = {
      -0.3951, 0.1532, 0.9058, -0.7111, -0.6752, -0.1959, -0.5816, 0.7215, -0.3757};
    components_ref_h.resize(len_comp);

    raft::allocate(components_ref, len_comp, stream);
    raft::update_device(components_ref, components_ref_h.data(), len_comp, stream);

    paramsTSVD prms;
    prms.n_cols       = params.n_col;
    prms.n_rows       = params.n_row;
    prms.n_components = params.n_col;
    if (params.algo == 0)
      prms.algorithm = solver::COV_EIG_DQ;
    else
      prms.algorithm = solver::COV_EIG_JACOBI;

    tsvdFit(handle, data, components, singular_vals, prms, stream);
  }

  void advancedTest()
  {
    params = ::testing::TestWithParam<TsvdInputs<T>>::GetParam();
    raft::random::Rng r(params.seed, raft::random::GenTaps);
    int len = params.len2;

    paramsTSVD prms;
    prms.n_cols       = params.n_col2;
    prms.n_rows       = params.n_row2;
    prms.n_components = params.n_col2;
    if (params.algo == 0)
      prms.algorithm = solver::COV_EIG_DQ;
    else if (params.algo == 1)
      prms.algorithm = solver::COV_EIG_JACOBI;
    else
      prms.n_components = params.n_col2 - 15;

    raft::allocate(data2, len, stream);
    r.uniform(data2, len, T(-1.0), T(1.0), stream);
    raft::allocate(data2_trans, prms.n_rows * prms.n_components, stream);

    int len_comp = params.n_col2 * prms.n_components;
    raft::allocate(components2, len_comp, stream);
    raft::allocate(explained_vars2, prms.n_components, stream);
    raft::allocate(explained_var_ratio2, prms.n_components, stream);
    raft::allocate(singular_vals2, prms.n_components, stream);

    tsvdFitTransform(handle,
                     data2,
                     data2_trans,
                     components2,
                     explained_vars2,
                     explained_var_ratio2,
                     singular_vals2,
                     prms,
                     stream);

    raft::allocate(data2_back, len, stream);
    tsvdInverseTransform(handle, data2_trans, components2, data2_back, prms, stream);
  }

  void SetUp() override
  {
    CUDA_CHECK(cudaStreamCreate(&stream));
    handle.set_stream(stream);
    basicTest();
    advancedTest();
  }

  void TearDown() override
  {
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(components));
    CUDA_CHECK(cudaFree(singular_vals));
    CUDA_CHECK(cudaFree(components_ref));
    CUDA_CHECK(cudaFree(data2));
    CUDA_CHECK(cudaFree(data2_trans));
    CUDA_CHECK(cudaFree(data2_back));
    CUDA_CHECK(cudaFree(components2));
    CUDA_CHECK(cudaFree(explained_vars2));
    CUDA_CHECK(cudaFree(explained_var_ratio2));
    CUDA_CHECK(cudaFree(singular_vals2));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

 protected:
  TsvdInputs<T> params;
  T *data, *components, *singular_vals, *components_ref, *explained_vars_ref;
  T *data2, *data2_trans, *data2_back, *components2, *explained_vars2, *explained_var_ratio2,
    *singular_vals2;
  raft::handle_t handle;
  cudaStream_t stream = 0;
};

const std::vector<TsvdInputs<float>> inputsf2 = {
  {0.01f, 4 * 3, 4, 3, 1024 * 128, 1024, 128, 1234ULL, 0},
  {0.01f, 4 * 3, 4, 3, 1024 * 128, 1024, 128, 1234ULL, 1},
  {0.05f, 4 * 3, 4, 3, 512 * 64, 512, 64, 1234ULL, 2},
  {0.05f, 4 * 3, 4, 3, 512 * 64, 512, 64, 1234ULL, 2}};

const std::vector<TsvdInputs<double>> inputsd2 = {
  {0.01, 4 * 3, 4, 3, 1024 * 128, 1024, 128, 1234ULL, 0},
  {0.01, 4 * 3, 4, 3, 1024 * 128, 1024, 128, 1234ULL, 1},
  {0.05, 4 * 3, 4, 3, 512 * 64, 512, 64, 1234ULL, 2},
  {0.05, 4 * 3, 4, 3, 512 * 64, 512, 64, 1234ULL, 2}};

typedef TsvdTest<float> TsvdTestLeftVecF;
TEST_P(TsvdTestLeftVecF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(components,
                                components_ref,
                                (params.n_col * params.n_col),
                                raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef TsvdTest<double> TsvdTestLeftVecD;
TEST_P(TsvdTestLeftVecD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(components,
                                components_ref,
                                (params.n_col * params.n_col),
                                raft::CompareApproxAbs<double>(params.tolerance)));
}

typedef TsvdTest<float> TsvdTestDataVecF;
TEST_P(TsvdTestDataVecF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(data2,
                                data2_back,
                                (params.n_col2 * params.n_col2),
                                raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef TsvdTest<double> TsvdTestDataVecD;
TEST_P(TsvdTestDataVecD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(data2,
                                data2_back,
                                (params.n_col2 * params.n_col2),
                                raft::CompareApproxAbs<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(TsvdTests, TsvdTestLeftVecF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(TsvdTests, TsvdTestLeftVecD, ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_CASE_P(TsvdTests, TsvdTestDataVecF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(TsvdTests, TsvdTestDataVecD, ::testing::ValuesIn(inputsd2));

}  // end namespace ML
