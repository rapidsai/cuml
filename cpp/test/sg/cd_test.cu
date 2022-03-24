/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include <raft/matrix/matrix.hpp>
#include <rmm/device_uvector.hpp>
#include <solver/cd.cuh>
#include <test_utils.h>

#include <raft/stats/mean.hpp>
#include <raft/stats/meanvar.hpp>
#include <raft/stats/stddev.hpp>

namespace ML {
namespace Solver {

using namespace MLCommon;

template <typename T>
struct CdInputs {
  T tol;
  int n_row;
  int n_col;
};

template <typename T>
class CdTest : public ::testing::TestWithParam<CdInputs<T>> {
 public:
  CdTest()
    : params(::testing::TestWithParam<CdInputs<T>>::GetParam()),
      stream(handle.get_stream()),
      data(params.n_row * params.n_col, stream),
      labels(params.n_row, stream),
      coef(params.n_col, stream),
      coef2(params.n_col, stream),
      coef3(params.n_col, stream),
      coef4(params.n_col, stream),
      coef_ref(params.n_col, stream),
      coef2_ref(params.n_col, stream),
      coef3_ref(params.n_col, stream),
      coef4_ref(params.n_col, stream)
  {
    RAFT_CUDA_TRY(cudaMemsetAsync(coef.data(), 0, coef.size() * sizeof(T), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(coef2.data(), 0, coef2.size() * sizeof(T), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(coef3.data(), 0, coef3.size() * sizeof(T), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(coef4.data(), 0, coef4.size() * sizeof(T), stream));

    RAFT_CUDA_TRY(cudaMemsetAsync(coef_ref.data(), 0, coef_ref.size() * sizeof(T), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(coef2_ref.data(), 0, coef2_ref.size() * sizeof(T), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(coef3_ref.data(), 0, coef3_ref.size() * sizeof(T), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(coef4_ref.data(), 0, coef4_ref.size() * sizeof(T), stream));
  }

 protected:
  void lasso()
  {
    int len = params.n_row * params.n_col;

    T data_h[len] = {1.0, 1.2, 2.0, 2.0, 4.5, 2.0, 2.0, 3.0};
    raft::update_device(data.data(), data_h, len, stream);

    T labels_h[params.n_row] = {6.0, 8.3, 9.8, 11.2};
    raft::update_device(labels.data(), labels_h, params.n_row, stream);

    /* How to reproduce the coefficients for this test:

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler(with_mean=True, with_std=True)
    x_norm = scaler.fit_transform(data_h)
    m = ElasticNet(fit_intercept=, normalize=, alpha=, l1_ratio=)
    m.fit(x_norm, y)
    print(m.coef_ / scaler.scale_ if normalize else m.coef_)
     */

    T coef_ref_h[params.n_col] = {4.90832, 0.35031};
    raft::update_device(coef_ref.data(), coef_ref_h, params.n_col, stream);

    T coef2_ref_h[params.n_col] = {2.53530, -0.36832};
    raft::update_device(coef2_ref.data(), coef2_ref_h, params.n_col, stream);

    T coef3_ref_h[params.n_col] = {2.932841, 1.15248};
    raft::update_device(coef3_ref.data(), coef3_ref_h, params.n_col, stream);

    T coef4_ref_h[params.n_col] = {1.75420431, -0.16215289};
    raft::update_device(coef4_ref.data(), coef4_ref_h, params.n_col, stream);

    bool fit_intercept  = false;
    bool normalize      = false;
    int epochs          = 200;
    T alpha             = T(0.2);
    T l1_ratio          = T(1.0);
    bool shuffle        = false;
    T tol               = T(1e-4);
    ML::loss_funct loss = ML::loss_funct::SQRD_LOSS;

    intercept = T(0);
    cdFit(handle,
          data.data(),
          params.n_row,
          params.n_col,
          labels.data(),
          coef.data(),
          &intercept,
          fit_intercept,
          normalize,
          epochs,
          loss,
          alpha,
          l1_ratio,
          shuffle,
          tol,
          stream);

    fit_intercept = true;
    intercept2    = T(0);
    cdFit(handle,
          data.data(),
          params.n_row,
          params.n_col,
          labels.data(),
          coef2.data(),
          &intercept2,
          fit_intercept,
          normalize,
          epochs,
          loss,
          alpha,
          l1_ratio,
          shuffle,
          tol,
          stream);

    alpha         = T(1.0);
    l1_ratio      = T(0.5);
    fit_intercept = false;
    intercept     = T(0);
    cdFit(handle,
          data.data(),
          params.n_row,
          params.n_col,
          labels.data(),
          coef3.data(),
          &intercept,
          fit_intercept,
          normalize,
          epochs,
          loss,
          alpha,
          l1_ratio,
          shuffle,
          tol,
          stream);

    fit_intercept = true;
    normalize     = true;
    intercept2    = T(0);
    cdFit(handle,
          data.data(),
          params.n_row,
          params.n_col,
          labels.data(),
          coef4.data(),
          &intercept2,
          fit_intercept,
          normalize,
          epochs,
          loss,
          alpha,
          l1_ratio,
          shuffle,
          tol,
          stream);
  }

  void SetUp() override { lasso(); }

 protected:
  CdInputs<T> params;
  raft::handle_t handle;
  cudaStream_t stream = 0;

  rmm::device_uvector<T> data, labels, coef, coef_ref;
  rmm::device_uvector<T> coef2, coef2_ref;
  rmm::device_uvector<T> coef3, coef3_ref;
  rmm::device_uvector<T> coef4, coef4_ref;
  T intercept, intercept2;
};

const std::vector<CdInputs<float>> inputsf2 = {{0.01f, 4, 2}};

const std::vector<CdInputs<double>> inputsd2 = {{0.01, 4, 2}};

typedef CdTest<float> CdTestF;
TEST_P(CdTestF, Fit)
{
  ASSERT_TRUE(raft::devArrMatch(
    coef_ref.data(), coef.data(), params.n_col, raft::CompareApproxAbs<float>(params.tol)));

  ASSERT_TRUE(raft::devArrMatch(
    coef2_ref.data(), coef2.data(), params.n_col, raft::CompareApproxAbs<float>(params.tol)));

  ASSERT_TRUE(raft::devArrMatch(
    coef3_ref.data(), coef3.data(), params.n_col, raft::CompareApproxAbs<float>(params.tol)));

  rmm::device_uvector<float> means_1(params.n_col, stream);
  rmm::device_uvector<float> means_2(params.n_col, stream);
  rmm::device_uvector<float> vars_1(params.n_col, stream);
  rmm::device_uvector<float> vars_2(params.n_col, stream);

  raft::stats::mean(means_1.data(), data.data(), params.n_col, params.n_row, false, false, stream);
  raft::stats::vars(
    vars_1.data(), data.data(), means_1.data(), params.n_col, params.n_row, false, false, stream);
  raft::stats::meanvar(
    means_2.data(), vars_2.data(), data.data(), params.n_col, params.n_row, false, false, stream);

  ASSERT_TRUE(raft::devArrMatch(
    means_1.data(), means_2.data(), params.n_col, raft::CompareApprox<float>(0.0001)));
  ASSERT_TRUE(raft::devArrMatch(
    vars_1.data(), vars_2.data(), params.n_col, raft::CompareApprox<float>(0.0001)));

  ASSERT_TRUE(raft::devArrMatch(
    coef4_ref.data(), coef4.data(), params.n_col, raft::CompareApproxAbs<float>(params.tol)));

  ASSERT_TRUE(raft::devArrMatch(
    coef4_ref.data(), coef4.data(), params.n_col, raft::CompareApproxAbs<float>(params.tol)));
}

typedef CdTest<double> CdTestD;
TEST_P(CdTestD, Fit)
{
  ASSERT_TRUE(raft::devArrMatch(
    coef_ref.data(), coef.data(), params.n_col, raft::CompareApproxAbs<double>(params.tol)));

  ASSERT_TRUE(raft::devArrMatch(
    coef2_ref.data(), coef2.data(), params.n_col, raft::CompareApproxAbs<double>(params.tol)));

  ASSERT_TRUE(raft::devArrMatch(
    coef3_ref.data(), coef3.data(), params.n_col, raft::CompareApproxAbs<double>(params.tol)));

  rmm::device_uvector<double> means_1(params.n_col, stream);
  rmm::device_uvector<double> means_2(params.n_col, stream);
  rmm::device_uvector<double> vars_1(params.n_col, stream);
  rmm::device_uvector<double> vars_2(params.n_col, stream);

  raft::stats::mean(means_1.data(), data.data(), params.n_col, params.n_row, false, false, stream);
  raft::stats::vars(
    vars_1.data(), data.data(), means_1.data(), params.n_col, params.n_row, false, false, stream);
  raft::stats::meanvar(
    means_2.data(), vars_2.data(), data.data(), params.n_col, params.n_row, false, false, stream);

  ASSERT_TRUE(raft::devArrMatch(
    means_1.data(), means_2.data(), params.n_col, raft::CompareApprox<double>(0.0001)));
  ASSERT_TRUE(raft::devArrMatch(
    vars_1.data(), vars_2.data(), params.n_col, raft::CompareApprox<double>(0.0001)));

  ASSERT_TRUE(raft::devArrMatch(
    coef4_ref.data(), coef4.data(), params.n_col, raft::CompareApproxAbs<double>(params.tol)));
}

INSTANTIATE_TEST_CASE_P(CdTests, CdTestF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(CdTests, CdTestD, ::testing::ValuesIn(inputsd2));

}  // namespace Solver
}  // end namespace ML
