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

#include <glm/ols.cuh>
#include <glm/ridge.cuh>
#include <gtest/gtest.h>
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <test_utils.h>

namespace ML {
namespace GLM {

template <typename T>
struct RidgeInputs {
  T tol;
  int n_row;
  int n_col;
  int n_row_2;
  int algo;
  T alpha;
};

template <typename T>
class RidgeTest : public ::testing::TestWithParam<RidgeInputs<T>> {
 public:
  RidgeTest()
    : params(::testing::TestWithParam<RidgeInputs<T>>::GetParam()),
      stream(handle.get_stream()),
      coef(params.n_col, stream),
      coef2(params.n_col, stream),
      coef3(params.n_col, stream),
      coef_ref(params.n_col, stream),
      coef2_ref(params.n_col, stream),
      coef3_ref(params.n_col, stream),
      pred(params.n_row_2, stream),
      pred_ref(params.n_row_2, stream),
      pred2(params.n_row_2, stream),
      pred2_ref(params.n_row_2, stream),
      pred3(params.n_row_2, stream),
      pred3_ref(params.n_row_2, stream),
      coef_sc(1, stream),
      coef_sc_ref(1, stream)
  {
    basicTest();
    basicTest2();
  }

 protected:
  void basicTest()
  {
    int len  = params.n_row * params.n_col;
    int len2 = params.n_row_2 * params.n_col;

    rmm::device_uvector<T> data(len, stream);
    rmm::device_uvector<T> pred_data(len2, stream);
    rmm::device_uvector<T> labels(params.n_row, stream);
    T alpha = params.alpha;

    /* How to reproduce the coefficients for this test:

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    scaler = StandardScaler(with_mean=True, with_std=True)
    x_norm = scaler.fit_transform(x_train)

    m = Ridge(
        fit_intercept=False, normalize=False, alpha=0.5)
    m.fit(x_train, y)
    print(m.coef_, m.predict(x_test))

    m = Ridge(
        fit_intercept=True, normalize=False, alpha=0.5)
    m.fit(x_train, y)
    print(m.coef_, m.predict(x_test))

    m = Ridge(
        fit_intercept=True, normalize=False, alpha=0.5)
    m.fit(x_norm, y)
    print(m.coef_ / scaler.scale_, m.predict(scaler.transform(x_test)))

     */

    T data_h[len] = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    raft::update_device(data.data(), data_h, len, stream);

    T labels_h[params.n_row] = {0.0, 0.1, 1.0};
    raft::update_device(labels.data(), labels_h, params.n_row, stream);

    T coef_ref_h[params.n_col] = {0.4, 0.4};
    raft::update_device(coef_ref.data(), coef_ref_h, params.n_col, stream);

    T coef2_ref_h[params.n_col] = {0.3454546, 0.34545454};
    raft::update_device(coef2_ref.data(), coef2_ref_h, params.n_col, stream);

    T coef3_ref_h[params.n_col] = {0.43846154, 0.43846154};
    raft::update_device(coef3_ref.data(), coef3_ref_h, params.n_col, stream);

    T pred_data_h[len2] = {0.5, 2.0, 0.2, 1.0};
    raft::update_device(pred_data.data(), pred_data_h, len2, stream);

    T pred_ref_h[params.n_row_2] = {0.28, 1.2};
    raft::update_device(pred_ref.data(), pred_ref_h, params.n_row_2, stream);

    T pred2_ref_h[params.n_row_2] = {0.37818182, 1.17272727};
    raft::update_device(pred2_ref.data(), pred2_ref_h, params.n_row_2, stream);

    T pred3_ref_h[params.n_row_2] = {0.38128205, 1.38974359};
    raft::update_device(pred3_ref.data(), pred3_ref_h, params.n_row_2, stream);

    intercept = T(0);

    ridgeFit(handle,
             data.data(),
             params.n_row,
             params.n_col,
             labels.data(),
             &alpha,
             1,
             coef.data(),
             &intercept,
             false,
             false,
             stream,
             params.algo);

    gemmPredict(handle,
                pred_data.data(),
                params.n_row_2,
                params.n_col,
                coef.data(),
                intercept,
                pred.data(),
                stream);

    raft::update_device(data.data(), data_h, len, stream);
    raft::update_device(labels.data(), labels_h, params.n_row, stream);

    intercept2 = T(0);
    ridgeFit(handle,
             data.data(),
             params.n_row,
             params.n_col,
             labels.data(),
             &alpha,
             1,
             coef2.data(),
             &intercept2,
             true,
             false,
             stream,
             params.algo);

    gemmPredict(handle,
                pred_data.data(),
                params.n_row_2,
                params.n_col,
                coef2.data(),
                intercept2,
                pred2.data(),
                stream);

    raft::update_device(data.data(), data_h, len, stream);
    raft::update_device(labels.data(), labels_h, params.n_row, stream);

    intercept3 = T(0);
    ridgeFit(handle,
             data.data(),
             params.n_row,
             params.n_col,
             labels.data(),
             &alpha,
             1,
             coef3.data(),
             &intercept3,
             true,
             true,
             stream,
             params.algo);

    gemmPredict(handle,
                pred_data.data(),
                params.n_row_2,
                params.n_col,
                coef3.data(),
                intercept3,
                pred3.data(),
                stream);
  }

  void basicTest2()
  {
    int len = params.n_row * params.n_col;

    rmm::device_uvector<T> data_sc(len, stream);
    rmm::device_uvector<T> labels_sc(len, stream);

    std::vector<T> data_h = {1.0, 1.0, 2.0, 2.0, 1.0, 2.0};
    data_h.resize(len);
    raft::update_device(data_sc.data(), data_h.data(), len, stream);

    std::vector<T> labels_h = {6.0, 8.0, 9.0, 11.0, -1.0, 2.0};
    labels_h.resize(len);
    raft::update_device(labels_sc.data(), labels_h.data(), len, stream);

    std::vector<T> coef_sc_ref_h = {1.8};
    coef_sc_ref_h.resize(1);
    raft::update_device(coef_sc_ref.data(), coef_sc_ref_h.data(), 1, stream);

    T intercept_sc = T(0);
    T alpha_sc     = T(1.0);

    ridgeFit(handle,
             data_sc.data(),
             len,
             1,
             labels_sc.data(),
             &alpha_sc,
             1,
             coef_sc.data(),
             &intercept_sc,
             true,
             false,
             stream,
             params.algo);
  }

 protected:
  raft::handle_t handle;
  cudaStream_t stream = 0;

  RidgeInputs<T> params;
  rmm::device_uvector<T> coef, coef_ref, pred, pred_ref;
  rmm::device_uvector<T> coef2, coef2_ref, pred2, pred2_ref;
  rmm::device_uvector<T> coef3, coef3_ref, pred3, pred3_ref;
  rmm::device_uvector<T> coef_sc, coef_sc_ref;
  T intercept, intercept2, intercept3;
};

const std::vector<RidgeInputs<float>> inputsf2 = {{0.001f, 3, 2, 2, 0, 0.5f},
                                                  {0.001f, 3, 2, 2, 1, 0.5f}};

const std::vector<RidgeInputs<double>> inputsd2 = {{0.001, 3, 2, 2, 0, 0.5},
                                                   {0.001, 3, 2, 2, 1, 0.5}};

typedef RidgeTest<float> RidgeTestF;
TEST_P(RidgeTestF, Fit)
{
  ASSERT_TRUE(raft::devArrMatch(
    coef_ref.data(), coef.data(), params.n_col, raft::CompareApproxAbs<float>(params.tol)));

  ASSERT_TRUE(raft::devArrMatch(
    coef2_ref.data(), coef2.data(), params.n_col, raft::CompareApproxAbs<float>(params.tol)));

  ASSERT_TRUE(raft::devArrMatch(
    coef3_ref.data(), coef3.data(), params.n_col, raft::CompareApproxAbs<float>(params.tol)));

  ASSERT_TRUE(raft::devArrMatch(
    pred_ref.data(), pred.data(), params.n_row_2, raft::CompareApproxAbs<float>(params.tol)));

  ASSERT_TRUE(raft::devArrMatch(
    pred2_ref.data(), pred2.data(), params.n_row_2, raft::CompareApproxAbs<float>(params.tol)));

  ASSERT_TRUE(raft::devArrMatch(
    pred3_ref.data(), pred3.data(), params.n_row_2, raft::CompareApproxAbs<float>(params.tol)));

  ASSERT_TRUE(raft::devArrMatch(
    coef_sc_ref.data(), coef_sc.data(), 1, raft::CompareApproxAbs<float>(params.tol)));
}

typedef RidgeTest<double> RidgeTestD;
TEST_P(RidgeTestD, Fit)
{
  ASSERT_TRUE(raft::devArrMatch(
    coef_ref.data(), coef.data(), params.n_col, raft::CompareApproxAbs<double>(params.tol)));

  ASSERT_TRUE(raft::devArrMatch(
    coef2_ref.data(), coef2.data(), params.n_col, raft::CompareApproxAbs<double>(params.tol)));

  ASSERT_TRUE(raft::devArrMatch(
    coef3_ref.data(), coef3.data(), params.n_col, raft::CompareApproxAbs<double>(params.tol)));

  ASSERT_TRUE(raft::devArrMatch(
    pred_ref.data(), pred.data(), params.n_row_2, raft::CompareApproxAbs<double>(params.tol)));

  ASSERT_TRUE(raft::devArrMatch(
    pred2_ref.data(), pred2.data(), params.n_row_2, raft::CompareApproxAbs<double>(params.tol)));

  ASSERT_TRUE(raft::devArrMatch(
    pred3_ref.data(), pred3.data(), params.n_row_2, raft::CompareApproxAbs<double>(params.tol)));

  ASSERT_TRUE(raft::devArrMatch(
    coef_sc_ref.data(), coef_sc.data(), 1, raft::CompareApproxAbs<double>(params.tol)));
}

INSTANTIATE_TEST_CASE_P(RidgeTests, RidgeTestF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(RidgeTests, RidgeTestD, ::testing::ValuesIn(inputsd2));

}  // namespace GLM
}  // end namespace ML
