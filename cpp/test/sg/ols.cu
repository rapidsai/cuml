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
#include <gtest/gtest.h>
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <rmm/cuda_stream_pool.hpp>
#include <test_utils.h>
#include <vector>

namespace ML {
namespace GLM {

enum class hconf { SINGLE, LEGACY_ONE, LEGACY_TWO, NON_BLOCKING_ONE, NON_BLOCKING_TWO };

raft::handle_t create_handle(hconf type)
{
  switch (type) {
    case hconf::LEGACY_ONE:
      return raft::handle_t(rmm::cuda_stream_legacy, std::make_shared<rmm::cuda_stream_pool>(1));
    case hconf::LEGACY_TWO:
      return raft::handle_t(rmm::cuda_stream_legacy, std::make_shared<rmm::cuda_stream_pool>(2));
    case hconf::NON_BLOCKING_ONE:
      return raft::handle_t(rmm::cuda_stream_per_thread,
                            std::make_shared<rmm::cuda_stream_pool>(1));
    case hconf::NON_BLOCKING_TWO:
      return raft::handle_t(rmm::cuda_stream_per_thread,
                            std::make_shared<rmm::cuda_stream_pool>(2));
    case hconf::SINGLE:
    default: return raft::handle_t();
  }
}

template <typename T>
struct OlsInputs {
  hconf hc;
  T tol;
  int n_row;
  int n_col;
  int n_row_2;
  int algo;
};

template <typename T>
class OlsTest : public ::testing::TestWithParam<OlsInputs<T>> {
 public:
  OlsTest()
    : params(::testing::TestWithParam<OlsInputs<T>>::GetParam()),
      handle(create_handle(params.hc)),
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
    rmm::device_uvector<T> labels(params.n_row, stream);
    rmm::device_uvector<T> pred_data(len2, stream);

    std::vector<T> data_h = {1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 3.0};
    data_h.resize(len);
    raft::update_device(data.data(), data_h.data(), len, stream);

    std::vector<T> labels_h = {6.0, 8.0, 9.0, 11.0};
    labels_h.resize(params.n_row);
    raft::update_device(labels.data(), labels_h.data(), params.n_row, stream);

    std::vector<T> coef_ref_h = {2.090908, 2.5454557};
    coef_ref_h.resize(params.n_col);
    raft::update_device(coef_ref.data(), coef_ref_h.data(), params.n_col, stream);

    std::vector<T> coef2_ref_h = {1.000001, 1.9999998};
    coef2_ref_h.resize(params.n_col);
    raft::update_device(coef2_ref.data(), coef2_ref_h.data(), params.n_col, stream);

    std::vector<T> coef3_ref_h = {0.99999, 2.00000};
    coef3_ref_h.resize(params.n_col);
    raft::update_device(coef3_ref.data(), coef3_ref_h.data(), params.n_col, stream);

    std::vector<T> pred_data_h = {3.0, 2.0, 5.0, 5.0};
    pred_data_h.resize(len2);
    raft::update_device(pred_data.data(), pred_data_h.data(), len2, stream);

    std::vector<T> pred_ref_h = {19.0, 16.9090};
    pred_ref_h.resize(params.n_row_2);
    raft::update_device(pred_ref.data(), pred_ref_h.data(), params.n_row_2, stream);

    std::vector<T> pred2_ref_h = {16.0, 15.0};
    pred2_ref_h.resize(params.n_row_2);
    raft::update_device(pred2_ref.data(), pred2_ref_h.data(), params.n_row_2, stream);

    std::vector<T> pred3_ref_h = {16.0, 15.0};
    pred3_ref_h.resize(params.n_row_2);
    raft::update_device(pred3_ref.data(), pred3_ref_h.data(), params.n_row_2, stream);

    intercept = T(0);

    olsFit(handle,
           data.data(),
           params.n_row,
           params.n_col,
           labels.data(),
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

    raft::update_device(data.data(), data_h.data(), len, stream);
    raft::update_device(labels.data(), labels_h.data(), params.n_row, stream);

    intercept2 = T(0);
    olsFit(handle,
           data.data(),
           params.n_row,
           params.n_col,
           labels.data(),
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

    raft::update_device(data.data(), data_h.data(), len, stream);
    raft::update_device(labels.data(), labels_h.data(), params.n_row, stream);

    intercept3 = T(0);
    olsFit(handle,
           data.data(),
           params.n_row,
           params.n_col,
           labels.data(),
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

    std::vector<T> data_h = {1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 3.0};
    data_h.resize(len);
    raft::update_device(data_sc.data(), data_h.data(), len, stream);

    std::vector<T> labels_h = {6.0, 8.0, 9.0, 11.0, -1.0, 2.0, -3.6, 3.3};
    labels_h.resize(len);
    raft::update_device(labels_sc.data(), labels_h.data(), len, stream);

    std::vector<T> coef_sc_ref_h = {-0.29285714};
    coef_sc_ref_h.resize(1);
    raft::update_device(coef_sc_ref.data(), coef_sc_ref_h.data(), 1, stream);

    T intercept_sc = T(0);

    olsFit(handle,
           data_sc.data(),
           len,
           1,
           labels_sc.data(),
           coef_sc.data(),
           &intercept_sc,
           true,
           false,
           stream,
           params.algo);
  }

 protected:
  OlsInputs<T> params;

  raft::handle_t handle;
  cudaStream_t stream = 0;

  rmm::device_uvector<T> coef, coef_ref, pred, pred_ref;
  rmm::device_uvector<T> coef2, coef2_ref, pred2, pred2_ref;
  rmm::device_uvector<T> coef3, coef3_ref, pred3, pred3_ref;
  rmm::device_uvector<T> coef_sc, coef_sc_ref;
  T *data, *labels, *data_sc, *labels_sc;
  T intercept, intercept2, intercept3;
};

const std::vector<OlsInputs<float>> inputsf2 = {{hconf::NON_BLOCKING_ONE, 0.001f, 4, 2, 2, 0},
                                                {hconf::NON_BLOCKING_TWO, 0.001f, 4, 2, 2, 1},
                                                {hconf::LEGACY_ONE, 0.001f, 4, 2, 2, 2},
                                                {hconf::LEGACY_TWO, 0.001f, 4, 2, 2, 2},
                                                {hconf::SINGLE, 0.001f, 4, 2, 2, 2}};

const std::vector<OlsInputs<double>> inputsd2 = {{hconf::SINGLE, 0.001, 4, 2, 2, 0},
                                                 {hconf::LEGACY_ONE, 0.001, 4, 2, 2, 1},
                                                 {hconf::LEGACY_TWO, 0.001, 4, 2, 2, 2}};

typedef OlsTest<float> OlsTestF;
TEST_P(OlsTestF, Fit)
{
  ASSERT_TRUE(devArrMatch(
    coef_ref.data(), coef.data(), params.n_col, raft::CompareApproxAbs<float>(params.tol)));

  ASSERT_TRUE(devArrMatch(
    coef2_ref.data(), coef2.data(), params.n_col, raft::CompareApproxAbs<float>(params.tol)));

  ASSERT_TRUE(devArrMatch(
    coef3_ref.data(), coef3.data(), params.n_col, raft::CompareApproxAbs<float>(params.tol)));

  ASSERT_TRUE(devArrMatch(
    pred_ref.data(), pred.data(), params.n_row_2, raft::CompareApproxAbs<float>(params.tol)));

  ASSERT_TRUE(devArrMatch(
    pred2_ref.data(), pred2.data(), params.n_row_2, raft::CompareApproxAbs<float>(params.tol)));

  ASSERT_TRUE(devArrMatch(
    pred3_ref.data(), pred3.data(), params.n_row_2, raft::CompareApproxAbs<float>(params.tol)));

  ASSERT_TRUE(
    devArrMatch(coef_sc_ref.data(), coef_sc.data(), 1, raft::CompareApproxAbs<float>(params.tol)));
}

typedef OlsTest<double> OlsTestD;
TEST_P(OlsTestD, Fit)
{
  ASSERT_TRUE(raft::devArrMatch(
    coef_ref.data(), coef.data(), params.n_col, raft::CompareApproxAbs<double>(params.tol)));

  ASSERT_TRUE(raft::devArrMatch(
    coef2_ref.data(), coef2.data(), params.n_col, raft::CompareApproxAbs<double>(params.tol)));

  ASSERT_TRUE(raft::devArrMatch(
    coef3_ref.data(), coef3.data(), params.n_col, raft::CompareApproxAbs<double>(params.tol)));

  ASSERT_TRUE(raft::devArrMatch(
    pred_ref.data(), pred.data(), params.n_row_2, raft::CompareApproxAbs<double>(params.tol)));

  ASSERT_TRUE(devArrMatch(
    pred2_ref.data(), pred2.data(), params.n_row_2, raft::CompareApproxAbs<double>(params.tol)));

  ASSERT_TRUE(raft::devArrMatch(
    pred3_ref.data(), pred3.data(), params.n_row_2, raft::CompareApproxAbs<double>(params.tol)));

  ASSERT_TRUE(
    devArrMatch(coef_sc_ref.data(), coef_sc.data(), 1, raft::CompareApproxAbs<double>(params.tol)));
}

INSTANTIATE_TEST_CASE_P(OlsTests, OlsTestF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(OlsTests, OlsTestD, ::testing::ValuesIn(inputsd2));

}  // namespace GLM
}  // end namespace ML
