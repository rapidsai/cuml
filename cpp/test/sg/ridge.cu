/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cuda_utils.h>
#include <gtest/gtest.h>
#include <test_utils.h>
#include "glm/ridge.h"
#include "ml_utils.h"

namespace ML {
namespace GLM {

using namespace MLCommon;

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
 protected:
  void basicTest() {
    params = ::testing::TestWithParam<RidgeInputs<T>>::GetParam();
    int len = params.n_row * params.n_col;
    int len2 = params.n_row_2 * params.n_col;

    allocate(data, len);
    allocate(labels, params.n_row);
    allocate(coef, params.n_col);
    allocate(coef2, params.n_col);
    allocate(coef3, params.n_col);
    allocate(coef_ref, params.n_col);
    allocate(coef2_ref, params.n_col);
    allocate(coef3_ref, params.n_col);
    allocate(pred_data, len2);
    allocate(pred, params.n_row_2);
    allocate(pred_ref, params.n_row_2);
    allocate(pred2, params.n_row_2);
    allocate(pred2_ref, params.n_row_2);
    allocate(pred3, params.n_row_2);
    allocate(pred3_ref, params.n_row_2);
    T alpha = params.alpha;

    T data_h[len] = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    updateDevice(data, data_h, len, stream);

    T labels_h[params.n_row] = {0.0, 0.1, 1.0};
    updateDevice(labels, labels_h, params.n_row, stream);

    T coef_ref_h[params.n_col] = {0.39999998, 0.4};
    updateDevice(coef_ref, coef_ref_h, params.n_col, stream);

    T coef2_ref_h[params.n_col] = {0.3454546, 0.34545454};
    updateDevice(coef2_ref, coef2_ref_h, params.n_col, stream);

    T coef3_ref_h[params.n_col] = {0.3799999, 0.38000008};
    updateDevice(coef3_ref, coef3_ref_h, params.n_col, stream);

    T pred_data_h[len2] = {0.5, 2.0, 0.2, 1.0};
    updateDevice(pred_data, pred_data_h, len2, stream);

    T pred_ref_h[params.n_row_2] = {0.28, 1.1999999};
    updateDevice(pred_ref, pred_ref_h, params.n_row_2, stream);

    T pred2_ref_h[params.n_row_2] = {0.37818184, 1.1727273};
    updateDevice(pred2_ref, pred2_ref_h, params.n_row_2, stream);

    T pred3_ref_h[params.n_row_2] = {0.37933332, 1.2533332};
    updateDevice(pred3_ref, pred3_ref_h, params.n_row_2, stream);

    intercept = T(0);

    ridgeFit(handle.getImpl(), data, params.n_row, params.n_col, labels, &alpha,
             1, coef, &intercept, false, false, stream, params.algo);

    ridgePredict(handle.getImpl(), pred_data, params.n_row_2, params.n_col,
                 coef, intercept, pred, stream);

    updateDevice(data, data_h, len, stream);
    updateDevice(labels, labels_h, params.n_row, stream);

    intercept2 = T(0);
    ridgeFit(handle.getImpl(), data, params.n_row, params.n_col, labels, &alpha,
             1, coef2, &intercept2, true, false, stream, params.algo);

    ridgePredict(handle.getImpl(), pred_data, params.n_row_2, params.n_col,
                 coef2, intercept2, pred2, stream);

    updateDevice(data, data_h, len, stream);
    updateDevice(labels, labels_h, params.n_row, stream);

    intercept3 = T(0);
    ridgeFit(handle.getImpl(), data, params.n_row, params.n_col, labels, &alpha,
             1, coef3, &intercept3, true, true, stream, params.algo);

    ridgePredict(handle.getImpl(), pred_data, params.n_row_2, params.n_col,
                 coef3, intercept3, pred3, stream);
  }

  void basicTest2() {
    params = ::testing::TestWithParam<RidgeInputs<T>>::GetParam();
    int len = params.n_row * params.n_col;

    allocate(data_sc, len);
    allocate(labels_sc, len);
    allocate(coef_sc, 1);
    allocate(coef_sc_ref, 1);

    std::vector<T> data_h = {1.0, 1.0, 2.0, 2.0, 1.0, 2.0};
    data_h.resize(len);
    updateDevice(data_sc, data_h.data(), len, stream);

    std::vector<T> labels_h = {6.0, 8.0, 9.0, 11.0, -1.0, 2.0};
    labels_h.resize(len);
    updateDevice(labels_sc, labels_h.data(), len, stream);

    std::vector<T> coef_sc_ref_h = {1.8};
    coef_sc_ref_h.resize(1);
    updateDevice(coef_sc_ref, coef_sc_ref_h.data(), 1, stream);

    T intercept_sc = T(0);
    T alpha_sc = T(1.0);

    ridgeFit(handle.getImpl(), data_sc, len, 1, labels_sc, &alpha_sc, 1,
             coef_sc, &intercept_sc, true, false, stream, params.algo);
  }

  void SetUp() override {
    CUDA_CHECK(cudaStreamCreate(&stream));
    handle.setStream(stream);
    basicTest();
    basicTest2();
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(labels));
    CUDA_CHECK(cudaFree(coef));
    CUDA_CHECK(cudaFree(coef_ref));
    CUDA_CHECK(cudaFree(coef2));
    CUDA_CHECK(cudaFree(coef2_ref));
    CUDA_CHECK(cudaFree(coef3));
    CUDA_CHECK(cudaFree(coef3_ref));
    CUDA_CHECK(cudaFree(pred_data));
    CUDA_CHECK(cudaFree(pred));
    CUDA_CHECK(cudaFree(pred_ref));
    CUDA_CHECK(cudaFree(pred2));
    CUDA_CHECK(cudaFree(pred2_ref));
    CUDA_CHECK(cudaFree(pred3));
    CUDA_CHECK(cudaFree(pred3_ref));

    CUDA_CHECK(cudaFree(data_sc));
    CUDA_CHECK(cudaFree(labels_sc));
    CUDA_CHECK(cudaFree(coef_sc));
    CUDA_CHECK(cudaFree(coef_sc_ref));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

 protected:
  RidgeInputs<T> params;
  T *data, *labels, *coef, *coef_ref, *pred_data, *pred, *pred_ref;
  T *coef2, *coef2_ref, *pred2, *pred2_ref;
  T *coef3, *coef3_ref, *pred3, *pred3_ref;
  T *data_sc, *labels_sc, *coef_sc, *coef_sc_ref;
  T intercept, intercept2, intercept3;
  cumlHandle handle;
  cudaStream_t stream;
};

const std::vector<RidgeInputs<float>> inputsf2 = {{0.001f, 3, 2, 2, 0, 0.5f},
                                                  {0.001f, 3, 2, 2, 1, 0.5f}};

const std::vector<RidgeInputs<double>> inputsd2 = {{0.001, 3, 2, 2, 0, 0.5},
                                                   {0.001, 3, 2, 2, 1, 0.5}};

typedef RidgeTest<float> RidgeTestF;
TEST_P(RidgeTestF, Fit) {
  ASSERT_TRUE(devArrMatch(coef_ref, coef, params.n_col,
                          CompareApproxAbs<float>(params.tol)));

  ASSERT_TRUE(devArrMatch(coef2_ref, coef2, params.n_col,
                          CompareApproxAbs<float>(params.tol)));

  ASSERT_TRUE(devArrMatch(coef3_ref, coef3, params.n_col,
                          CompareApproxAbs<float>(params.tol)));

  ASSERT_TRUE(devArrMatch(pred_ref, pred, params.n_row_2,
                          CompareApproxAbs<float>(params.tol)));

  ASSERT_TRUE(devArrMatch(pred2_ref, pred2, params.n_row_2,
                          CompareApproxAbs<float>(params.tol)));

  ASSERT_TRUE(devArrMatch(pred3_ref, pred3, params.n_row_2,
                          CompareApproxAbs<float>(params.tol)));

  ASSERT_TRUE(
    devArrMatch(coef_sc_ref, coef_sc, 1, CompareApproxAbs<float>(params.tol)));
}

typedef RidgeTest<double> RidgeTestD;
TEST_P(RidgeTestD, Fit) {
  ASSERT_TRUE(devArrMatch(coef_ref, coef, params.n_col,
                          CompareApproxAbs<double>(params.tol)));

  ASSERT_TRUE(devArrMatch(coef2_ref, coef2, params.n_col,
                          CompareApproxAbs<double>(params.tol)));

  ASSERT_TRUE(devArrMatch(coef3_ref, coef3, params.n_col,
                          CompareApproxAbs<double>(params.tol)));

  ASSERT_TRUE(devArrMatch(pred_ref, pred, params.n_row_2,
                          CompareApproxAbs<double>(params.tol)));

  ASSERT_TRUE(devArrMatch(pred2_ref, pred2, params.n_row_2,
                          CompareApproxAbs<double>(params.tol)));

  ASSERT_TRUE(devArrMatch(pred3_ref, pred3, params.n_row_2,
                          CompareApproxAbs<double>(params.tol)));

  ASSERT_TRUE(
    devArrMatch(coef_sc_ref, coef_sc, 1, CompareApproxAbs<double>(params.tol)));
}

INSTANTIATE_TEST_CASE_P(RidgeTests, RidgeTestF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(RidgeTests, RidgeTestD, ::testing::ValuesIn(inputsd2));

}  // namespace GLM
}  // end namespace ML
