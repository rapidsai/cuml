/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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
#include <memory>
#include <vector>


#include <glm/glm_batch_gradient.h>
#include <glm/glm_vectors.h>
#include "linalg/cublas_wrappers.h"
#include "linalg/transpose.h"
#include "linalg/unary_op.h"
#include "random/rng.h"
#include "test_utils.h"


namespace ML {
using namespace MLCommon;

using namespace ML::GLM;

template <typename T>
struct VectestParam {
  T tol;
  int N;
  int D;
};

class Vectest : public ::testing::Test {
public:
  virtual void SetUp() {
    CUBLAS_CHECK(cublasCreate(&cublas));
  }

  virtual void TearDown() {
    CUBLAS_CHECK(cublasDestroy(cublas));
  }

  cublasHandle_t cublas;
};


template <typename T>
class VectestP : public ::testing::TestWithParam<VectestParam<T>> {
public:
  typedef SimpleVec<T> Vec;
  typedef SimpleMat<T> Mat;

  void SetUp() override {
    cublasCreate(&cublas);
    cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_HOST);
  }

  void TearDown() override {}

  void gemvTest() {
    VectestParam<T> param =
      ::testing::TestWithParam<VectestParam<T>>::GetParam();

    int N = param.N, D = param.D;

    T one = 1, zero = 0;

    Random::Rng<T> r(1234567890);

    SimpleMat<T, COL_MAJOR> Acm(N, D);
    SimpleMat<T, ROW_MAJOR> Arm(N, D);

    Vec wD(D);
    Vec wN(N);

    r.normal(Acm.data, Acm.len, 0, 1);

    LinAlg::transpose(Acm.data, Arm.data, N, D, cublas);


    r.normal(wD.data, wD.len, 0, 1);
    r.normal(wN.data, wN.len, 0, 1);

    Vec res_cm(N);
    Vec res_rm(N);
    Vec resT_cm(D);
    Vec resT_rm(D);

    Vec ref_cm(N);
    Vec ref_rm(N);
    Vec refT_cm(D);
    Vec refT_rm(D);

    LinAlg::cublasgemv(cublas, CUBLAS_OP_N, N, D, &one, Acm.data, N, wD.data, 1,
                       &zero, ref_cm.data, 1);

    LinAlg::cublasgemv(cublas, CUBLAS_OP_T, N, D, &one, Acm.data, N, wN.data, 1,
                       &zero, refT_cm.data, 1);


    res_cm.assign_gemv(one, Acm, wD, zero, cublas);
    res_rm.assign_gemv(one, Arm, wD, zero, cublas);

    resT_cm.assign_gemvT(one, Acm, wN, zero, cublas);
    resT_rm.assign_gemvT(one, Arm, wN, zero, cublas);

    printf("tolera %f\n", param.tol);
    ASSERT_TRUE(devArrMatch(ref_cm.data, res_cm.data, res_cm.len,
                            CompareApprox<T>(param.tol)));

    ASSERT_TRUE(devArrMatch(ref_cm.data, res_rm.data, res_rm.len,
                            CompareApprox<T>(param.tol)));

    ASSERT_TRUE(devArrMatch(refT_cm.data, resT_cm.data, resT_cm.len,
                            CompareApprox<T>(param.tol)));

    ASSERT_TRUE(devArrMatch(refT_cm.data, resT_rm.data, resT_rm.len,
                            CompareApprox<T>(param.tol)));
  }

  VectestParam<T> param;
  cublasHandle_t cublas;
};

template <typename T> T tol();

template <typename T>
std::vector<VectestParam<T>> test_cases() {
  std::vector<VectestParam<T>> ret;
  std::vector<int> dims = {1, 2, 10, 10000};

  int n_iters = 2;

  for (auto N : dims) {
    for (auto D : dims) {
      for (int it = 0; it < n_iters; it++) {
        VectestParam<T> tmp;
        tmp.N = N;
        tmp.D = D;
        tmp.tol = tol<T>() * sqrt(std::max(T(D), T(N)));
        ret.push_back(tmp);
      }
    }
  }
  return ret;
}


typedef VectestP<double> VectestD;
typedef VectestP<float> VectestF;

TEST_P(VectestD, glm_gemvTestD) { gemvTest(); }

INSTANTIATE_TEST_CASE_P(glm_vectestsD, VectestD,
                        ::testing::ValuesIn(test_cases<double>()));


TEST_P(VectestF, glm_gemvTestF) { gemvTest(); }

INSTANTIATE_TEST_CASE_P(glm_vectestsF, VectestF,
                        ::testing::ValuesIn(test_cases<float>()));

TEST_F(Vectest, glm_vectest_row_vs_col){
    constexpr bool debug=false;
    int N = 3, D = 4;

    std::vector<double> host_data_rm = {1,2,3,4,5,6,7,8,9,10,11,12};
    std::vector<double> host_data_cm = {1,5,9,2,6,10,3,7,11,4,8,12};
    std::vector<double> host_vec_D = {1,2,3,4};
    std::vector<double> host_vec_N = {1,2,3};

    SimpleMat<double, COL_MAJOR> Acm(N,D);
    SimpleMat<double, ROW_MAJOR> Arm(N,D);

    SimpleVec<double> wD(D);
    SimpleVec<double> wN(N);

    SimpleVec<double> resDcm(D);
    SimpleVec<double> resNcm(N);

    SimpleVec<double> resDrm(D);
    SimpleVec<double> resNrm(N);

    updateDevice(Acm.data, &host_data_cm[0], Acm.len);
    updateDevice(Arm.data, &host_data_rm[0], Arm.len);
    updateDevice(wD.data, &host_vec_D[0],wD.len);
    updateDevice(wN.data, &host_vec_N[0],wN.len);
    resNcm.assign_gemv(1.0, Acm, wD, 0.0, cublas);
    resDcm.assign_gemvT(1.0, Acm, wN, 0.0, cublas);

    resNrm.assign_gemv(1.0, Arm, wD, 0.0, cublas);
    resDrm.assign_gemvT(1.0, Arm, wN, 0.0, cublas);

    ASSERT_TRUE(devArrMatch(resNrm.data, resNcm.data, N, CompareApprox<double>(1e-6)));
    ASSERT_TRUE(devArrMatch(resDrm.data, resDcm.data, D, CompareApprox<double>(1e-6)));

    SimpleVec<double> eta(N), grad(D);
    SquaredLoss<double, ROW_MAJOR> sl_rm(Arm.data, resNrm.data, eta.data, N,D, false, 0.1);

    SquaredLoss<double, COL_MAJOR> sl_cm(Acm.data, resNcm.data, eta.data, N,D, false, 0.1);

    SimpleVec<double> wD0(D, 0);
    wD0.fill(0.0);
    double frm = sl_rm(wD0, grad);
    double fcm = sl_cm(wD0, grad);
    ASSERT_LT(abs(frm - fcm), 1e-6);

    if (debug) {
      Acm.print();
      Arm.print();

      resDcm.print();
      resDrm.print();
      resNcm.print();
      resNrm.print();
      printf("%f %f\n", frm, fcm);
    }
}

}; // namespace ML
