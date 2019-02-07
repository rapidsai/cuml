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

#ifdef VERBOSE_LOG
#include <iostream>
#endif

#include <glm/glm_batch_gradient.h>
#include "linalg/cublas_wrappers.h"
#include "linalg/unary_op.h"
#include "random/rng.h"


namespace ML {
using namespace ML::GLM;

struct GradcheckParam {
  double tol;
  double l2_penalty;
  int N;
  int D;
  double h;
};

template <typename T>
class GradTest : public ::testing::TestWithParam<GradcheckParam> {
public:
  void SetUp() override {
    T one = 1.0, zero = 0.0;

    param = ::testing::TestWithParam<GradcheckParam>::GetParam();
    N = param.N;
    D = param.D;

    cublasCreate(&cublas);
    cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_HOST);

    Random::Rng<T> r(1234567890);

    allocate(w, D);
    allocate(X, D * N);
    allocate(y, N);

    allocate(eta, N);

    r.normal(w, D, 0.0, 0.2);
    r.normal(X, D * N, 0.0, 0.5);

    LinAlg::cublasgemv(cublas, CUBLAS_OP_N, N, D, &one, X, N, w, 1, &zero, eta,
                       1);


    // generate class labels by thresholding the logit
    auto threshold = [] __device__(T eta) { return eta > T(0.0); };

    LinAlg::unaryOp(y, eta, N, threshold);

    allocate(loss_val, 1);
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(X));
    CUDA_CHECK(cudaFree(y));
    CUDA_CHECK(cudaFree(eta));
    CUDA_CHECK(cudaFree(w));
    CUDA_CHECK(cudaFree(loss_val));
  }

  void run() {
    LogisticLoss<T> logreg(X, y, eta, N, D, false, param.l2_penalty);
    T err_log = run(logreg);


    LogisticLoss<T, COL_MAJOR> logreg1(X, y, eta, N, D, false,
                                        param.l2_penalty);
    T err_log1 = run(logreg1);

    SquaredLoss<T> leastsq(X, y, eta, N, D, false, param.l2_penalty);
    T err_lin = run(leastsq);

    printf("%f %f\n", err_log, err_lin);
    EXPECT_LT(err_log, param.tol);
    EXPECT_LT(err_lin, param.tol);
  }
  template <typename LossFunction>
  T run(LossFunction &loss) {
    SimpleVec<T> grad(loss.n_param), w_t(loss.n_param);


    allocate(loss_val, 1);


    std::vector<T> grad_num(loss.n_param, 0);
    std::vector<T> grad_host(loss.n_param, 0);


    loss(w_t, grad);
    updateHost(&grad_host[0], grad.data, grad_host.size());

    numeric_grad(loss, X, y, w_t.data, &grad_num[0], loss_val, eta, (T)param.h);
    T abs_err = 0;
    T l1norm = 0;
    for (int it = 0; it < grad_host.size(); it++) {
      printf("Grad_w[%d]: %f %f\n", it, grad_host[it], grad_num[it]);
      abs_err += abs(grad_host[it] - grad_num[it]);
      l1norm += abs(grad_host[it]);
    }


    return abs_err / (l1norm + 1e-6);
  }

protected:
  GradcheckParam param;
  cublasHandle_t cublas;
  T *X = 0;
  T *y = 0;
  T *w = 0;
  T *eta = 0;
  T *loss_val = 0;
  int N, D;
};


const std::vector<GradcheckParam> inputsD = {
  GradcheckParam{.tol = 1e-6, .l2_penalty = 0.0, .N = 10, .D = 5, .h = 1e-8}};

const std::vector<GradcheckParam> inputsF = {
  GradcheckParam{.tol = 5e-3, .l2_penalty = 0.0, .N = 10, .D = 5, .h = 1e-4}};


typedef GradTest<double> GradTestD;
typedef GradTest<float> GradTestF;

TEST_P(GradTestD, GradTestD) { run(); }

INSTANTIATE_TEST_CASE_P(glm_gradtestsD, GradTestD, ::testing::ValuesIn(inputsD));


TEST_P(GradTestF, GradTestF) { run(); }

INSTANTIATE_TEST_CASE_P(glm_gradtestsF, GradTestF, ::testing::ValuesIn(inputsF));

}; // namespace ML
