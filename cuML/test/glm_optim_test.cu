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
#include <iostream>
#include <memory>

#include <glm/qn_c.h>
#include "linalg/cublas_wrappers.h"
#include "linalg/transpose.h"
#include "linalg/unary_op.h"
#include "random/rng.h"

#include <glm/gradient_descent.h>
#include <glm/lbfgs.h>
#include <glm/glm_vectors.h>

#include <glm/glm_batch_gradient.h>

using namespace MLCommon;

// we link this against the dispatch method of the c api
template <typename T, typename LossFunction>
int fit_dispatch(T *X, T *y, int N, int D, bool has_bias, T l1, T l2,
                 int max_iter, T grad_tol, T value_rel_tol,
                 int linesearch_max_iter, int lbfgs_memory, int verbosity,
                 T *w0, // initial value and result
                 T *fx, int *num_iters);


namespace ML {

using namespace ML::GLM;

template <typename T>
struct OptimTestParam {
  int N;
  int D;
  T l1;
  T l2;
  int loss; // 0:logistic,1:squared
  bool has_bias;
  int run;
  T tol;
  int verbosity;
};

template <typename T>
void PrintTo(const OptimTestParam<T> &param, std::ostream *os) {
  *os << "N=" << param.N << ";D=" << param.D << ";l1=" << param.l1
      << ";l2=" << param.l2 << ";loss_fun=" << param.loss
      << ";has_bias=" << param.has_bias << ";run=" << param.run
      << ";tol=" << param.tol;
}

template <typename T>
class OptimTest : public ::testing::TestWithParam<OptimTestParam<T>> {
protected:
  typedef SimpleVec<T> Vector;
  typedef SimpleMat<T> Matrix;
  cublasHandle_t cublas;

  norm1<T> l1norm;
  norm2<T> l2norm;

  T one = 1.0, zero = 0.0;


public:
  typedef SimpleVec<T> vec;

  void SetUp() override {
    cublasCreate(&cublas);
    cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_HOST);
  }

  void TearDown() override {}

  void run_test() {
    OptimTestParam<T> params =
      ::testing::TestWithParam<OptimTestParam<T>>::GetParam();
    int N = params.N;
    int D = params.D;
    T l1 = params.l1;
    T l2 = params.l2;
    bool has_bias = params.has_bias;
    int verbosity = params.verbosity;

    op_pseudo_grad<T> pseudo_grad(l1);

    // Generate test data
    Matrix X(N, D);
    Matrix XT(N, D, 0);
    Vector y(N);
    Vector eta(N);
    Vector w(D + has_bias);


    // For various parameters, fit
    Random::Rng<T> r(params.run);

    r.normal(X.data, X.len, T(0), T(1.0 / sqrt(T(D))));
    r.normal(w.data, w.len, T(0), T(1.0));
    cudaThreadSynchronize();

    LinAlg::transpose(X.data, XT.data, N, D, cublas);
    cudaThreadSynchronize();

    LinAlg::cublasgemv(cublas, CUBLAS_OP_N, N, D, &one, X.data, N, w.data, 1,
                       &zero, eta.data, 1);

    // generate class labels by thresholding the logit
    auto toprob = [] __device__(T z_) { return 1.0 / (1.0 + myExp(-z_)); };
    LinAlg::unaryOp(eta.data, eta.data, N, toprob);

    T *yptr = y.data;

    uint64_t offset = 0;
    // Bernoulli supports drawing n samples with fixed p, not with different p_i
    auto rr = [yptr] __device__(T val, unsigned idx) {
      return T(val > yptr[idx]);
    };
    MLCommon::Random::randImpl<T, T, decltype(rr)>(
      offset, yptr, N, rr, 256, ceildiv(N, 256), MLCommon::Random::GenPhilox);

    cudaThreadSynchronize();

    Vector wtest(w.len);
    Vector grad(w.len);
    Vector pseudo(w.len);

    int max_iter = 10000;
    T grad_tol = params.tol;
    T value_rel_tol = 1e-5;
    int linesearch_max_iter = 50;
    int lbfgs_memory = max(5, int(0.1 * D));
    int num_iters = 0;

    T gnorm, xnorm;

    T fx = 0;

    if (params.loss == 1) {
      SquaredLoss<T> loss_fun(X.data, y.data, eta.data, N, D, has_bias, l2);

      T fx_rm, fx_cm;

      wtest.fill(0);
      fit_dispatch<T, SquaredLoss<T, ROW_MAJOR>>(
        XT.data, y.data, N, D, has_bias, l1, l2, max_iter, grad_tol,
        value_rel_tol, linesearch_max_iter, lbfgs_memory, verbosity,
        wtest.data, // initial value and result
        &fx_rm, &num_iters);


      wtest.fill(0);
      fit_dispatch<T, SquaredLoss<T, COL_MAJOR>>(
        X.data, y.data, N, D, has_bias, l1, l2, max_iter, grad_tol,
        value_rel_tol, linesearch_max_iter, lbfgs_memory, verbosity,
        wtest.data, // initial value and result
        &fx_cm, &num_iters);

      fx = loss_fun(wtest, grad);

      ASSERT_LT(abs(fx_rm - fx_cm),
                params.tol * sqrt(std::max(T(N), T(D))));
      ASSERT_LT(abs(fx_rm - fx_cm), params.tol * sqrt(std::max(T(N), T(D))));
    } else {

      LogisticLoss<T, COL_MAJOR> loss_fun(X.data, y.data, eta.data, N, D, has_bias, l2);

      T fx_rm, fx_cm;

      wtest.fill(0);
      fit_dispatch<T, LogisticLoss<T, ROW_MAJOR>>(
        XT.data, y.data, N, D, has_bias, l1, l2, max_iter, grad_tol,
        value_rel_tol, linesearch_max_iter, lbfgs_memory, verbosity,
        wtest.data, // initial value and result
        &fx_rm, &num_iters);


      wtest.fill(0);
      fit_dispatch<T, LogisticLoss<T, COL_MAJOR>>(
        X.data, y.data, N, D, has_bias, l1, l2, max_iter, grad_tol,
        value_rel_tol, linesearch_max_iter, lbfgs_memory, verbosity,
        wtest.data, // initial value and result
        &fx_cm, &num_iters);

      fx = loss_fun(wtest, grad);

      ASSERT_LT(abs(fx_rm - fx_cm),
                params.tol * sqrt(std::max(T(N), T(D))));
      ASSERT_LT(abs(fx_rm - fx_cm), params.tol * sqrt(std::max(T(N), T(D))));
    }

    if (l1 > 0) {
      grad.assign_binary(wtest, grad, pseudo_grad);
      fx += l1norm(wtest) * l1;
    }

    gnorm = l2norm(grad);
    xnorm = l2norm(wtest);
    ASSERT_LT(gnorm, params.tol * std::max(T(1.0), xnorm));
  }
};

template <typename T>
T tol() {
  return 1e-6;
};
template <>
float tol<float>() {
  return 1e-4;
};

template <typename T>
std::vector<OptimTestParam<T>> test_cases() {
  std::vector<OptimTestParam<T>> params;
  std::vector<int> Ns = {100, 1000};
  //std::vector<int> Ds = {1, 10, 100, 1000};
  std::vector<int> Ds = {1, 10, 100};

  std::vector<int> losses = {0, 1};
  std::vector<bool> has_biases = {false, true};

  std::vector<T> l1s = {0.0, 0.01, 0.1, 1.0};
  std::vector<T> l2s = {0.0, 0.01, 0.1, 1.0};

  T eps = tol<T>();

  int n_iters = 1;
  int verbosity = 0;

  /*
     n_iters = 10;
     verbosity=1;
     Ns = {1000};
     Ds = {1000};
     has_biases ={false};
     losses ={1};
     l1s={0.0};
     l2s={0.0};
     */

  for (auto loss : losses) {
    for (auto N : Ns) {
      for (auto D : Ds) {
        for (T l2 : l2s) {
          for (T l1 : l1s) {
            for (auto has_bias : has_biases) {
              for (int run = 0; run < n_iters; run++) {
                OptimTestParam<T> tmp;
                tmp.N = N;
                tmp.D = D;
                tmp.l1 = l1;
                tmp.l2 = l2;
                tmp.loss = loss;
                tmp.has_bias = has_bias;
                tmp.run = run;
                tmp.tol = tol<T>();
                tmp.verbosity = verbosity;
                params.push_back(tmp);
              }
            }
          }
        }
      }
    }
  }
  return params;
}

typedef OptimTest<double> OptimTestD;
typedef OptimTest<float> OptimTestF;

TEST_P(OptimTestD, QuasiNewtonD) { run_test(); }

TEST_P(OptimTestF, QuasiNewtonF) { run_test(); }

INSTANTIATE_TEST_CASE_P(glm_optimtestsD, OptimTestD,
                        ::testing::ValuesIn(test_cases<double>()));

INSTANTIATE_TEST_CASE_P(glm_optimtestsF, OptimTestF,
                        ::testing::ValuesIn(test_cases<float>()));

}; // namespace ML
